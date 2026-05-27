from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from goalflow.data import TrackCatalog, as_text, normalize_text
from goalflow.embeddings import TRACK_EMBEDDING_CHANNELS, TrackEmbeddingStore, UserEmbeddingStore
from goalflow.pipeline import GoalFlowConfig, default_index_weights, default_query_weights
from goalflow.state import ConversationState, build_state_for_dev_turn


EXPERIMENT_DIR = Path("goalflow_musiccrs/experiments/rerank_v2_independent_features")
OLD300_PKL = "goalflow_musiccrs/cache/ltr_candidate_frames/dev_7d9af67ef612.pkl"
MATRIX_DIR = "goalflow_musiccrs/experiments/source_candidate_matrix_full_s20_k800/source_candidate_matrix"
BEAM800_CHOICE = (
    "goalflow_musiccrs/experiments/source_candidate_matrix_full_s20_k800/"
    "source_candidate_matrix/beam_search_target800_strict/best_choice.tsv"
)
META_COLUMNS = {
    "group_id",
    "session_id",
    "user_id",
    "turn_number",
    "track_id",
    "label",
    "gold_track_id",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "for",
    "from",
    "give",
    "have",
    "i",
    "in",
    "is",
    "it",
    "me",
    "more",
    "music",
    "of",
    "on",
    "or",
    "play",
    "please",
    "recommend",
    "song",
    "songs",
    "some",
    "that",
    "the",
    "this",
    "to",
    "track",
    "want",
    "with",
    "you",
}
SOURCE_PREFIXES = (
    "rrf_score",
    "source_count",
    "best_source_rank",
    "best_source_recip_rank",
    "best_source_bucket",
    "rank_",
    "hit20_rank_",
    "hit100_rank_",
    "source_present_",
    "source_rank_",
    "source_recip_rank_",
    "source_bucket_",
    "old_rrf",
    "beam_rrf",
    "old_source",
    "beam_source",
    "old_best",
    "beam_best",
    "is_old_pool_candidate",
    "is_beam_pool_candidate",
    "is_in_both_old_and_beam",
    "is_nextgen_candidate",
    "nextgen_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank v2 with independent relevance features.")
    parser.add_argument("--project-root", default="goalflow_musiccrs")
    parser.add_argument("--out-dir", default=str(EXPERIMENT_DIR))
    parser.add_argument("--old300-pkl", default=OLD300_PKL)
    parser.add_argument("--matrix-dir", default=MATRIX_DIR)
    parser.add_argument("--beam800-choice", default=BEAM800_CHOICE)
    parser.add_argument("--pools", default="old300")
    parser.add_argument("--feature-sets", default="all")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--lambdarank-truncation-level", type=int, default=50)
    parser.add_argument("--objective", choices=["lambdarank", "rank_xendcg"], default="lambdarank")
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--embedding-channels",
        default="metadata,attributes,lyrics,audio,image,track_cf",
        help="Comma-separated channel names from goalflow.embeddings, or empty to disable.",
    )
    parser.add_argument("--disable-user-cf", action="store_true")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--max-union-size", type=int, default=0)
    parser.add_argument("--custom-pool-name", default="nextgen")
    parser.add_argument("--custom-pool-pkl", default="")
    return parser.parse_args()


def stable_hash_text(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:n]


def stable_json_hash(payload: Any, n: int = 12) -> str:
    return stable_hash_text(json.dumps(payload, sort_keys=True, default=str), n=n)


def tokenize(text: Any) -> set[str]:
    raw = as_text(text).lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9']+", raw)
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def token_list(text: Any) -> list[str]:
    raw = as_text(text).lower()
    return [token for token in re.findall(r"[a-z0-9][a-z0-9']+", raw) if token not in STOPWORDS and len(token) > 1]


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_count(a: set[str], b: set[str]) -> int:
    if not a or not b:
        return 0
    return len(a & b)


def weighted_overlap(a: set[str], b: set[str], idf: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    return float(sum(idf.get(token, 1.0) for token in a & b))


def normalized_contains(needle: str, haystack: str) -> int:
    return int(bool(needle and len(needle) >= 3 and needle in haystack))


def rank_bucket(rank: float) -> int:
    if rank <= 0 or rank >= 9999:
        return -1
    if rank <= 50:
        return 0
    if rank <= 100:
        return 1
    if rank <= 200:
        return 2
    if rank <= 400:
        return 3
    return 4


def load_dev_states(config: GoalFlowConfig) -> list[ConversationState]:
    dataset = load_dataset(config.conversation_dataset_name, split="test")
    states: list[ConversationState] = []
    for item in tqdm(dataset, desc="Build dev states"):
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id:
                states.append(state)
    return states


def state_key(state: ConversationState) -> str:
    return f"{state.session_id}::{state.user_id}::{state.turn_number}"


def fold_for_state(state: ConversationState, folds: int) -> int:
    return int(hashlib.sha1(state.session_id.encode("utf-8")).hexdigest()[:8], 16) % folds


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_meta(matrix_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    with (matrix_dir / "meta.txt").open(encoding="utf-8") as f:
        for line in f:
            key, value = line.rstrip("\n").split("\t", 1)
            if key in {"num_turns", "num_sources", "num_tracks", "max_k"}:
                meta[key] = int(value)
            elif key == "k_values":
                meta[key] = [int(item) for item in value.split()]
            else:
                meta[key] = value
    return meta


def read_choice(path: Path, source_name_to_index: dict[str, int]) -> dict[int, int]:
    choice: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            choice[source_name_to_index[row["source_name"]]] = int(row["selected_k"])
    return choice


def source_weight(source_name: str) -> float:
    if ":" not in source_name:
        return 1.0
    index_name, query_name = source_name.split(":", 1)
    return default_index_weights().get(index_name, 1.0) * default_query_weights().get(query_name, 1.0)


class TrackTextCache:
    def __init__(self, catalog: TrackCatalog):
        self.catalog = catalog
        self.track_ids = catalog.track_ids
        self.title_tokens: dict[str, set[str]] = {}
        self.artist_tokens: dict[str, set[str]] = {}
        self.album_tokens: dict[str, set[str]] = {}
        self.tag_tokens: dict[str, set[str]] = {}
        self.metadata_tokens: dict[str, set[str]] = {}
        self.title_norm: dict[str, str] = {}
        self.artist_norm: dict[str, str] = {}
        self.album_norm: dict[str, str] = {}
        self.release_year: dict[str, int] = {}
        self.popularity: dict[str, float] = {}
        self.duration: dict[str, float] = {}
        self.idf: dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        df = Counter()
        for track_id in tqdm(self.track_ids, desc="Precompute track text"):
            view = self.catalog.view(track_id)
            title = tokenize(view.track_name)
            artist = tokenize(view.artist_name)
            album = tokenize(view.album_name)
            tags = tokenize(view.tag_list)
            metadata = title | artist | album | tags | tokenize(view.release_date)
            self.title_tokens[track_id] = title
            self.artist_tokens[track_id] = artist
            self.album_tokens[track_id] = album
            self.tag_tokens[track_id] = tags
            self.metadata_tokens[track_id] = metadata
            self.title_norm[track_id] = normalize_text(view.track_name)
            self.artist_norm[track_id] = normalize_text(view.artist_name)
            self.album_norm[track_id] = normalize_text(view.album_name)
            self.release_year[track_id] = self.catalog.release_year(track_id) or 0
            self.popularity[track_id] = float(view.popularity or 0.0)
            self.duration[track_id] = float(view.duration or 0.0)
            for token in metadata:
                df[token] += 1
        n = max(len(self.track_ids), 1)
        self.idf = {token: math.log((n + 1) / (count + 1)) + 1.0 for token, count in df.items()}


def query_context(state: ConversationState, catalog: TrackCatalog) -> dict[str, Any]:
    user_history = []
    pos_text = []
    neg_text = []
    previous_artists = Counter()
    accepted_artists = Counter()
    rejected_artists = Counter()
    previous_albums = Counter()
    accepted_albums = Counter()
    rejected_albums = Counter()
    pos_tags: set[str] = set()
    neg_tags: set[str] = set()
    pos_years = []
    neg_years = []
    for turn in state.history_turns:
        role = turn.get("role")
        content = as_text(turn.get("content"))
        if role == "user":
            user_history.append(content)
    for track_id in state.previous_music_track_ids:
        if catalog.has_track(track_id):
            artist = catalog.normalized_field(track_id, "artist_name")
            album = catalog.normalized_field(track_id, "album_name")
            previous_artists[artist] += 1
            previous_albums[album] += 1
    for track_id in state.positive_seed_ids:
        if catalog.has_track(track_id):
            pos_text.append(catalog.compact_summary(track_id))
            artist = catalog.normalized_field(track_id, "artist_name")
            album = catalog.normalized_field(track_id, "album_name")
            accepted_artists[artist] += 1
            accepted_albums[album] += 1
            pos_tags |= catalog.tag_words(track_id)
            year = catalog.release_year(track_id)
            if year:
                pos_years.append(year)
    for track_id in state.negative_seed_ids:
        if catalog.has_track(track_id):
            neg_text.append(catalog.compact_summary(track_id))
            artist = catalog.normalized_field(track_id, "artist_name")
            album = catalog.normalized_field(track_id, "album_name")
            rejected_artists[artist] += 1
            rejected_albums[album] += 1
            neg_tags |= catalog.tag_words(track_id)
            year = catalog.release_year(track_id)
            if year:
                neg_years.append(year)
    current = state.current_user_query
    goal = state.listener_goal
    history_text = "\n".join(user_history)
    current_goal_history = "\n".join([current, goal, history_text])
    return {
        "current_text": current,
        "goal_text": goal,
        "history_user_text": history_text,
        "positive_text": "\n".join(pos_text),
        "negative_text": "\n".join(neg_text),
        "current_tokens": tokenize(current),
        "goal_tokens": tokenize(goal),
        "history_tokens": tokenize(history_text),
        "positive_tokens": tokenize("\n".join(pos_text)),
        "negative_tokens": tokenize("\n".join(neg_text)),
        "all_tokens": tokenize(current_goal_history),
        "current_norm": normalize_text(current),
        "goal_norm": normalize_text(goal),
        "history_norm": normalize_text(history_text),
        "all_norm": normalize_text(current_goal_history),
        "previous_artists": previous_artists,
        "accepted_artists": accepted_artists,
        "rejected_artists": rejected_artists,
        "previous_albums": previous_albums,
        "accepted_albums": accepted_albums,
        "rejected_albums": rejected_albums,
        "pos_tags": pos_tags,
        "neg_tags": neg_tags,
        "pos_years": pos_years,
        "neg_years": neg_years,
    }


def load_old300_pool(path: Path, states: list[ConversationState]) -> pd.DataFrame:
    df = pd.read_pickle(path).copy()
    df["gold_track_id"] = [states[int(group_id)].gold_track_id for group_id in df["group_id"].to_numpy()]
    df["is_old_pool_candidate"] = 1
    df["is_beam_pool_candidate"] = 0
    df["is_in_both_old_and_beam"] = 0
    df["old_rrf_score"] = df.get("rrf_score", 0.0)
    df["old_source_count"] = df.get("source_count", 0.0)
    df["old_best_source_rank"] = df.get("best_source_rank", 9999.0)
    df["old_rrf_rank"] = df.groupby("group_id")["old_rrf_score"].rank(method="first", ascending=False)
    df["beam_rrf_score"] = 0.0
    df["beam_source_count"] = 0.0
    df["beam_best_source_rank"] = 9999.0
    df["beam_rrf_rank"] = 9999.0
    return normalize_pool_base(df)


def load_beam800_pool(
    matrix_dir: Path,
    choice_path: Path,
    states: list[ConversationState],
    rrf_k: int,
) -> pd.DataFrame:
    meta = read_meta(matrix_dir)
    source_rows = read_tsv(matrix_dir / "sources.tsv")
    track_rows = read_tsv(matrix_dir / "track_ids.tsv")
    example_rows = read_tsv(matrix_dir / "examples.tsv")
    source_names = [row["source_name"] for row in source_rows]
    source_name_to_index = {row["source_name"]: int(row["source_index"]) for row in source_rows}
    track_ids = [row["track_id"] for row in track_rows]
    choice = read_choice(choice_path, source_name_to_index)
    num_turns = int(meta["num_turns"])
    num_sources = int(meta["num_sources"])
    max_k = int(meta["max_k"])
    candidates = np.memmap(
        matrix_dir / str(meta["candidate_file"]),
        dtype=np.int32,
        mode="r",
        shape=(num_turns, num_sources, max_k),
    )
    counts = np.memmap(
        matrix_dir / str(meta["counts_file"]),
        dtype=np.uint16,
        mode="r",
        shape=(num_turns, num_sources),
    )
    rows: list[dict[str, Any]] = []
    for group_id, state in enumerate(tqdm(states, desc="Build beam800 pool")):
        example = example_rows[group_id]
        if example["session_id"] != state.session_id or int(example["turn_number"]) != state.turn_number:
            raise ValueError(f"Matrix example order mismatch at group {group_id}")
        fused: dict[str, dict[str, Any]] = {}
        for source_index, selected_k in choice.items():
            take = min(int(selected_k), int(counts[group_id, source_index]))
            if take <= 0:
                continue
            source_name = source_names[source_index]
            weight = source_weight(source_name)
            for rank, track_index_raw in enumerate(candidates[group_id, source_index, :take], start=1):
                track_index = int(track_index_raw)
                if track_index < 0:
                    continue
                track_id = track_ids[track_index]
                item = fused.setdefault(
                    track_id,
                    {
                        "group_id": group_id,
                        "session_id": state.session_id,
                        "user_id": state.user_id,
                        "turn_number": state.turn_number,
                        "track_id": track_id,
                        "gold_track_id": state.gold_track_id,
                        "label": 1 if track_id == state.gold_track_id else 0,
                        "rrf_score": 0.0,
                        "source_count": 0,
                        "best_source_rank": 9999,
                    },
                )
                item["rrf_score"] += weight / (rrf_k + rank)
                item["source_count"] += 1
                item["best_source_rank"] = min(item["best_source_rank"], rank)
                item[f"rank_{source_name.replace(':', '_')}"] = rank
        ordered = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
        for rrf_rank, item in enumerate(ordered, start=1):
            item["beam_rrf_rank"] = rrf_rank
            rows.append(item)
    df = pd.DataFrame(rows)
    df["is_old_pool_candidate"] = 0
    df["is_beam_pool_candidate"] = 1
    df["is_in_both_old_and_beam"] = 0
    df["beam_rrf_score"] = df["rrf_score"]
    df["beam_source_count"] = df["source_count"]
    df["beam_best_source_rank"] = df["best_source_rank"]
    df["old_rrf_score"] = 0.0
    df["old_source_count"] = 0.0
    df["old_best_source_rank"] = 9999.0
    df["old_rrf_rank"] = 9999.0
    return normalize_pool_base(df)


def normalize_pool_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["group_id", "turn_number", "label", "is_old_pool_candidate", "is_beam_pool_candidate", "is_in_both_old_and_beam"]:
        if col in df:
            df[col] = df[col].fillna(0).astype(np.int32)
    for col in ["rrf_score", "source_count", "best_source_rank"]:
        if col not in df:
            df[col] = 0.0 if col != "best_source_rank" else 9999.0
    df["best_source_recip_rank"] = 1.0 / (df["best_source_rank"].astype(float) + 60.0)
    df["best_source_bucket"] = df["best_source_rank"].apply(rank_bucket).astype(np.int8)
    rank_cols = [c for c in df.columns if c.startswith("rank_")]
    for col in rank_cols:
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", col.removeprefix("rank_")).strip("_")
        rank = df[col].fillna(9999.0).astype(np.float32)
        df[f"source_rank_{safe}"] = rank
        df[f"source_present_{safe}"] = (rank < 9999.0).astype(np.int8)
        df[f"source_recip_rank_{safe}"] = np.where(rank < 9999.0, 1.0 / (rank + 60.0), 0.0).astype(np.float32)
        df[f"source_bucket_{safe}"] = pd.Series(rank).apply(rank_bucket).astype(np.int8)
    return df


def make_union_pool(old_df: pd.DataFrame, beam_df: pd.DataFrame, max_union_size: int = 0) -> pd.DataFrame:
    old = old_df.copy()
    beam = beam_df.copy()
    old["_pool_order"] = old.groupby("group_id")["old_rrf_score"].rank(method="first", ascending=False)
    beam["_pool_order"] = beam.groupby("group_id")["beam_rrf_score"].rank(method="first", ascending=False)
    combined = pd.concat([old, beam], ignore_index=True, sort=False)
    rank_cols = [c for c in combined.columns if c.startswith("rank_") or c.startswith("source_")]
    agg: dict[str, Any] = {
        "session_id": "first",
        "user_id": "first",
        "turn_number": "first",
        "gold_track_id": "first",
        "label": "max",
        "is_old_pool_candidate": "max",
        "is_beam_pool_candidate": "max",
        "old_rrf_score": "max",
        "beam_rrf_score": "max",
        "old_source_count": "max",
        "beam_source_count": "max",
        "old_best_source_rank": "min",
        "beam_best_source_rank": "min",
        "old_rrf_rank": "min",
        "beam_rrf_rank": "min",
    }
    for col in rank_cols:
        if col.startswith("source_present_"):
            agg[col] = "max"
        elif col.startswith("source_bucket_"):
            agg[col] = "min"
        else:
            agg[col] = "min"
    union = combined.groupby(["group_id", "track_id"], as_index=False).agg(agg)
    union["is_in_both_old_and_beam"] = (
        (union["is_old_pool_candidate"] > 0) & (union["is_beam_pool_candidate"] > 0)
    ).astype(np.int8)
    union["source_count"] = union[["old_source_count", "beam_source_count"]].max(axis=1)
    union["rrf_score"] = union[["old_rrf_score", "beam_rrf_score"]].max(axis=1)
    union["best_source_rank"] = union[["old_best_source_rank", "beam_best_source_rank"]].min(axis=1)
    union["best_source_recip_rank"] = 1.0 / (union["best_source_rank"].astype(float) + 60.0)
    union["best_source_bucket"] = union["best_source_rank"].apply(rank_bucket).astype(np.int8)
    if max_union_size > 0:
        pieces = []
        for _, group in tqdm(union.groupby("group_id", sort=False), desc="Truncate union pool"):
            group = group.copy()
            group["_keep_score"] = (
                group["is_old_pool_candidate"] * 1_000_000.0
                + group["is_in_both_old_and_beam"] * 10_000.0
                + group["old_rrf_score"].fillna(0.0) * 1_000.0
                + group["beam_rrf_score"].fillna(0.0)
            )
            pieces.append(group.sort_values("_keep_score", ascending=False).head(max_union_size).drop(columns=["_keep_score"]))
        union = pd.concat(pieces, ignore_index=True)
    return normalize_pool_base(union)


def pool_summary(pool_name: str, df: pd.DataFrame, states: list[ConversationState]) -> dict[str, Any]:
    sizes = df.groupby("group_id").size()
    gold_in_pool = int(df[df["label"] == 1]["group_id"].nunique())
    rank_cols = [c for c in df.columns if c.startswith("source_present_")]
    source_presence = {}
    for col in rank_cols:
        source_presence[col.removeprefix("source_present_")] = int(df[col].fillna(0).sum())
    top_sources = sorted(source_presence.items(), key=lambda item: item[1], reverse=True)[:25]
    return {
        "pool": pool_name,
        "groups": int(len(sizes)),
        "rows": int(len(df)),
        "group_size_min": int(sizes.min()),
        "group_size_p50": float(sizes.quantile(0.5)),
        "group_size_mean": float(sizes.mean()),
        "group_size_p95": float(sizes.quantile(0.95)),
        "group_size_max": int(sizes.max()),
        "gold_in_pool": gold_in_pool,
        "gold_coverage": gold_in_pool / len(states),
        "candidate_source_coverage_top25": json.dumps(top_sources, ensure_ascii=False),
    }


def build_feature_frame(
    pool_name: str,
    base_df: pd.DataFrame,
    states: list[ConversationState],
    catalog: TrackCatalog,
    text_cache: TrackTextCache,
    track_embeddings: TrackEmbeddingStore | None,
    user_embeddings: UserEmbeddingStore | None,
    embedding_channels: list[str],
) -> pd.DataFrame:
    df = base_df.reset_index(drop=True).copy()
    n = len(df)
    for col in ["intent", "category", "specificity"]:
        if col in df:
            df = df.drop(columns=[col])
    df["pool_name"] = pool_name
    df["session_fold"] = [fold_for_state(states[int(group_id)], 5) for group_id in df["group_id"].to_numpy()]

    numeric: dict[str, np.ndarray] = {}
    names_float = [
        "current_title_overlap",
        "current_artist_overlap",
        "current_album_overlap",
        "current_tags_overlap",
        "current_metadata_overlap",
        "current_title_jaccard",
        "current_artist_jaccard",
        "current_album_jaccard",
        "current_tags_jaccard",
        "current_metadata_jaccard",
        "current_metadata_idf",
        "goal_title_overlap",
        "goal_artist_overlap",
        "goal_album_overlap",
        "goal_tags_overlap",
        "goal_metadata_overlap",
        "goal_metadata_jaccard",
        "goal_metadata_idf",
        "history_metadata_overlap",
        "history_metadata_jaccard",
        "positive_metadata_overlap",
        "positive_metadata_jaccard",
        "negative_metadata_overlap",
        "negative_metadata_jaccard",
        "tag_overlap_count",
        "tag_overlap_ratio",
        "candidate_artist_prev_count",
        "candidate_artist_accepted_count",
        "candidate_artist_rejected_count",
        "candidate_album_prev_count",
        "candidate_album_accepted_count",
        "candidate_album_rejected_count",
        "same_tags_positive_overlap",
        "same_tags_negative_overlap",
        "release_year_pos_seed_min_abs_diff",
        "release_year_pos_seed_mean_abs_diff",
        "release_year_neg_seed_min_abs_diff",
        "popularity_prior",
        "log_popularity_prior",
        "duration_prior",
        "metadata_completeness_prior",
    ]
    names_int = [
        "exact_title_match_current",
        "exact_artist_match_current",
        "exact_album_match_current",
        "exact_title_match_goal",
        "exact_artist_match_goal",
        "exact_album_match_goal",
        "candidate_title_in_current",
        "candidate_artist_in_current",
        "candidate_album_in_current",
        "candidate_title_in_goal",
        "candidate_artist_in_goal",
        "candidate_album_in_goal",
        "candidate_title_in_history",
        "candidate_artist_in_history",
        "candidate_album_in_history",
        "previously_recommended",
        "same_artist_as_positive_seed_count",
        "same_album_as_positive_seed_count",
        "same_artist_as_negative_seed_count",
        "same_album_as_negative_seed_count",
        "exact_artist_query_exception",
        "exact_title_query_exception",
    ]
    for name in names_float:
        numeric[name] = np.zeros(n, dtype=np.float32)
    for name in names_int:
        numeric[name] = np.zeros(n, dtype=np.int16)

    emb_names: list[str] = []
    if track_embeddings is not None:
        for channel in embedding_channels:
            for suffix in [
                "candidate_valid",
                "pos_max",
                "pos_mean",
                "neg_max",
                "neg_mean",
                "pos_minus_neg",
                "weak_history_max",
                "weak_history_mean",
            ]:
                emb_names.append(f"emb_{channel}_{suffix}")
                numeric[f"emb_{channel}_{suffix}"] = np.zeros(n, dtype=np.float32)
    if user_embeddings is not None and track_embeddings is not None:
        for name in ["emb_user_cf_valid", "emb_user_cf_score"]:
            numeric[name] = np.zeros(n, dtype=np.float32)

    tid_array = df["track_id"].to_numpy()
    group_array = df["group_id"].to_numpy(dtype=np.int32)
    position_by_group: dict[int, np.ndarray] = {}
    for group_id, group_index in df.groupby("group_id", sort=False).groups.items():
        position_by_group[int(group_id)] = np.fromiter(group_index, dtype=np.int64)

    context_cache = [query_context(state, catalog) for state in tqdm(states, desc="Build state contexts")]
    for group_id, positions in tqdm(position_by_group.items(), desc=f"Add independent features {pool_name}"):
        state = states[group_id]
        ctx = context_cache[group_id]
        q_current = ctx["current_tokens"]
        q_goal = ctx["goal_tokens"]
        q_history = ctx["history_tokens"]
        q_pos = ctx["positive_tokens"]
        q_neg = ctx["negative_tokens"]
        q_all = ctx["all_tokens"]
        for pos in positions:
            track_id = tid_array[pos]
            title = text_cache.title_tokens.get(track_id, set())
            artist = text_cache.artist_tokens.get(track_id, set())
            album = text_cache.album_tokens.get(track_id, set())
            tags = text_cache.tag_tokens.get(track_id, set())
            metadata = text_cache.metadata_tokens.get(track_id, set())
            title_norm = text_cache.title_norm.get(track_id, "")
            artist_norm = text_cache.artist_norm.get(track_id, "")
            album_norm = text_cache.album_norm.get(track_id, "")
            current_norm = ctx["current_norm"]
            goal_norm = ctx["goal_norm"]
            history_norm = ctx["history_norm"]

            numeric["current_title_overlap"][pos] = overlap_count(q_current, title)
            numeric["current_artist_overlap"][pos] = overlap_count(q_current, artist)
            numeric["current_album_overlap"][pos] = overlap_count(q_current, album)
            numeric["current_tags_overlap"][pos] = overlap_count(q_current, tags)
            numeric["current_metadata_overlap"][pos] = overlap_count(q_current, metadata)
            numeric["current_title_jaccard"][pos] = jaccard(q_current, title)
            numeric["current_artist_jaccard"][pos] = jaccard(q_current, artist)
            numeric["current_album_jaccard"][pos] = jaccard(q_current, album)
            numeric["current_tags_jaccard"][pos] = jaccard(q_current, tags)
            numeric["current_metadata_jaccard"][pos] = jaccard(q_current, metadata)
            numeric["current_metadata_idf"][pos] = weighted_overlap(q_current, metadata, text_cache.idf)

            numeric["goal_title_overlap"][pos] = overlap_count(q_goal, title)
            numeric["goal_artist_overlap"][pos] = overlap_count(q_goal, artist)
            numeric["goal_album_overlap"][pos] = overlap_count(q_goal, album)
            numeric["goal_tags_overlap"][pos] = overlap_count(q_goal, tags)
            numeric["goal_metadata_overlap"][pos] = overlap_count(q_goal, metadata)
            numeric["goal_metadata_jaccard"][pos] = jaccard(q_goal, metadata)
            numeric["goal_metadata_idf"][pos] = weighted_overlap(q_goal, metadata, text_cache.idf)

            numeric["history_metadata_overlap"][pos] = overlap_count(q_history, metadata)
            numeric["history_metadata_jaccard"][pos] = jaccard(q_history, metadata)
            numeric["positive_metadata_overlap"][pos] = overlap_count(q_pos, metadata)
            numeric["positive_metadata_jaccard"][pos] = jaccard(q_pos, metadata)
            numeric["negative_metadata_overlap"][pos] = overlap_count(q_neg, metadata)
            numeric["negative_metadata_jaccard"][pos] = jaccard(q_neg, metadata)
            numeric["tag_overlap_count"][pos] = overlap_count(q_all, tags)
            numeric["tag_overlap_ratio"][pos] = safe_div(overlap_count(q_all, tags), len(tags))

            numeric["exact_title_match_current"][pos] = int(title_norm and normalize_text(state.current_user_query) == title_norm)
            numeric["exact_artist_match_current"][pos] = int(artist_norm and normalize_text(state.current_user_query) == artist_norm)
            numeric["exact_album_match_current"][pos] = int(album_norm and normalize_text(state.current_user_query) == album_norm)
            numeric["exact_title_match_goal"][pos] = int(title_norm and normalize_text(state.listener_goal) == title_norm)
            numeric["exact_artist_match_goal"][pos] = int(artist_norm and normalize_text(state.listener_goal) == artist_norm)
            numeric["exact_album_match_goal"][pos] = int(album_norm and normalize_text(state.listener_goal) == album_norm)
            numeric["candidate_title_in_current"][pos] = normalized_contains(title_norm, current_norm)
            numeric["candidate_artist_in_current"][pos] = normalized_contains(artist_norm, current_norm)
            numeric["candidate_album_in_current"][pos] = normalized_contains(album_norm, current_norm)
            numeric["candidate_title_in_goal"][pos] = normalized_contains(title_norm, goal_norm)
            numeric["candidate_artist_in_goal"][pos] = normalized_contains(artist_norm, goal_norm)
            numeric["candidate_album_in_goal"][pos] = normalized_contains(album_norm, goal_norm)
            numeric["candidate_title_in_history"][pos] = normalized_contains(title_norm, history_norm)
            numeric["candidate_artist_in_history"][pos] = normalized_contains(artist_norm, history_norm)
            numeric["candidate_album_in_history"][pos] = normalized_contains(album_norm, history_norm)
            numeric["exact_artist_query_exception"][pos] = int(
                numeric["candidate_artist_in_current"][pos] or numeric["candidate_artist_in_goal"][pos]
            )
            numeric["exact_title_query_exception"][pos] = int(
                numeric["candidate_title_in_current"][pos] or numeric["candidate_title_in_goal"][pos]
            )

            numeric["previously_recommended"][pos] = int(track_id in state.previous_music_track_ids)
            numeric["candidate_artist_prev_count"][pos] = ctx["previous_artists"].get(artist_norm, 0)
            numeric["candidate_artist_accepted_count"][pos] = ctx["accepted_artists"].get(artist_norm, 0)
            numeric["candidate_artist_rejected_count"][pos] = ctx["rejected_artists"].get(artist_norm, 0)
            numeric["candidate_album_prev_count"][pos] = ctx["previous_albums"].get(album_norm, 0)
            numeric["candidate_album_accepted_count"][pos] = ctx["accepted_albums"].get(album_norm, 0)
            numeric["candidate_album_rejected_count"][pos] = ctx["rejected_albums"].get(album_norm, 0)
            numeric["same_artist_as_positive_seed_count"][pos] = ctx["accepted_artists"].get(artist_norm, 0)
            numeric["same_album_as_positive_seed_count"][pos] = ctx["accepted_albums"].get(album_norm, 0)
            numeric["same_artist_as_negative_seed_count"][pos] = ctx["rejected_artists"].get(artist_norm, 0)
            numeric["same_album_as_negative_seed_count"][pos] = ctx["rejected_albums"].get(album_norm, 0)
            numeric["same_tags_positive_overlap"][pos] = overlap_count(tags, ctx["pos_tags"])
            numeric["same_tags_negative_overlap"][pos] = overlap_count(tags, ctx["neg_tags"])
            year = text_cache.release_year.get(track_id, 0)
            if year and ctx["pos_years"]:
                diffs = [abs(year - seed_year) for seed_year in ctx["pos_years"]]
                numeric["release_year_pos_seed_min_abs_diff"][pos] = min(diffs)
                numeric["release_year_pos_seed_mean_abs_diff"][pos] = float(sum(diffs) / len(diffs))
            else:
                numeric["release_year_pos_seed_min_abs_diff"][pos] = 9999.0
                numeric["release_year_pos_seed_mean_abs_diff"][pos] = 9999.0
            if year and ctx["neg_years"]:
                numeric["release_year_neg_seed_min_abs_diff"][pos] = min(abs(year - seed_year) for seed_year in ctx["neg_years"])
            else:
                numeric["release_year_neg_seed_min_abs_diff"][pos] = 9999.0
            numeric["popularity_prior"][pos] = text_cache.popularity.get(track_id, 0.0)
            numeric["log_popularity_prior"][pos] = math.log1p(max(text_cache.popularity.get(track_id, 0.0), 0.0))
            numeric["duration_prior"][pos] = text_cache.duration.get(track_id, 0.0)
            numeric["metadata_completeness_prior"][pos] = float(
                bool(title_norm) + bool(artist_norm) + bool(album_norm) + bool(tags) + bool(year)
            )

        if track_embeddings is not None:
            group_track_ids = [tid_array[pos] for pos in positions]
            emb_indices = np.array(
                [track_embeddings.track_index.get(track_id, -1) for track_id in group_track_ids],
                dtype=np.int64,
            )
            for channel in embedding_channels:
                matrix = track_embeddings.matrices[channel]
                valid = (emb_indices >= 0) & matrix.valid[np.clip(emb_indices, 0, len(matrix.valid) - 1)]
                numeric[f"emb_{channel}_candidate_valid"][positions] = valid.astype(np.float32)
                pos_indices = track_embeddings.indices_for(state.positive_seed_ids)
                neg_indices = track_embeddings.indices_for(state.negative_seed_ids)
                weak_indices = track_embeddings.indices_for(state.previous_music_track_ids)
                for label, seed_indices in [("pos", pos_indices), ("neg", neg_indices), ("weak_history", weak_indices)]:
                    if valid.any() and seed_indices:
                        seed_valid = matrix.valid[seed_indices]
                        if seed_valid.any():
                            cand_vec = matrix.normalized[emb_indices[valid]]
                            seed_vec = matrix.normalized[np.array(seed_indices)[seed_valid]]
                            scores = cand_vec @ seed_vec.T
                            max_scores = np.zeros(len(positions), dtype=np.float32)
                            mean_scores = np.zeros(len(positions), dtype=np.float32)
                            max_scores[valid] = scores.max(axis=1)
                            mean_scores[valid] = scores.mean(axis=1)
                            numeric[f"emb_{channel}_{label}_max"][positions] = max_scores
                            numeric[f"emb_{channel}_{label}_mean"][positions] = mean_scores
                numeric[f"emb_{channel}_pos_minus_neg"][positions] = (
                    numeric[f"emb_{channel}_pos_max"][positions] - numeric[f"emb_{channel}_neg_max"][positions]
                )
            if user_embeddings is not None and user_embeddings.has_user(state.user_id):
                user = user_embeddings.user_vectors[state.user_id]
                matrix = track_embeddings.matrices["track_cf"]
                valid = (
                    (emb_indices >= 0)
                    & matrix.valid[np.clip(emb_indices, 0, len(matrix.valid) - 1)]
                    & (matrix.raw.shape[1] == user.shape[0])
                )
                scores = np.zeros(len(positions), dtype=np.float32)
                if valid.any():
                    scores[valid] = matrix.raw[emb_indices[valid]] @ user
                numeric["emb_user_cf_valid"][positions] = valid.astype(np.float32)
                numeric["emb_user_cf_score"][positions] = scores

    for name, values in numeric.items():
        df[name] = values
    for group_id, state in enumerate(states):
        pass
    df["intent"] = [infer_intent_light(states[int(group_id)]) for group_id in group_array]
    df["category"] = [states[int(group_id)].category for group_id in group_array]
    df["specificity"] = [states[int(group_id)].specificity for group_id in group_array]
    return reduce_dtypes(df)


def infer_intent_light(state: ConversationState) -> str:
    text = f"{state.current_user_query} {state.listener_goal}".lower()
    if any(word in text for word in ["called", "song named", "track named", "specific song", "\"", "'"]):
        return "specific_track"
    if "album" in text:
        return "album"
    if "artist" in text or "by " in text:
        return "artist"
    if any(word in text for word in ["mood", "relax", "energetic", "sad", "happy", "party", "workout"]):
        return "mood"
    return "general"


def reduce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in META_COLUMNS or col in {"pool_name", "intent", "category", "specificity", "nextgen_family_names"}:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[col]):
            if col == "group_id":
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int16, errors="ignore")
    return df


def source_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in META_COLUMNS or col in {"pool_name", "intent", "category", "specificity", "session_fold", "nextgen_family_names"}:
            continue
        if col.startswith(SOURCE_PREFIXES):
            cols.append(col)
    return cols


def independent_feature_columns(df: pd.DataFrame) -> list[str]:
    source_cols = set(source_feature_columns(df))
    out = []
    for col in df.columns:
        if col in META_COLUMNS or col in {"pool_name", "session_fold", "nextgen_family_names"}:
            continue
        if col in source_cols:
            continue
        out.append(col)
    return out


def feature_columns_for(df: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "source":
        cols = source_feature_columns(df)
    elif feature_set == "independent":
        cols = independent_feature_columns(df)
    elif feature_set == "all":
        ignored = set(META_COLUMNS) | {"pool_name", "session_fold", "nextgen_family_names"}
        cols = [col for col in df.columns if col not in ignored]
    else:
        raise ValueError(f"Unknown feature_set={feature_set}")
    return sorted(cols)


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    category_values: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = df.copy()
    categorical = [col for col in ["intent", "category", "specificity"] if col in feature_cols]
    if category_values is None:
        category_values = {}
        for col in categorical:
            values = sorted(str(value) for value in out[col].fillna("missing").unique())
            category_values[col] = values or ["missing"]
    for col in feature_cols:
        if col not in out:
            out[col] = np.nan
    for col in categorical:
        out[col] = pd.Categorical(out[col].fillna("missing").astype(str), categories=category_values[col])
    for col in feature_cols:
        if col in categorical:
            continue
        if col.startswith("rank_") or col.startswith("source_rank_") or "best_source_rank" in col or col.endswith("_rank"):
            out[col] = out[col].fillna(9999.0).astype(np.float32)
        else:
            out[col] = out[col].fillna(0.0).astype(np.float32)
    return out, category_values


def mean_ndcg_and_predictions(
    df: pd.DataFrame,
    states: list[ConversationState],
    score_col: str = "model_score",
) -> tuple[float, int, list[dict[str, Any]]]:
    scores = []
    hit20 = 0
    predictions = []
    for group_id, group in df.groupby("group_id", sort=True):
        state = states[int(group_id)]
        ranked = list(group.sort_values(score_col, ascending=False)["track_id"])
        gold = state.gold_track_id
        gold_rank = None
        for rank, track_id in enumerate(ranked, start=1):
            if track_id == gold:
                gold_rank = rank
                break
        score = 0.0
        if gold_rank is not None and gold_rank <= 20:
            score = 1.0 / math.log2(gold_rank + 1)
            hit20 += 1
        scores.append(score)
        predictions.append(
            {
                "session_id": state.session_id,
                "user_id": state.user_id,
                "turn_number": state.turn_number,
                "gold_track_id": gold,
                "gold_rank": gold_rank,
                "predicted_track_ids": ranked[:20],
            }
        )
    return float(np.mean(scores)), hit20, predictions


def train_oof(
    df: pd.DataFrame,
    states: list[ConversationState],
    feature_set: str,
    args: argparse.Namespace,
    out_dir: Path,
    pool_name: str,
) -> dict[str, Any]:
    start = time.time()
    feature_cols = feature_columns_for(df, feature_set)
    feature_hash = stable_json_hash(feature_cols)
    params = {
        "objective": args.objective,
        "metric": "ndcg",
        "eval_at": [20],
        "lambdarank_truncation_level": args.lambdarank_truncation_level,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "subsample": args.subsample,
        "subsample_freq": 1,
        "colsample_bytree": args.colsample_bytree,
        "random_state": 2026,
        "force_row_wise": True,
    }
    params_hash = stable_json_hash(params)
    folds = {group_id: fold_for_state(states[int(group_id)], args.folds) for group_id in df["group_id"].unique()}
    split_hash = stable_json_hash({str(k): int(v) for k, v in sorted(folds.items())})
    scored_frames = []
    fold_summaries = []
    feature_importances = []
    for fold in range(args.folds):
        valid_groups = {group_id for group_id, group_fold in folds.items() if group_fold == fold}
        train_groups = set(folds) - valid_groups
        train_df = df[df["group_id"].isin(train_groups)].copy()
        positive_groups = set(train_df.groupby("group_id")["label"].sum().loc[lambda s: s > 0].index)
        train_df = train_df[train_df["group_id"].isin(positive_groups)].copy()
        valid_df = df[df["group_id"].isin(valid_groups)].copy()
        train_df, category_values = prepare_features(train_df, feature_cols)
        valid_df, _ = prepare_features(valid_df, feature_cols, category_values=category_values)
        train_df = train_df.sort_values("group_id")
        valid_df = valid_df.sort_values("group_id")
        train_group_sizes = train_df.groupby("group_id", sort=False).size().to_list()
        valid_group_sizes = valid_df.groupby("group_id", sort=False).size().to_list()
        categorical = [col for col in ["intent", "category", "specificity"] if col in feature_cols]
        ranker = lgb.LGBMRanker(**params)
        ranker.fit(
            train_df[feature_cols],
            train_df["label"],
            group=train_group_sizes,
            eval_set=[(valid_df[feature_cols], valid_df["label"])],
            eval_group=[valid_group_sizes],
            eval_at=[20],
            categorical_feature=categorical,
            callbacks=[
                lgb.early_stopping(args.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )
        valid_df = valid_df.copy()
        valid_df["model_score"] = ranker.predict(valid_df[feature_cols])
        fold_ndcg, fold_hit20, _ = mean_ndcg_and_predictions(valid_df, states)
        fold_summaries.append(
            {
                "fold": fold,
                "valid_groups": len(valid_groups),
                "train_groups_with_positive": len(positive_groups),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "best_iteration": int(getattr(ranker, "best_iteration_", 0) or args.n_estimators),
                "hit20": int(fold_hit20),
                "ndcg20": float(fold_ndcg),
            }
        )
        importances = ranker.feature_importances_
        feature_importances.append(pd.DataFrame({"feature": feature_cols, f"fold_{fold}": importances}))
        scored_frames.append(valid_df[["group_id", "session_id", "user_id", "turn_number", "track_id", "label", "model_score"]])
    scored = pd.concat(scored_frames, ignore_index=True)
    ndcg20, hit20, predictions = mean_ndcg_and_predictions(scored, states)
    pred_path = out_dir / f"predictions_oof_{pool_name}.jsonl"
    if feature_set != "all":
        pred_path = out_dir / f"predictions_oof_{pool_name}_{feature_set}.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    fi = feature_importances[0]
    for item in feature_importances[1:]:
        fi = fi.merge(item, on="feature", how="outer")
    fold_cols = [col for col in fi.columns if col.startswith("fold_")]
    fi["importance_mean"] = fi[fold_cols].mean(axis=1)
    fi = fi.sort_values("importance_mean", ascending=False)
    if feature_set == "all":
        fi.to_csv(out_dir / f"feature_importance_{pool_name}_all.csv", index=False)
    else:
        fi.to_csv(out_dir / f"feature_importance_{pool_name}_{feature_set}.csv", index=False)
    (out_dir / "feature_columns.txt").write_text("\n".join(feature_cols) + "\n", encoding="utf-8")
    return {
        "method": "LTR_v2",
        "pool": pool_name,
        "features": feature_set,
        "ranker": args.objective,
        "max_candidates": 0,
        "hit20": int(hit20),
        "ndcg20": float(ndcg20),
        "train_rows": int(sum(item["train_rows"] for item in fold_summaries)),
        "train_time": float(time.time() - start),
        "best_iteration": float(np.mean([item["best_iteration"] for item in fold_summaries])),
        "params_hash": params_hash,
        "feature_hash": feature_hash,
        "split_hash": split_hash,
        "folds": fold_summaries,
        "prediction_path": str(pred_path),
        "feature_count": len(feature_cols),
    }


def extra_gold_diagnostics(old_df: pd.DataFrame, beam_df: pd.DataFrame, scored_union: pd.DataFrame | None, out_dir: Path) -> None:
    old_hit = set(old_df.loc[old_df["label"] == 1, "group_id"].astype(int))
    beam_hit = set(beam_df.loc[beam_df["label"] == 1, "group_id"].astype(int))
    rows = []
    for label, groups, source in [
        ("extra_gold", sorted(beam_hit - old_hit), beam_df),
        ("lost_gold", sorted(old_hit - beam_hit), old_df),
        ("both_hit", sorted(old_hit & beam_hit), beam_df),
    ]:
        gold = source[(source["label"] == 1) & (source["group_id"].isin(groups))].copy()
        for _, row in gold.iterrows():
            rows.append(
                {
                    "bucket": label,
                    "group_id": int(row["group_id"]),
                    "track_id": row["track_id"],
                    "source_count": float(row.get("source_count", 0.0)),
                    "best_source_rank": float(row.get("best_source_rank", 9999.0)),
                    "best_source_bucket": rank_bucket(float(row.get("best_source_rank", 9999.0))),
                    "rrf_score": float(row.get("rrf_score", 0.0)),
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "extra_gold_diagnostics.csv", index=False)


def build_or_load_features(
    pool_name: str,
    base_df: pd.DataFrame,
    states: list[ConversationState],
    catalog: TrackCatalog,
    text_cache: TrackTextCache,
    track_embeddings: TrackEmbeddingStore | None,
    user_embeddings: UserEmbeddingStore | None,
    embedding_channels: list[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> pd.DataFrame:
    feature_path = out_dir / f"features_{pool_name}.pkl"
    meta_path = out_dir / f"features_{pool_name}.json"
    if feature_path.exists() and meta_path.exists() and not args.rebuild_features:
        print(f"Loaded feature cache: {feature_path}")
        return pd.read_pickle(feature_path)
    df = build_feature_frame(
        pool_name=pool_name,
        base_df=base_df,
        states=states,
        catalog=catalog,
        text_cache=text_cache,
        track_embeddings=track_embeddings,
        user_embeddings=user_embeddings,
        embedding_channels=embedding_channels,
    )
    df.to_pickle(feature_path)
    meta_path.write_text(json.dumps({"rows": len(df), "columns": list(df.columns)}, indent=2), encoding="utf-8")
    print(f"Wrote feature cache: {feature_path}")
    return df


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = GoalFlowConfig(project_root=Path(args.project_root), tid="rerank_v2_independent_features")
    params_payload = vars(args).copy()
    (out_dir / "params.json").write_text(json.dumps(params_payload, indent=2), encoding="utf-8")
    catalog = TrackCatalog(config.track_metadata_name)
    states = load_dev_states(config)
    split_manifest = [
        {"group_id": i, "session_id": state.session_id, "user_id": state.user_id, "turn_number": state.turn_number, "fold": fold_for_state(state, args.folds)}
        for i, state in enumerate(states)
    ]
    split_hash = stable_json_hash(split_manifest)
    (out_dir / "split_manifest.json").write_text(
        json.dumps({"split_hash": split_hash, "rows": split_manifest}, indent=2),
        encoding="utf-8",
    )
    text_cache = TrackTextCache(catalog)
    embedding_channels = [item.strip() for item in args.embedding_channels.split(",") if item.strip()]
    track_embeddings = None
    user_embeddings = None
    if embedding_channels:
        channel_map = {name: TRACK_EMBEDDING_CHANNELS[name] for name in embedding_channels}
        track_embeddings = TrackEmbeddingStore(channels=channel_map)
        if not args.disable_user_cf and "track_cf" in embedding_channels:
            user_embeddings = UserEmbeddingStore()

    pools_requested = [item.strip() for item in args.pools.split(",") if item.strip()]
    feature_sets = [item.strip() for item in args.feature_sets.split(",") if item.strip()]
    base_pools: dict[str, pd.DataFrame] = {}
    old_base = None
    beam_base = None
    if any(pool in pools_requested for pool in ["old300", "union"]):
        old_base = load_old300_pool(Path(args.old300_pkl), states)
        base_pools["old300"] = old_base
    if any(pool in pools_requested for pool in ["beam800", "union"]):
        beam_base = load_beam800_pool(Path(args.matrix_dir), Path(args.beam800_choice), states, args.rrf_k)
        base_pools["beam800"] = beam_base
    if "union" in pools_requested:
        if old_base is None:
            old_base = load_old300_pool(Path(args.old300_pkl), states)
        if beam_base is None:
            beam_base = load_beam800_pool(Path(args.matrix_dir), Path(args.beam800_choice), states, args.rrf_k)
        base_pools["union"] = make_union_pool(old_base, beam_base, max_union_size=args.max_union_size)
    if args.custom_pool_pkl and args.custom_pool_name in pools_requested:
        custom = pd.read_pickle(args.custom_pool_pkl).copy()
        if "nextgen_family_names" in custom.columns:
            custom = custom.drop(columns=["nextgen_family_names"])
        base_pools[args.custom_pool_name] = normalize_pool_base(custom)

    pool_rows = []
    for pool_name, base_df in base_pools.items():
        summary = pool_summary(pool_name, base_df, states)
        pool_rows.append(summary)
    pd.DataFrame(pool_rows).to_csv(out_dir / "pool_summary.csv", index=False)

    metrics = []
    ablations = []
    for pool_name in pools_requested:
        base_df = base_pools[pool_name]
        feature_df = build_or_load_features(
            pool_name,
            base_df,
            states,
            catalog,
            text_cache,
            track_embeddings,
            user_embeddings,
            embedding_channels,
            args,
            out_dir,
        )
        p_summary = next(item for item in pool_rows if item["pool"] == pool_name)
        for feature_set in feature_sets:
            result = train_oof(feature_df, states, feature_set, args, out_dir, pool_name)
            row = {
                "method": result["method"],
                "pool": pool_name,
                "features": feature_set,
                "ranker": result["ranker"],
                "max_candidates": result["max_candidates"],
                "avg_group_size": p_summary["group_size_mean"],
                "p95_group_size": p_summary["group_size_p95"],
                "gold_in_pool": p_summary["gold_in_pool"],
                "hit20": result["hit20"],
                "ndcg20": result["ndcg20"],
                "train_rows": result["train_rows"],
                "best_iteration": result["best_iteration"],
                "params_hash": result["params_hash"],
                "feature_hash": result["feature_hash"],
                "split_hash": result["split_hash"],
                "train_time": result["train_time"],
                "feature_count": result["feature_count"],
            }
            metrics.append(row)
            if feature_set in {"source", "independent", "all"}:
                ablations.append(row)
            pd.DataFrame(metrics).to_csv(out_dir / "metrics_summary.csv", index=False)
            pd.DataFrame(ablations).to_csv(out_dir / "ablation_summary.csv", index=False)
            (out_dir / f"folds_{pool_name}_{feature_set}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if old_base is not None and beam_base is not None:
        extra_gold_diagnostics(old_base, beam_base, None, out_dir)
    readme = [
        "# Rerank V2 Independent Features",
        "",
        "This experiment separates candidate recall from learned relevance ranking.",
        "RRF/source rank features are retained only as auxiliary features.",
        "",
        "Run order should be: old300 control, union, beam800, then ablations.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"done out_dir={out_dir}")


if __name__ == "__main__":
    main()
