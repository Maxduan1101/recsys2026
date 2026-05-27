from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from goalflow.data import BLIND_A_DATASET, TrackCatalog, normalize_text
from goalflow.embeddings import TRACK_EMBEDDING_CHANNELS, TrackEmbeddingStore, UserEmbeddingStore
from goalflow.pipeline import GoalFlowConfig
from goalflow.state import ConversationState, build_state_for_blind_item
from run_rerank_v2 import (
    MATRIX_DIR,
    BEAM800_CHOICE,
    OLD300_PKL,
    TrackTextCache,
    jaccard,
    load_beam800_pool,
    load_dev_states,
    load_old300_pool,
    make_union_pool,
    normalize_pool_base,
    overlap_count,
    pool_summary,
    query_context,
    rank_bucket,
    read_meta,
)


DEFAULT_OUT_DIR = Path("goalflow_musiccrs/experiments/nextgen_candidate_pool_v1")
DEFAULT_SEED_CHANNEL_KS = {
    "track_cf": 180,
    "attributes": 120,
    "audio": 100,
    "lyrics": 80,
    "image": 60,
}
SEED_CHANNEL_FAMILY = {
    "track_cf": "cf_seed",
    "attributes": "attributes_seed",
    "audio": "audio_seed",
    "lyrics": "lyrics_seed",
    "image": "image_seed",
    "metadata": "metadata_seed",
}
NEXTGEN_FAMILIES = [
    "bm25_tail",
    "cf_seed",
    "attributes_seed",
    "audio_seed",
    "lyrics_seed",
    "image_seed",
    "metadata_seed",
    "user_cf",
]


TAIL_SOURCE_RANGES = [
    ("enriched:current", 51, 800, 1.0),
    ("enriched:current_goal", 101, 800, 1.0),
    ("metadata_all:goal", 1, 800, 0.9),
    ("tags:current_goal", 51, 800, 0.9),
    ("tags:seed_current", 401, 800, 0.85),
    ("enriched:legacy_history", 201, 800, 0.85),
    ("metadata_all:legacy_history", 1, 800, 0.75),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build next-generation candidate pool with tail rescue and CF seed retrieval.")
    parser.add_argument("--project-root", default="goalflow_musiccrs")
    parser.add_argument("--mode", choices=["dev", "blind"], default="dev")
    parser.add_argument("--blind-dataset-name", default=BLIND_A_DATASET)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--old300-pkl", default=OLD300_PKL)
    parser.add_argument("--matrix-dir", default=MATRIX_DIR)
    parser.add_argument("--beam800-choice", default=BEAM800_CHOICE)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--tail-max-add", type=int, default=180)
    parser.add_argument("--tail-score-cutoff", type=float, default=-1e9)
    parser.add_argument("--cf-seed-k", type=int, default=180)
    parser.add_argument(
        "--seed-channels",
        default="track_cf",
        help="Comma-separated official embedding channels used for seed-neighbor retrieval.",
    )
    parser.add_argument(
        "--seed-channel-ks",
        default="",
        help="Comma-separated channel:k overrides, e.g. track_cf:180,attributes:120,audio:100.",
    )
    parser.add_argument("--user-cf-k", type=int, default=80)
    parser.add_argument("--enable-user-cf", action="store_true")
    parser.add_argument("--max-pool-size", type=int, default=1400)
    parser.add_argument("--write-pkl", action="store_true")
    return parser.parse_args()


def load_blind_states(config: GoalFlowConfig) -> list[ConversationState]:
    dataset = load_dataset(config.blind_dataset_name, split="test")
    return [build_state_for_blind_item(item) for item in tqdm(dataset, desc="Build blind states")]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def source_matrix(matrix_dir: Path):
    meta = read_meta(matrix_dir)
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
    source_rows = read_tsv(matrix_dir / "sources.tsv")
    track_rows = read_tsv(matrix_dir / "track_ids.tsv")
    source_names = [row["source_name"] for row in source_rows]
    source_name_to_index = {row["source_name"]: int(row["source_index"]) for row in source_rows}
    track_ids = [row["track_id"] for row in track_rows]
    return candidates, counts, source_names, source_name_to_index, track_ids


def text_flags(state: ConversationState) -> dict[str, bool]:
    text = " ".join([state.current_user_query, state.listener_goal]).lower()
    double_quoted = re.findall(r'"([^"]{3,80})"', text)
    single_quoted = re.findall(r"(?<!\w)'([^']{3,80})'(?!\w)", text)
    has_quoted_phrase = bool(double_quoted or single_quoted)
    broad = any(word in text for word in ["similar", "like this", "more like", "another", "recommend", "discover", "playlist"])
    exact_entity = (
        (has_quoted_phrase and not broad)
        or "exact title" in text
        or "exact artist" in text
        or "specific track" in text
    )
    return {
        "exact_entity": exact_entity,
        "broad": broad,
        "mood": any(word in text for word in ["mood", "melanch", "sad", "dark", "dream", "energetic", "chill", "relax", "atmospheric"]),
        "lyrics": any(word in text for word in ["lyric", "lyrics", "story", "storytelling", "narrative", "words", "verse", "chorus"]),
        "cover": any(word in text for word in ["cover art", "album cover", "artwork", "cover image", "picture"]),
        "audio": any(word in text for word in ["bass", "beat", "instrumental", "guitar", "piano", "drum", "vocal", "female", "harsh", "heavy", "soft"]),
    }


def add_candidate(
    bucket: dict[str, dict[str, Any]],
    track_id: str,
    source_name: str,
    rank: int,
    score: float,
) -> None:
    item = bucket.setdefault(
        track_id,
        {
            "track_id": track_id,
            "rrf_score": 0.0,
            "source_count": 0,
            "best_source_rank": 9999,
            "nextgen_sources": set(),
            "rank_cols": {},
        },
    )
    item["rrf_score"] += float(score)
    item["source_count"] += 1
    item["best_source_rank"] = min(int(item["best_source_rank"]), int(rank))
    item["nextgen_sources"].add(source_name)
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", source_name).strip("_")
    col = f"rank_nextgen_{safe}"
    previous = item["rank_cols"].get(col, 9999)
    item["rank_cols"][col] = min(previous, int(rank))


def lexical_tail_score(
    track_id: str,
    ctx: dict[str, Any],
    text_cache: TrackTextCache,
) -> float:
    metadata = text_cache.metadata_tokens.get(track_id, set())
    tags = text_cache.tag_tokens.get(track_id, set())
    if not metadata:
        return 0.0
    current = overlap_count(ctx["current_tokens"], metadata)
    goal = overlap_count(ctx["goal_tokens"], metadata)
    history = overlap_count(ctx["history_tokens"], metadata)
    tag_overlap = overlap_count(ctx["all_tokens"], tags)
    return (
        0.05 * current
        + 0.04 * goal
        + 0.025 * history
        + 0.02 * tag_overlap
        + 0.10 * jaccard(ctx["all_tokens"], tags)
    )


def seed_relation_score(
    track_id: str,
    state: ConversationState,
    catalog: TrackCatalog,
) -> float:
    if not catalog.has_track(track_id):
        return 0.0
    artist = catalog.normalized_field(track_id, "artist_name")
    album = catalog.normalized_field(track_id, "album_name")
    score = 0.0
    for seed_id in state.positive_seed_ids:
        if not catalog.has_track(seed_id):
            continue
        if artist and artist == catalog.normalized_field(seed_id, "artist_name"):
            score += 0.12
        if album and album == catalog.normalized_field(seed_id, "album_name"):
            score += 0.08
        score += min(0.08, 0.005 * len(catalog.tag_words(track_id) & catalog.tag_words(seed_id)))
    for seed_id in state.negative_seed_ids:
        if not catalog.has_track(seed_id):
            continue
        if artist and artist == catalog.normalized_field(seed_id, "artist_name"):
            score -= 0.18
        if album and album == catalog.normalized_field(seed_id, "album_name"):
            score -= 0.12
        score -= min(0.08, 0.004 * len(catalog.tag_words(track_id) & catalog.tag_words(seed_id)))
    return score


def build_tail_rescue_for_group(
    group_id: int,
    state: ConversationState,
    base_set: set[str],
    candidates: np.memmap,
    counts: np.memmap,
    source_name_to_index: dict[str, int],
    track_ids: list[str],
    catalog: TrackCatalog,
    text_cache: TrackTextCache,
    max_add: int,
    score_cutoff: float,
) -> list[dict[str, Any]]:
    ctx = query_context(state, catalog)
    raw: dict[str, dict[str, Any]] = {}
    for source_name, lo, hi, weight in TAIL_SOURCE_RANGES:
        source_index = source_name_to_index.get(source_name)
        if source_index is None:
            continue
        limit = min(int(counts[group_id, source_index]), hi)
        if limit < lo:
            continue
        row = candidates[group_id, source_index, lo - 1 : limit]
        for offset, track_index_raw in enumerate(row, start=lo):
            track_index = int(track_index_raw)
            if track_index < 0:
                continue
            track_id = track_ids[track_index]
            if track_id in base_set:
                continue
            score = weight / math.log2(offset + 2)
            add_candidate(raw, track_id, source_name, offset, score)
    scored = []
    for track_id, item in raw.items():
        score = (
            float(item["rrf_score"])
            + 0.15 * math.log1p(len(item["nextgen_sources"]))
            + lexical_tail_score(track_id, ctx, text_cache)
            + seed_relation_score(track_id, state, catalog)
            + 0.01 * math.log1p(max(catalog.view(track_id).popularity, 0.0))
        )
        if score >= score_cutoff:
            scored.append((score, track_id, item))
    scored.sort(reverse=True)
    out = []
    for rank, (score, track_id, item) in enumerate(scored[:max_add], start=1):
        row = {
            "track_id": track_id,
            "nextgen_score": score,
            "nextgen_family": "bm25_tail",
            "rank_nextgen_bm25_tail": rank,
        }
        row.update(item["rank_cols"])
        out.append(row)
    return out


def weighted_seed_vector(
    state: ConversationState,
    track_store: TrackEmbeddingStore,
    channel: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    matrix = track_store.matrices[channel]
    pos_vectors = []
    pos_weights = []
    neg_vectors = []
    neg_weights = []
    recent_track_to_age = {
        track_id: max(0, len(state.previous_music_track_ids) - index - 1)
        for index, track_id in enumerate(state.previous_music_track_ids)
    }
    for track_id in state.positive_seed_ids:
        index = track_store.track_index.get(track_id)
        if index is None or not matrix.valid[index]:
            continue
        age = recent_track_to_age.get(track_id, 0)
        pos_vectors.append(matrix.normalized[index])
        pos_weights.append(1.0 * (0.85**age))
    for track_id in state.negative_seed_ids:
        index = track_store.track_index.get(track_id)
        if index is None or not matrix.valid[index]:
            continue
        age = recent_track_to_age.get(track_id, 0)
        neg_vectors.append(matrix.normalized[index])
        neg_weights.append(1.0 * (0.85**age))
    if not pos_vectors and state.previous_music_track_ids:
        # Weak fallback: use the latest neutral recommendation with low weight.
        for track_id in state.previous_music_track_ids[-1:]:
            index = track_store.track_index.get(track_id)
            if index is not None and matrix.valid[index]:
                pos_vectors.append(matrix.normalized[index])
                pos_weights.append(0.35)
    pos = None
    neg = None
    if pos_vectors:
        pos = np.average(np.vstack(pos_vectors), axis=0, weights=np.asarray(pos_weights, dtype=np.float32)).astype(np.float32)
        norm = np.linalg.norm(pos)
        pos = pos / norm if norm > 1e-12 else None
    if neg_vectors:
        neg = np.average(np.vstack(neg_vectors), axis=0, weights=np.asarray(neg_weights, dtype=np.float32)).astype(np.float32)
        norm = np.linalg.norm(neg)
        neg = neg / norm if norm > 1e-12 else None
    return pos, neg


def topk_scores(scores: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(scores)
    if not finite.any() or top_k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    candidate_indices = np.flatnonzero(finite)
    values = scores[candidate_indices]
    k = min(top_k, len(values))
    top = np.argpartition(-values, k - 1)[:k]
    ordered = top[np.argsort(-values[top])]
    return candidate_indices[ordered], values[ordered].astype(np.float32)


def parse_seed_channels(value: str) -> list[str]:
    channels = []
    for item in value.split(","):
        channel = item.strip()
        if not channel:
            continue
        if channel not in TRACK_EMBEDDING_CHANNELS:
            raise ValueError(f"unknown embedding channel: {channel}")
        channels.append(channel)
    return channels


def parse_seed_channel_ks(value: str, cf_seed_k: int) -> dict[str, int]:
    out = dict(DEFAULT_SEED_CHANNEL_KS)
    out["track_cf"] = cf_seed_k
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"bad --seed-channel-ks item: {item}")
        channel, raw_k = item.split(":", 1)
        channel = channel.strip()
        if channel not in TRACK_EMBEDDING_CHANNELS:
            raise ValueError(f"unknown embedding channel in --seed-channel-ks: {channel}")
        out[channel] = int(raw_k)
    return out


def channel_allowed(channel: str, flags: dict[str, bool], state: ConversationState) -> bool:
    has_seed = bool(state.positive_seed_ids or state.previous_music_track_ids)
    if not has_seed:
        return False
    semantic_allowed = (not flags["exact_entity"]) or flags["broad"] or flags["mood"] or flags["audio"] or bool(state.positive_seed_ids)
    if channel == "track_cf":
        return semantic_allowed
    if channel == "attributes":
        return semantic_allowed and (flags["mood"] or flags["audio"] or flags["broad"] or bool(state.positive_seed_ids))
    if channel == "audio":
        return semantic_allowed and (flags["audio"] or flags["mood"] or flags["broad"])
    if channel == "lyrics":
        return semantic_allowed and flags["lyrics"]
    if channel == "image":
        return semantic_allowed and flags["cover"]
    if channel == "metadata":
        return semantic_allowed and (flags["broad"] or flags["mood"])
    return False


def add_seed_embedding_sources_for_group(
    group_id: int,
    state: ConversationState,
    base_set: set[str],
    bucket: dict[str, dict[str, Any]],
    track_store: TrackEmbeddingStore,
    seed_channels: list[str],
    seed_channel_ks: dict[str, int],
) -> None:
    excluded = set(base_set) | set(state.previous_music_track_ids) | set(state.negative_seed_ids)
    flags = text_flags(state)
    scale = {
        "track_cf": 1.25,
        "attributes": 1.00,
        "audio": 0.95,
        "lyrics": 0.85,
        "image": 0.70,
        "metadata": 0.80,
    }
    for channel in seed_channels:
        top_k = int(seed_channel_ks.get(channel, 0))
        if top_k <= 0 or channel not in track_store.matrices:
            continue
        if not channel_allowed(channel, flags, state):
            continue
        matrix = track_store.matrices[channel]
        pos, neg = weighted_seed_vector(state, track_store, channel)
        if pos is not None:
            scores = matrix.normalized @ pos
            if neg is not None:
                scores = scores - 0.5 * (matrix.normalized @ neg)
            scores = scores.astype(np.float32)
            scores[~matrix.valid] = -np.inf
            for track_id in excluded:
                index = track_store.track_index.get(track_id)
                if index is not None:
                    scores[index] = -np.inf
            indices, values = topk_scores(scores, top_k)
            family = SEED_CHANNEL_FAMILY.get(channel, f"{channel}_seed")
            for rank, (index, value) in enumerate(zip(indices, values), start=1):
                add_candidate(bucket, track_store.track_ids[int(index)], family, rank, scale.get(channel, 0.8) * float(value) / (20.0 + rank))


def add_user_cf_for_group(
    state: ConversationState,
    base_set: set[str],
    bucket: dict[str, dict[str, Any]],
    track_store: TrackEmbeddingStore,
    user_store: UserEmbeddingStore | None,
    user_cf_k: int,
    enable_user_cf: bool,
) -> None:
    excluded = set(base_set) | set(state.previous_music_track_ids) | set(state.negative_seed_ids)
    flags = text_flags(state)

    if enable_user_cf and user_store is not None and user_cf_k > 0 and not flags["exact_entity"] and (flags["broad"] or not state.positive_seed_ids):
        scores = user_store.user_track_scores(state.user_id, track_store)
        if np.isfinite(scores).any():
            scores = scores.copy()
            for track_id in excluded:
                index = track_store.track_index.get(track_id)
                if index is not None:
                    scores[index] = -np.inf
            indices, values = topk_scores(scores, user_cf_k)
            for rank, (index, value) in enumerate(zip(indices, values), start=1):
                add_candidate(bucket, track_store.track_ids[int(index)], "user_cf", rank, 0.75 * float(value) / (30.0 + rank))


def new_rows_for_group(
    group_id: int,
    state: ConversationState,
    base_set: set[str],
    tail_rows: list[dict[str, Any]],
    embedding_bucket: dict[str, dict[str, Any]],
    max_pool_size: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in tail_rows:
        item = merged.setdefault(row["track_id"], {"track_id": row["track_id"], "families": set(), "score": 0.0, "rank_cols": {}})
        item["families"].add("bm25_tail")
        item["score"] += float(row.get("nextgen_score", 0.0))
        for key, value in row.items():
            if key.startswith("rank_"):
                item["rank_cols"][key] = min(int(value), int(item["rank_cols"].get(key, 9999)))
    for track_id, emb in embedding_bucket.items():
        if track_id in base_set:
            continue
        item = merged.setdefault(track_id, {"track_id": track_id, "families": set(), "score": 0.0, "rank_cols": {}})
        item["families"].update(emb["nextgen_sources"])
        item["score"] += float(emb["rrf_score"])
        item["rank_cols"].update(emb["rank_cols"])

    allowed_new = max(0, max_pool_size - len(base_set)) if max_pool_size > 0 else len(merged)
    scored = sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:allowed_new]
    rows = []
    for item in scored:
        best_rank = min(item["rank_cols"].values()) if item["rank_cols"] else 9999
        row = {
            "group_id": group_id,
            "session_id": state.session_id,
            "user_id": state.user_id,
            "turn_number": state.turn_number,
            "track_id": item["track_id"],
            "gold_track_id": state.gold_track_id,
            "label": 1 if item["track_id"] == state.gold_track_id else 0,
            "rrf_score": float(item["score"]),
            "source_count": len(item["families"]),
            "best_source_rank": best_rank,
            "is_old_pool_candidate": 0,
            "is_beam_pool_candidate": 0,
            "is_in_both_old_and_beam": 0,
            "is_nextgen_candidate": 1,
            "nextgen_source_count": len(item["families"]),
            "nextgen_family_names": ",".join(sorted(item["families"])),
            "old_rrf_score": 0.0,
            "beam_rrf_score": 0.0,
            "old_source_count": 0.0,
            "beam_source_count": 0.0,
            "old_best_source_rank": 9999.0,
            "beam_best_source_rank": 9999.0,
            "old_rrf_rank": 9999.0,
            "beam_rrf_rank": 9999.0,
        }
        for family in NEXTGEN_FAMILIES:
            row[f"nextgen_family_{family}"] = int(family in item["families"])
        row.update(item["rank_cols"])
        rows.append(row)
    return rows


def family_summary(df: pd.DataFrame, base_hit: set[int], states: list[ConversationState]) -> pd.DataFrame:
    rows = []
    family_cols = sorted(
        col
        for col in df.columns
        if col.startswith("nextgen_family_") and col != "nextgen_family_names"
    )
    all_groups = set(range(len(states)))
    for col in family_cols:
        if col not in df.columns:
            continue
        groups = set(df.loc[(df[col].fillna(0) > 0) & (df["label"] == 1), "group_id"].astype(int))
        rows.append(
            {
                "family": col.removeprefix("nextgen_family_"),
                "gold_hit": len(groups),
                "unique_extra_over_base": len(groups - base_hit),
                "missed_before_share": len(groups - base_hit) / max(len(all_groups - base_hit), 1),
                "candidate_rows": int((df[col].fillna(0) > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("unique_extra_over_base", ascending=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = GoalFlowConfig(project_root=Path(args.project_root), tid="nextgen_candidate_pool_v1")
    config.blind_dataset_name = args.blind_dataset_name
    catalog = TrackCatalog(config.track_metadata_name)
    states = load_dev_states(config) if args.mode == "dev" else load_blind_states(config)

    old = load_old300_pool(Path(args.old300_pkl), states)
    beam = load_beam800_pool(Path(args.matrix_dir), Path(args.beam800_choice), states, args.rrf_k)
    base = make_union_pool(old, beam)
    base["is_nextgen_candidate"] = 0
    base["nextgen_source_count"] = 0
    for family in NEXTGEN_FAMILIES:
        base[f"nextgen_family_{family}"] = 0
    base["nextgen_family_names"] = ""
    base_sets = {
        int(group_id): set(group["track_id"].astype(str))
        for group_id, group in tqdm(base.groupby("group_id", sort=False), desc="Index base pool")
    }

    candidates, counts, _source_names, source_name_to_index, track_ids = source_matrix(Path(args.matrix_dir))
    text_cache = TrackTextCache(catalog)
    seed_channels = parse_seed_channels(args.seed_channels)
    seed_channel_ks = parse_seed_channel_ks(args.seed_channel_ks, args.cf_seed_k)
    required_channels = set(seed_channels)
    if args.enable_user_cf:
        required_channels.add("track_cf")
    track_store = TrackEmbeddingStore(channels={channel: TRACK_EMBEDDING_CHANNELS[channel] for channel in sorted(required_channels)})
    user_store = UserEmbeddingStore() if args.enable_user_cf else None

    rows: list[dict[str, Any]] = []
    for group_id, state in enumerate(tqdm(states, desc="Build nextgen additions")):
        base_set = base_sets[group_id]
        tail_rows = build_tail_rescue_for_group(
            group_id=group_id,
            state=state,
            base_set=base_set,
            candidates=candidates,
            counts=counts,
            source_name_to_index=source_name_to_index,
            track_ids=track_ids,
            catalog=catalog,
            text_cache=text_cache,
            max_add=args.tail_max_add,
            score_cutoff=args.tail_score_cutoff,
        )
        emb_bucket: dict[str, dict[str, Any]] = {}
        add_seed_embedding_sources_for_group(
            group_id=group_id,
            state=state,
            base_set=base_set,
            bucket=emb_bucket,
            track_store=track_store,
            seed_channels=seed_channels,
            seed_channel_ks=seed_channel_ks,
        )
        add_user_cf_for_group(
            state=state,
            base_set=base_set,
            bucket=emb_bucket,
            track_store=track_store,
            user_store=user_store,
            user_cf_k=args.user_cf_k,
            enable_user_cf=args.enable_user_cf,
        )
        rows.extend(
            new_rows_for_group(
                group_id=group_id,
                state=state,
                base_set=base_set,
                tail_rows=tail_rows,
                embedding_bucket=emb_bucket,
                max_pool_size=args.max_pool_size,
            )
        )

    additions = pd.DataFrame(rows)
    if len(additions):
        full = pd.concat([base, additions], ignore_index=True, sort=False)
    else:
        full = base.copy()
    full = normalize_pool_base(full)

    summary_rows = [
        pool_summary("union_base", base, states),
        pool_summary("nextgen_v1", full, states),
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "pool_summary.csv", index=False)
    base_hit = set(base.loc[base["label"] == 1, "group_id"].astype(int))
    family_summary(full, base_hit, states).to_csv(out_dir / "family_summary.csv", index=False)
    if len(additions):
        additions.groupby("group_id").size().describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_csv(
            out_dir / "addition_size_describe.csv"
        )
    if args.write_pkl:
        pkl_df = full.drop(columns=["nextgen_family_names"], errors="ignore")
        pkl_df.to_pickle(out_dir / "nextgen_pool.pkl")
    (out_dir / "params.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    readme = [
        "# Nextgen Candidate Pool V1",
        "",
        "Sources implemented:",
        "- adaptive BM25 tail rescue over selected deep source ranges",
        "- track CF positive/weak seed neighbor retrieval",
        "- optional user CF prior retrieval",
        "",
        "This script intentionally does not implement query-to-Qwen/CLAP/SigLIP retrieval yet because it requires query encoders not present in the official embedding tables.",
        "",
        f"Mode: `{args.mode}`",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"wrote nextgen pool experiment to {out_dir}")


if __name__ == "__main__":
    main()
