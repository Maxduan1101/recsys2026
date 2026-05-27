from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from goalflow.data import TrackCatalog, as_text, normalize_text
from goalflow.pipeline import GoalFlowConfig
from run_rerank_v2 import (
    TrackTextCache,
    jaccard,
    load_dev_states,
    overlap_count,
    query_context,
    read_meta,
    tokenize,
)


DEFAULT_OUT_DIR = Path("goalflow_musiccrs/experiments/rerank_v2_independent_features")
DEFAULT_MATRIX_DIR = Path("goalflow_musiccrs/experiments/source_candidate_matrix_full_s20_k800/source_candidate_matrix")
DEFAULT_CHOICE = DEFAULT_MATRIX_DIR / "beam_search_target800_strict" / "best_choice.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep dive into gold tracks missed by current pools.")
    parser.add_argument("--project-root", default="goalflow_musiccrs")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--matrix-dir", default=str(DEFAULT_MATRIX_DIR))
    parser.add_argument("--choice-path", default=str(DEFAULT_CHOICE))
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_choice(path: Path, source_name_to_index: dict[str, int]) -> dict[int, int]:
    choice: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            choice[source_name_to_index[row["source_name"]]] = int(row["selected_k"])
    return choice


def load_bucket_sets(out_dir: Path, num_groups: int) -> tuple[set[int], set[int], set[int], dict[int, str]]:
    detail = pd.read_csv(out_dir / "extra_gold_diagnostics.csv")
    extra = set(detail.loc[detail["bucket"] == "extra_gold_beam_not_old", "group_id"].astype(int))
    lost = set(detail.loc[detail["bucket"] == "lost_gold_old_not_beam", "group_id"].astype(int))
    both = set(detail.loc[detail["bucket"] == "both_hit_old_and_beam", "group_id"].astype(int))
    union_hit = extra | lost | both
    bucket = {}
    for group_id in range(num_groups):
        if group_id in both:
            bucket[group_id] = "both_hit"
        elif group_id in extra:
            bucket[group_id] = "beam_extra"
        elif group_id in lost:
            bucket[group_id] = "old_only_lost_by_beam"
        elif group_id not in union_hit:
            bucket[group_id] = "missed_union"
        else:
            bucket[group_id] = "unknown"
    return extra | both, lost | both, union_hit, bucket


def source_rank_matrix(matrix_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    meta = read_meta(matrix_dir)
    source_rows = read_tsv(matrix_dir / "sources.tsv")
    track_rows = read_tsv(matrix_dir / "track_ids.tsv")
    source_names = [row["source_name"] for row in source_rows]
    track_ids = [row["track_id"] for row in track_rows]
    num_turns = int(meta["num_turns"])
    num_sources = int(meta["num_sources"])
    max_k = int(meta["max_k"])
    candidates = np.memmap(
        matrix_dir / str(meta["candidate_file"]),
        dtype=np.int32,
        mode="r",
        shape=(num_turns, num_sources, max_k),
    )
    gold = np.memmap(matrix_dir / str(meta["gold_file"]), dtype=np.int32, mode="r", shape=(num_turns,))
    ranks = np.zeros((num_turns, num_sources), dtype=np.int16)
    for source_index in range(num_sources):
        matches = candidates[:, source_index, :] == gold[:, None]
        hit = matches.any(axis=1)
        if hit.any():
            ranks[hit, source_index] = matches.argmax(axis=1)[hit].astype(np.int16) + 1
    return ranks, np.asarray(gold, dtype=np.int32), source_names, track_ids


def selected_hit_info(ranks: np.ndarray, selected_k: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    selected_hit = np.zeros(ranks.shape[0], dtype=bool)
    selected_beyond = np.zeros(ranks.shape[0], dtype=bool)
    for source_index, k in selected_k.items():
        r = ranks[:, source_index]
        selected_hit |= (r > 0) & (r <= k)
        selected_beyond |= r > k
    return selected_hit, selected_beyond


def count_ranks_under(ranks: np.ndarray, k: int) -> np.ndarray:
    return ((ranks > 0) & (ranks <= k)).sum(axis=1)


def first_hit_source(ranks: np.ndarray, source_names: list[str]) -> tuple[list[str], list[int]]:
    names = []
    rank_values = []
    for row in ranks:
        hit_sources = np.flatnonzero(row > 0)
        if len(hit_sources) == 0:
            names.append("")
            rank_values.append(0)
            continue
        best = min(hit_sources, key=lambda source_index: int(row[source_index]))
        names.append(source_names[int(best)])
        rank_values.append(int(row[int(best)]))
    return names, rank_values


def normalized_contains(needle: str, haystack: str) -> int:
    return int(bool(needle and len(needle) >= 3 and needle in haystack))


def quantile(series: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) == 0:
        return float("nan")
    return float(numeric.quantile(q))


def bool_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(pd.to_numeric(series, errors="coerce").fillna(0).mean())


def summarize_bucket(df: pd.DataFrame, bucket: str) -> dict[str, Any]:
    part = df[df["bucket"] == bucket]
    return {
        "bucket": bucket,
        "count": int(len(part)),
        "category_top": json.dumps(part["category"].value_counts().head(8).to_dict(), ensure_ascii=False),
        "specificity_top": json.dumps(part["specificity"].value_counts().to_dict(), ensure_ascii=False),
        "turn_mean": float(part["turn_number"].mean()),
        "turn_p50": quantile(part["turn_number"], 0.5),
        "positive_seed_mean": float(part["num_positive_seeds"].mean()),
        "negative_seed_mean": float(part["num_negative_seeds"].mean()),
        "prev_music_mean": float(part["num_previous_music"].mean()),
        "popularity_mean": float(part["gold_popularity"].mean()),
        "popularity_p50": quantile(part["gold_popularity"], 0.5),
        "release_year_p50": quantile(part["gold_release_year"], 0.5),
        "tag_count_p50": quantile(part["gold_tag_count"], 0.5),
        "current_metadata_overlap_p50": quantile(part["current_metadata_overlap"], 0.5),
        "goal_metadata_overlap_p50": quantile(part["goal_metadata_overlap"], 0.5),
        "history_metadata_overlap_p50": quantile(part["history_metadata_overlap"], 0.5),
        "tag_overlap_ratio_p50": quantile(part["tag_overlap_ratio"], 0.5),
        "title_in_current_rate": bool_rate(part["title_in_current"]),
        "artist_in_current_rate": bool_rate(part["artist_in_current"]),
        "artist_in_history_rate": bool_rate(part["artist_in_history"]),
        "same_artist_positive_rate": bool_rate(part["same_artist_positive"]),
        "same_album_positive_rate": bool_rate(part["same_album_positive"]),
        "source_count_top800_mean": float(part["source_count_top800"].mean()),
        "source_count_top800_p50": quantile(part["source_count_top800"], 0.5),
        "best_all_source_rank_p50": quantile(part["best_all_source_rank"], 0.5),
        "best_all_source_rank_p90": quantile(part["best_all_source_rank"], 0.9),
        "all20_top800_hit_rate": bool_rate(part["all20_top800_hit"]),
        "selected_beam_topk_hit_rate": bool_rate(part["selected_beam_topk_hit"]),
        "selected_source_beyond_k_rate": bool_rate(part["selected_source_beyond_k"]),
        "unselected_source_top800_hit_rate": bool_rate(part["unselected_source_top800_hit"]),
    }


def source_recovery_rows(df: pd.DataFrame, source_names: list[str], ranks: np.ndarray, selected_k: dict[int, int]) -> list[dict[str, Any]]:
    missed_groups = df.loc[df["bucket"] == "missed_union", "group_id"].astype(int).to_numpy()
    rows = []
    for source_index, source_name in enumerate(source_names):
        r = ranks[missed_groups, source_index]
        selected = selected_k.get(source_index, 0)
        rows.append(
            {
                "source_index": source_index,
                "source_name": source_name,
                "selected_k": selected,
                "missed_gold_hit_top50": int(((r > 0) & (r <= 50)).sum()),
                "missed_gold_hit_top100": int(((r > 0) & (r <= 100)).sum()),
                "missed_gold_hit_top200": int(((r > 0) & (r <= 200)).sum()),
                "missed_gold_hit_top400": int(((r > 0) & (r <= 400)).sum()),
                "missed_gold_hit_top800": int(((r > 0) & (r <= 800)).sum()),
                "hit_beyond_selected_k": int((r > selected).sum()) if selected > 0 else int((r > 0).sum()),
                "median_rank_when_hit": quantile(pd.Series(r[r > 0]), 0.5) if (r > 0).any() else float("nan"),
            }
        )
    return rows


def tag_overrepresentation(df: pd.DataFrame, catalog: TrackCatalog) -> list[dict[str, Any]]:
    missed = df[df["bucket"] == "missed_union"]
    hit = df[df["bucket"] != "missed_union"]
    missed_counter: Counter[str] = Counter()
    hit_counter: Counter[str] = Counter()
    for track_id in missed["gold_track_id"]:
        missed_counter.update(catalog.tag_words(track_id))
    for track_id in hit["gold_track_id"]:
        hit_counter.update(catalog.tag_words(track_id))
    rows = []
    total_missed = max(sum(missed_counter.values()), 1)
    total_hit = max(sum(hit_counter.values()), 1)
    for tag, missed_count in missed_counter.most_common(120):
        hit_count = hit_counter.get(tag, 0)
        missed_rate = missed_count / total_missed
        hit_rate = hit_count / total_hit
        rows.append(
            {
                "tag_word": tag,
                "missed_count": missed_count,
                "hit_count": hit_count,
                "missed_rate": missed_rate,
                "hit_rate": hit_rate,
                "missed_vs_hit_ratio": missed_rate / max(hit_rate, 1e-9),
            }
        )
    return sorted(rows, key=lambda row: row["missed_vs_hit_ratio"], reverse=True)


def simple_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    columns = list(view.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.4g}")
            else:
                text = str(value).replace("|", "\\|").replace("\n", " ")
                values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    matrix_dir = Path(args.matrix_dir)
    config = GoalFlowConfig(project_root=Path(args.project_root), tid="rerank_v2_independent_features")

    catalog = TrackCatalog(config.track_metadata_name)
    states = load_dev_states(config)
    text_cache = TrackTextCache(catalog)
    old_hit, beam_hit, union_hit, bucket_by_group = load_bucket_sets(out_dir, len(states))
    ranks, gold_indices, source_names, track_ids_by_index = source_rank_matrix(matrix_dir)
    source_name_to_index = {name: index for index, name in enumerate(source_names)}
    selected_k = read_choice(Path(args.choice_path), source_name_to_index)
    selected_topk_hit, selected_beyond = selected_hit_info(ranks, selected_k)
    first_names, first_ranks = first_hit_source(ranks, source_names)

    rows = []
    for group_id, state in enumerate(states):
        gold_track_id = state.gold_track_id
        if not gold_track_id or not catalog.has_track(gold_track_id):
            continue
        view = catalog.view(gold_track_id)
        ctx = query_context(state, catalog)
        title_norm = text_cache.title_norm[gold_track_id]
        artist_norm = text_cache.artist_norm[gold_track_id]
        album_norm = text_cache.album_norm[gold_track_id]
        title = text_cache.title_tokens[gold_track_id]
        artist = text_cache.artist_tokens[gold_track_id]
        album = text_cache.album_tokens[gold_track_id]
        tags = text_cache.tag_tokens[gold_track_id]
        metadata = text_cache.metadata_tokens[gold_track_id]
        source_row = ranks[group_id]
        source_count_top800 = int((source_row > 0).sum())
        source_count_top400 = int(((source_row > 0) & (source_row <= 400)).sum())
        source_count_top200 = int(((source_row > 0) & (source_row <= 200)).sum())
        source_count_top100 = int(((source_row > 0) & (source_row <= 100)).sum())
        source_count_top50 = int(((source_row > 0) & (source_row <= 50)).sum())
        hit_sources = np.flatnonzero(source_row > 0)
        best_rank = int(source_row[hit_sources].min()) if len(hit_sources) else 0
        unselected_hit = any(source_row[source_index] > 0 and selected_k.get(source_index, 0) == 0 for source_index in range(len(source_names)))
        selected_beyond_k = bool(selected_beyond[group_id])

        pos_artists = {catalog.normalized_field(track_id, "artist_name") for track_id in state.positive_seed_ids if catalog.has_track(track_id)}
        neg_artists = {catalog.normalized_field(track_id, "artist_name") for track_id in state.negative_seed_ids if catalog.has_track(track_id)}
        pos_albums = {catalog.normalized_field(track_id, "album_name") for track_id in state.positive_seed_ids if catalog.has_track(track_id)}
        neg_albums = {catalog.normalized_field(track_id, "album_name") for track_id in state.negative_seed_ids if catalog.has_track(track_id)}
        previous_artists = {
            catalog.normalized_field(track_id, "artist_name")
            for track_id in state.previous_music_track_ids
            if catalog.has_track(track_id)
        }
        current_goal_history = " ".join([state.current_user_query, state.listener_goal, ctx["history_user_text"]]).lower()
        row = {
            "group_id": group_id,
            "bucket": bucket_by_group[group_id],
            "old_hit": int(group_id in old_hit),
            "beam_hit": int(group_id in beam_hit),
            "union_hit": int(group_id in union_hit),
            "all20_top800_hit": int(source_count_top800 > 0),
            "selected_beam_topk_hit": int(selected_topk_hit[group_id]),
            "selected_source_beyond_k": int(selected_beyond_k),
            "unselected_source_top800_hit": int(unselected_hit),
            "source_count_top50": source_count_top50,
            "source_count_top100": source_count_top100,
            "source_count_top200": source_count_top200,
            "source_count_top400": source_count_top400,
            "source_count_top800": source_count_top800,
            "best_all_source_rank": best_rank,
            "best_all_source_name": first_names[group_id],
            "best_all_source_rank_check": first_ranks[group_id],
            "session_id": state.session_id,
            "user_id": state.user_id,
            "turn_number": state.turn_number,
            "category": state.category,
            "specificity": state.specificity,
            "gold_track_id": gold_track_id,
            "gold_track_name": view.track_name,
            "gold_artist_name": view.artist_name,
            "gold_album_name": view.album_name,
            "gold_popularity": view.popularity,
            "gold_release_year": text_cache.release_year[gold_track_id],
            "gold_duration": text_cache.duration[gold_track_id],
            "gold_tag_count": len(tags),
            "gold_metadata_token_count": len(metadata),
            "num_previous_music": len(state.previous_music_track_ids),
            "num_positive_seeds": len(state.positive_seed_ids),
            "num_negative_seeds": len(state.negative_seed_ids),
            "current_query_token_count": len(ctx["current_tokens"]),
            "goal_token_count": len(ctx["goal_tokens"]),
            "history_token_count": len(ctx["history_tokens"]),
            "current_metadata_overlap": overlap_count(ctx["current_tokens"], metadata),
            "current_metadata_jaccard": jaccard(ctx["current_tokens"], metadata),
            "goal_metadata_overlap": overlap_count(ctx["goal_tokens"], metadata),
            "goal_metadata_jaccard": jaccard(ctx["goal_tokens"], metadata),
            "history_metadata_overlap": overlap_count(ctx["history_tokens"], metadata),
            "history_metadata_jaccard": jaccard(ctx["history_tokens"], metadata),
            "positive_metadata_overlap": overlap_count(ctx["positive_tokens"], metadata),
            "negative_metadata_overlap": overlap_count(ctx["negative_tokens"], metadata),
            "tag_overlap_count": overlap_count(ctx["all_tokens"], tags),
            "tag_overlap_ratio": overlap_count(ctx["all_tokens"], tags) / max(len(tags), 1),
            "title_in_current": normalized_contains(title_norm, normalize_text(state.current_user_query)),
            "artist_in_current": normalized_contains(artist_norm, normalize_text(state.current_user_query)),
            "album_in_current": normalized_contains(album_norm, normalize_text(state.current_user_query)),
            "title_in_goal": normalized_contains(title_norm, normalize_text(state.listener_goal)),
            "artist_in_goal": normalized_contains(artist_norm, normalize_text(state.listener_goal)),
            "album_in_goal": normalized_contains(album_norm, normalize_text(state.listener_goal)),
            "title_in_history": normalized_contains(title_norm, normalize_text(ctx["history_user_text"])),
            "artist_in_history": normalized_contains(artist_norm, normalize_text(ctx["history_user_text"])),
            "album_in_history": normalized_contains(album_norm, normalize_text(ctx["history_user_text"])),
            "same_artist_positive": int(artist_norm in pos_artists),
            "same_artist_negative": int(artist_norm in neg_artists),
            "same_artist_previous": int(artist_norm in previous_artists),
            "same_album_positive": int(album_norm in pos_albums),
            "same_album_negative": int(album_norm in neg_albums),
            "query_mentions_exact_lyrics": int("exact lyric" in current_goal_history or "lyrics" in current_goal_history),
            "query_mentions_cover_art": int(
                "cover art" in current_goal_history
                or "album cover" in current_goal_history
                or "artwork" in current_goal_history
                or "cover image" in current_goal_history
            ),
            "query_mentions_popularity": int("popular" in current_goal_history or "well-known" in current_goal_history or "widely recognized" in current_goal_history),
            "query_mentions_similar": int("similar" in current_goal_history or "like this" in current_goal_history or "more like" in current_goal_history),
            "query_mentions_instrumental": int("instrumental" in current_goal_history),
            "query_mentions_female_vocal": int("female" in current_goal_history and "vocal" in current_goal_history),
            "current_user_query": state.current_user_query,
            "listener_goal": state.listener_goal,
            "gold_tags": view.tag_list,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "missed_gold_deep_dive_groups.csv", index=False)
    bucket_stats = pd.DataFrame([summarize_bucket(df, bucket) for bucket in ["both_hit", "beam_extra", "old_only_lost_by_beam", "missed_union"]])
    bucket_stats.to_csv(out_dir / "missed_gold_deep_dive_bucket_stats.csv", index=False)
    category = (
        df.groupby(["bucket", "category", "specificity"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["bucket", "count"], ascending=[True, False])
    )
    category.to_csv(out_dir / "missed_gold_deep_dive_category_specificity.csv", index=False)
    pd.DataFrame(source_recovery_rows(df, source_names, ranks, selected_k)).to_csv(
        out_dir / "missed_gold_source_recovery.csv",
        index=False,
    )
    pd.DataFrame(tag_overrepresentation(df, catalog)).to_csv(
        out_dir / "missed_gold_tag_overrepresentation.csv",
        index=False,
    )
    examples = df[df["bucket"] == "missed_union"].sort_values(
        ["all20_top800_hit", "source_count_top800", "turn_number", "category"],
        ascending=[False, False, True, True],
    )
    examples.head(120).to_csv(out_dir / "missed_gold_examples.csv", index=False)

    missed = df[df["bucket"] == "missed_union"]
    recoverable = missed[missed["all20_top800_hit"] == 1]
    unrecovered = missed[missed["all20_top800_hit"] == 0]
    source_recovery = pd.read_csv(out_dir / "missed_gold_source_recovery.csv")
    top_recovery = source_recovery.sort_values("missed_gold_hit_top800", ascending=False).head(10)
    top_category = (
        missed.groupby(["category", "specificity"]).size().reset_index(name="count").sort_values("count", ascending=False).head(12)
    )
    lines = [
        "# Missed Gold Deep Dive",
        "",
        "This analyzes gold tracks not recalled by the current `old300 ∪ beam800` union pool.",
        "",
        "## Headline",
        "",
        f"- Total turns: {len(df)}.",
        f"- Union recalled gold: {int(df['union_hit'].sum())}.",
        f"- Union missed gold: {len(missed)}.",
        f"- Missed but present in at least one of the 20 source top800 lists: {len(recoverable)}.",
        f"- Missed by all 20 source top800 lists: {len(unrecovered)}.",
        f"- Recoverable missed share: {len(recoverable) / max(len(missed), 1):.3f}.",
        "",
        "## Bucket Stats",
        "",
        simple_markdown_table(bucket_stats),
        "",
        "## Top Missed Category / Specificity Buckets",
        "",
        simple_markdown_table(top_category),
        "",
        "## Sources That Could Recover Missed Gold",
        "",
        simple_markdown_table(top_recovery[
            [
                "source_name",
                "selected_k",
                "missed_gold_hit_top50",
                "missed_gold_hit_top100",
                "missed_gold_hit_top200",
                "missed_gold_hit_top400",
                "missed_gold_hit_top800",
                "median_rank_when_hit",
            ]
        ]),
        "",
        "## Interpretation",
        "",
        "- If a missed gold is present in all-source top800, the recall problem is mostly source-budget selection or rank depth.",
        "- If a missed gold is absent from all-source top800, the current BM25-style text sources cannot see it; those cases need new retrieval signals.",
        "- Compare `current_metadata_overlap`, `history_metadata_overlap`, and `tag_overlap_ratio` in the bucket stats to see whether misses lack lexical anchors.",
        "- High missed counts in later turns with many positive seeds suggest seed-similarity retrieval should be improved, especially embedding/audio/CF seed retrieval.",
    ]
    (out_dir / "missed_gold_deep_dive.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote missed gold deep dive to {out_dir}")


if __name__ == "__main__":
    main()
