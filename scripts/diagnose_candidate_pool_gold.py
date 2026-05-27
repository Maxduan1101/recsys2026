from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from goalflow.pipeline import default_index_weights, default_query_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose gold-track support inside candidate pools.")
    parser.add_argument("--old-pkl", required=True)
    parser.add_argument("--beam300-pkl", required=True)
    parser.add_argument("--old-pred-json", required=True)
    parser.add_argument("--beam300-pred-json", required=True)
    parser.add_argument(
        "--matrix-dir",
        default="goalflow_musiccrs/experiments/source_candidate_matrix_full_s20_k800/source_candidate_matrix",
    )
    parser.add_argument(
        "--beam800-choice",
        default="goalflow_musiccrs/experiments/source_candidate_matrix_full_s20_k800/source_candidate_matrix/beam_search_target800_strict/best_choice.tsv",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--out-dir", default="goalflow_musiccrs/experiments/candidate_pool_gold_diagnosis")
    return parser.parse_args()


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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def source_weight(source_name: str) -> float:
    if ":" not in source_name:
        return 1.0
    index_name, query_name = source_name.split(":", 1)
    return default_index_weights().get(index_name, 1.0) * default_query_weights().get(query_name, 1.0)


def read_choice(path: Path, source_name_to_index: dict[str, int]) -> dict[int, int]:
    choice: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            choice[source_name_to_index[row["source_name"]]] = int(row["selected_k"])
    return choice


def pred_rank_by_group(path: Path, examples: list[dict[str, str]]) -> dict[int, int | None]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    key_to_group = {
        (row["session_id"], row["user_id"], int(row["turn_number"])): i
        for i, row in enumerate(examples)
    }
    out: dict[int, int | None] = {}
    for row in rows:
        group_id = key_to_group[(row["session_id"], row["user_id"], int(row["turn_number"]))]
        gold = examples[group_id]["gold_track_id"]
        rank = None
        for i, track_id in enumerate(row["predicted_track_ids"], start=1):
            if track_id == gold:
                rank = i
                break
        out[group_id] = rank
    return out


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def censored_rank_percentile(known_le20_ranks: list[int], total: int, q: float) -> float | str | None:
    if total <= 0:
        return None
    target = math.ceil((q / 100.0) * total)
    known = sorted(int(value) for value in known_le20_ranks)
    if len(known) >= target:
        return float(known[target - 1])
    return ">20"


def summarize_pool_from_df(
    name: str,
    df: pd.DataFrame,
    top20_ranks: dict[int, int | None],
) -> tuple[dict[str, Any], pd.DataFrame]:
    gold = df[df["label"] == 1].copy()
    gold["gold_in_pool"] = True
    gold["gold_ltr_pred_rank"] = gold["group_id"].map(top20_ranks)
    gold["gold_ltr_pred_rank_le20"] = gold["gold_ltr_pred_rank"].notna()

    if "rrf_rank" not in df.columns:
        ranks = df.groupby("group_id")["rrf_score"].rank(method="first", ascending=False)
        gold["rrf_rank"] = ranks.loc[gold.index].astype(float)

    ltr_ranks_known = [int(item) for item in gold["gold_ltr_pred_rank"].dropna().tolist()]
    ltr_le20 = int(gold["gold_ltr_pred_rank_le20"].sum())
    summary = {
        "pool": name,
        "rows": int(len(df)),
        "groups": int(df["group_id"].nunique()),
        "gold_in_pool": int(len(gold)),
        "gold_source_count_mean": float(gold["source_count"].mean()) if len(gold) else None,
        "gold_source_count_p50": percentile(gold["source_count"].tolist(), 50),
        "gold_best_source_rank_p50": percentile(gold["best_source_rank"].tolist(), 50),
        "gold_best_source_rank_p90": percentile(gold["best_source_rank"].tolist(), 90),
        "gold_rrf_rank_p50": percentile(gold["rrf_rank"].tolist(), 50),
        "gold_ltr_pred_rank_p50": censored_rank_percentile(ltr_ranks_known, len(gold), 50),
        "gold_ltr_pred_rank_le20_count": ltr_le20,
        "gold_ltr_pred_rank_note": "top20-json only; ranks >20 are censored",
    }
    keep = [
        "group_id",
        "track_id",
        "source_count",
        "best_source_rank",
        "rrf_score",
        "rrf_rank",
        "gold_ltr_pred_rank",
    ]
    return summary, gold[keep].copy()


def summarize_beam800_from_matrix(
    matrix_dir: Path,
    choice_path: Path,
    rrf_k: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    meta = read_meta(matrix_dir)
    examples = read_tsv(matrix_dir / "examples.tsv")
    sources = read_tsv(matrix_dir / "sources.tsv")
    source_names = [row["source_name"] for row in sources]
    source_name_to_index = {row["source_name"]: int(row["source_index"]) for row in sources}
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
    gold = np.memmap(matrix_dir / "gold.i32", dtype=np.int32, mode="r", shape=(num_turns,))

    rows = []
    total_union = 0
    for turn_index in range(num_turns):
        fused: dict[int, float] = {}
        gold_index = int(gold[turn_index])
        gold_source_ranks: list[int] = []
        gold_rrf_score = 0.0
        for source_index, selected_k in choice.items():
            take = min(selected_k, int(counts[turn_index, source_index]))
            if take <= 0:
                continue
            source_name = source_names[source_index]
            weight = source_weight(source_name)
            for rank, track_index_raw in enumerate(candidates[turn_index, source_index, :take], start=1):
                track_index = int(track_index_raw)
                if track_index < 0:
                    continue
                score = weight / (rrf_k + rank)
                fused[track_index] = fused.get(track_index, 0.0) + score
                if track_index == gold_index:
                    gold_source_ranks.append(rank)
                    gold_rrf_score += score
        total_union += len(fused)
        if gold_source_ranks:
            better = sum(1 for score in fused.values() if score > gold_rrf_score)
            rows.append(
                {
                    "group_id": turn_index,
                    "track_id": examples[turn_index]["gold_track_id"],
                    "source_count": len(gold_source_ranks),
                    "best_source_rank": min(gold_source_ranks),
                    "rrf_score": gold_rrf_score,
                    "rrf_rank": better + 1,
                    "gold_ltr_pred_rank": None,
                }
            )
    gold_df = pd.DataFrame(rows)
    summary = {
        "pool": "beam800",
        "rows": None,
        "groups": num_turns,
        "avg_union_size": total_union / num_turns,
        "gold_in_pool": int(len(gold_df)),
        "gold_source_count_mean": float(gold_df["source_count"].mean()) if len(gold_df) else None,
        "gold_source_count_p50": percentile(gold_df["source_count"].tolist(), 50),
        "gold_best_source_rank_p50": percentile(gold_df["best_source_rank"].tolist(), 50),
        "gold_best_source_rank_p90": percentile(gold_df["best_source_rank"].tolist(), 90),
        "gold_rrf_rank_p50": percentile(gold_df["rrf_rank"].tolist(), 50),
        "gold_ltr_pred_rank_p50": None,
        "gold_ltr_pred_rank_le20_count": None,
        "gold_ltr_pred_rank_note": "not scored by full-pool LTR yet",
    }
    return summary, gold_df


def source_count_distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = Counter(int(value) for value in df["source_count"].dropna().tolist())
    return {str(key): counts[key] for key in sorted(counts)}


def compare_sets(label: str, base: pd.DataFrame, other: pd.DataFrame) -> dict[str, Any]:
    base_groups = set(int(item) for item in base["group_id"].tolist())
    other_groups = set(int(item) for item in other["group_id"].tolist())
    extra_groups = sorted(other_groups - base_groups)
    dropped_groups = sorted(base_groups - other_groups)
    extra = other[other["group_id"].isin(extra_groups)]
    dropped = base[base["group_id"].isin(dropped_groups)]
    extra_single_late = extra[(extra["source_count"] == 1) & (extra["best_source_rank"] > 300)]
    return {
        "comparison": label,
        "base_hit": len(base_groups),
        "other_hit": len(other_groups),
        "extra_hit_count": len(extra_groups),
        "dropped_hit_count": len(dropped_groups),
        "base_hit_source_count_distribution": source_count_distribution(base),
        "extra_hit_source_count_distribution": source_count_distribution(extra),
        "dropped_old_hit_source_count_distribution": source_count_distribution(dropped),
        "extra_hit_source_count_mean": float(extra["source_count"].mean()) if len(extra) else None,
        "extra_hit_best_source_rank_p50": percentile(extra["best_source_rank"].tolist(), 50),
        "extra_hit_best_source_rank_p90": percentile(extra["best_source_rank"].tolist(), 90),
        "extra_source_count_eq1_and_best_rank_gt300": int(len(extra_single_late)),
    }


def main() -> None:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = read_tsv(matrix_dir / "examples.tsv")
    old_ranks = pred_rank_by_group(Path(args.old_pred_json), examples)
    beam300_ranks = pred_rank_by_group(Path(args.beam300_pred_json), examples)

    old_df = pd.read_pickle(args.old_pkl)
    beam300_df = pd.read_pickle(args.beam300_pkl)
    old_summary, old_gold = summarize_pool_from_df("old300", old_df, old_ranks)
    beam300_summary, beam300_gold = summarize_pool_from_df("beam300", beam300_df, beam300_ranks)
    beam800_summary, beam800_gold = summarize_beam800_from_matrix(
        matrix_dir=matrix_dir,
        choice_path=Path(args.beam800_choice),
        rrf_k=args.rrf_k,
    )

    summaries = [old_summary, beam300_summary, beam800_summary]
    comparisons = [
        compare_sets("beam300_vs_old300", old_gold, beam300_gold),
        compare_sets("beam800_vs_old300", old_gold, beam800_gold),
    ]
    result = {"summary": summaries, "comparisons": comparisons}
    (out_dir / "gold_pool_diagnosis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(out_dir / "gold_pool_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out_dir / "gold_pool_comparisons.csv", index=False)
    print(json.dumps(result, indent=2))
    print(f"summary_csv={out_dir / 'gold_pool_summary.csv'}")
    print(f"comparison_csv={out_dir / 'gold_pool_comparisons.csv'}")


if __name__ == "__main__":
    main()
