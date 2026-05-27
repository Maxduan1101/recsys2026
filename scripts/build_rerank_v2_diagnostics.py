from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from goalflow.pipeline import GoalFlowConfig
from run_rerank_v2 import load_dev_states, rank_bucket


DEFAULT_OUT_DIR = Path("goalflow_musiccrs/experiments/rerank_v2_independent_features")


DIAGNOSTIC_FEATURES = [
    "source_count",
    "best_source_rank",
    "best_source_bucket",
    "rrf_score",
    "old_rrf_score",
    "beam_rrf_score",
    "old_source_count",
    "beam_source_count",
    "old_best_source_rank",
    "beam_best_source_rank",
    "old_rrf_rank",
    "beam_rrf_rank",
    "is_old_pool_candidate",
    "is_beam_pool_candidate",
    "is_in_both_old_and_beam",
    "current_metadata_overlap",
    "current_metadata_jaccard",
    "goal_metadata_overlap",
    "goal_metadata_jaccard",
    "history_metadata_overlap",
    "history_metadata_jaccard",
    "positive_feedback_metadata_overlap",
    "positive_feedback_metadata_jaccard",
    "negative_feedback_metadata_overlap",
    "negative_feedback_metadata_jaccard",
    "tag_overlap_count",
    "tag_overlap_ratio",
    "candidate_artist_prev_count",
    "candidate_album_prev_count",
    "candidate_artist_accepted_count",
    "candidate_artist_rejected_count",
    "same_tags_positive_overlap",
    "same_tags_negative_overlap",
    "emb_user_cf_score",
    "emb_audio_weak_history_max",
    "emb_audio_weak_history_mean",
    "emb_audio_pos_mean",
    "emb_audio_neg_mean",
    "emb_track_cf_weak_history_max",
    "emb_track_cf_weak_history_mean",
    "emb_track_cf_pos_mean",
    "emb_track_cf_neg_mean",
    "emb_metadata_pos_mean",
    "emb_metadata_neg_mean",
    "emb_attributes_pos_mean",
    "emb_attributes_neg_mean",
    "log_popularity_prior",
    "duration_prior",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rerank v2 gold diagnostics.")
    parser.add_argument("--project-root", default="goalflow_musiccrs")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return parser.parse_args()


def read_gold_ranks(path: Path) -> dict[int, int | None]:
    ranks: dict[int, int | None] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = int(row.get("group_id", -1))
            if key < 0:
                # Older prediction files do not write group_id; recover by sequence order later.
                key = len(ranks)
            ranks[key] = row.get("gold_rank")
    return ranks


def load_gold_rows(out_dir: Path, pool: str) -> pd.DataFrame:
    df = pd.read_pickle(out_dir / f"features_{pool}.pkl")
    gold = df[df["label"] == 1].copy()
    keep = [
        "group_id",
        "session_id",
        "user_id",
        "turn_number",
        "track_id",
        "label",
        "gold_track_id",
    ]
    for col in DIAGNOSTIC_FEATURES:
        if col in gold.columns and col not in keep:
            keep.append(col)
    return gold[keep].copy()


def safe_quantile(values: pd.Series, q: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) == 0:
        return float("nan")
    return float(values.quantile(q))


def rank_count(ranks: list[int | None], limit: int) -> int:
    return int(sum(rank is not None and rank <= limit for rank in ranks))


def rank_p(values: list[int | None], q: float) -> float:
    numeric = pd.Series([rank for rank in values if rank is not None], dtype=float)
    if len(numeric) == 0:
        return float("nan")
    return float(numeric.quantile(q))


def source_count_distribution(values: pd.Series) -> str:
    counts = values.fillna(-1).astype(int).value_counts().sort_index().to_dict()
    return json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False)


def bucket_distribution(values: pd.Series) -> str:
    counts = values.fillna(-1).astype(int).value_counts().sort_index().to_dict()
    return json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False)


def pool_diagnostics(
    pool_name: str,
    gold_rows: pd.DataFrame,
    gold_ranks: dict[int, int | None],
) -> dict[str, Any]:
    ranks = [gold_ranks.get(int(group_id)) for group_id in gold_rows["group_id"]]
    if pool_name == "old300" and "old_rrf_rank" in gold_rows:
        rrf_col = "old_rrf_rank"
    elif pool_name == "beam800" and "beam_rrf_rank" in gold_rows:
        rrf_col = "beam_rrf_rank"
    elif "old_rrf_rank" in gold_rows and "beam_rrf_rank" in gold_rows:
        rrf_col = "_combined_rrf_rank"
        gold_rows = gold_rows.copy()
        gold_rows[rrf_col] = gold_rows[["old_rrf_rank", "beam_rrf_rank"]].min(axis=1)
    else:
        rrf_col = None
    return {
        "pool": pool_name,
        "gold_in_pool": int(len(gold_rows)),
        "gold_source_count_mean": float(pd.to_numeric(gold_rows.get("source_count", pd.Series(dtype=float)), errors="coerce").mean()),
        "gold_source_count_p50": safe_quantile(gold_rows.get("source_count", pd.Series(dtype=float)), 0.50),
        "gold_best_source_rank_p50": safe_quantile(gold_rows.get("best_source_rank", pd.Series(dtype=float)), 0.50),
        "gold_best_source_rank_p90": safe_quantile(gold_rows.get("best_source_rank", pd.Series(dtype=float)), 0.90),
        "gold_rrf_rank_p50": safe_quantile(gold_rows[rrf_col], 0.50) if rrf_col else float("nan"),
        "gold_ltr_pred_rank_p50": rank_p(ranks, 0.50),
        "gold_ltr_pred_rank_p90": rank_p(ranks, 0.90),
        "gold_ltr_pred_rank_le20_count": rank_count(ranks, 20),
        "gold_ltr_pred_rank_le50_count": rank_count(ranks, 50),
        "gold_ltr_pred_rank_le100_count": rank_count(ranks, 100),
        "gold_ltr_pred_rank_le300_count": rank_count(ranks, 300),
    }


def summarize_bucket(bucket: str, rows: pd.DataFrame, union_ranks: dict[int, int | None]) -> dict[str, Any]:
    ranks = [union_ranks.get(int(group_id)) for group_id in rows["group_id"]]
    out: dict[str, Any] = {
        "bucket": bucket,
        "count": int(len(rows)),
        "source_count_distribution": source_count_distribution(rows.get("source_count", pd.Series(dtype=float))),
        "best_source_bucket_distribution": bucket_distribution(rows.get("best_source_bucket", pd.Series(dtype=float))),
        "source_count_mean": float(pd.to_numeric(rows.get("source_count", pd.Series(dtype=float)), errors="coerce").mean()),
        "source_count_p50": safe_quantile(rows.get("source_count", pd.Series(dtype=float)), 0.50),
        "best_source_rank_p50": safe_quantile(rows.get("best_source_rank", pd.Series(dtype=float)), 0.50),
        "best_source_rank_p90": safe_quantile(rows.get("best_source_rank", pd.Series(dtype=float)), 0.90),
        "rrf_score_mean": float(pd.to_numeric(rows.get("rrf_score", pd.Series(dtype=float)), errors="coerce").mean()),
        "union_ltr_gold_rank_p50": rank_p(ranks, 0.50),
        "union_ltr_gold_rank_p90": rank_p(ranks, 0.90),
        "union_ltr_rank_le20_count": rank_count(ranks, 20),
        "union_ltr_rank_le50_count": rank_count(ranks, 50),
        "union_ltr_rank_le100_count": rank_count(ranks, 100),
        "union_ltr_rank_le300_count": rank_count(ranks, 300),
    }
    for col in DIAGNOSTIC_FEATURES:
        if col in rows.columns and col not in out and pd.api.types.is_numeric_dtype(rows[col]):
            out[f"{col}_mean"] = float(pd.to_numeric(rows[col], errors="coerce").mean())
            out[f"{col}_p50"] = safe_quantile(rows[col], 0.50)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    config = GoalFlowConfig(project_root=Path(args.project_root), tid="rerank_v2_independent_features")
    states = load_dev_states(config)

    old_gold = load_gold_rows(out_dir, "old300")
    beam_gold = load_gold_rows(out_dir, "beam800")
    union_gold = load_gold_rows(out_dir, "union")

    old_hit = set(old_gold["group_id"].astype(int))
    beam_hit = set(beam_gold["group_id"].astype(int))
    union_hit = set(union_gold["group_id"].astype(int))
    all_groups = set(range(len(states)))

    ranks = {
        "old300": read_gold_ranks(out_dir / "predictions_oof_old300.jsonl"),
        "beam800": read_gold_ranks(out_dir / "predictions_oof_beam800.jsonl"),
        "union": read_gold_ranks(out_dir / "predictions_oof_union.jsonl"),
    }

    pool_rows = [
        pool_diagnostics("old300", old_gold, ranks["old300"]),
        pool_diagnostics("beam800", beam_gold, ranks["beam800"]),
        pool_diagnostics("union", union_gold, ranks["union"]),
    ]
    pd.DataFrame(pool_rows).to_csv(out_dir / "pool_gold_rank_diagnostics.csv", index=False)

    bucket_groups = {
        "extra_gold_beam_not_old": sorted(beam_hit - old_hit),
        "lost_gold_old_not_beam": sorted(old_hit - beam_hit),
        "both_hit_old_and_beam": sorted(old_hit & beam_hit),
        "missed_by_old_and_beam": sorted(all_groups - union_hit),
    }
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for bucket, groups in bucket_groups.items():
        if bucket == "missed_by_old_and_beam":
            summary_rows.append(
                {
                    "bucket": bucket,
                    "count": len(groups),
                    "source_count_distribution": "{}",
                    "best_source_bucket_distribution": "{}",
                    "source_count_mean": float("nan"),
                    "source_count_p50": float("nan"),
                    "best_source_rank_p50": float("nan"),
                    "best_source_rank_p90": float("nan"),
                    "rrf_score_mean": float("nan"),
                    "union_ltr_gold_rank_p50": float("nan"),
                    "union_ltr_gold_rank_p90": float("nan"),
                    "union_ltr_rank_le20_count": 0,
                    "union_ltr_rank_le50_count": 0,
                    "union_ltr_rank_le100_count": 0,
                    "union_ltr_rank_le300_count": 0,
                }
            )
            continue
        rows = union_gold[union_gold["group_id"].isin(groups)].copy()
        summary_rows.append(summarize_bucket(bucket, rows, ranks["union"]))
        for _, row in rows.iterrows():
            group_id = int(row["group_id"])
            item = {
                "bucket": bucket,
                "group_id": group_id,
                "session_id": row.get("session_id", ""),
                "user_id": row.get("user_id", ""),
                "turn_number": int(row.get("turn_number", -1)),
                "track_id": row.get("track_id", ""),
                "union_ltr_gold_rank": ranks["union"].get(group_id),
                "old_ltr_gold_rank": ranks["old300"].get(group_id),
                "beam_ltr_gold_rank": ranks["beam800"].get(group_id),
            }
            for col in DIAGNOSTIC_FEATURES:
                if col in row:
                    value = row[col]
                    item[col] = None if pd.isna(value) else value
            detail_rows.append(item)

    pd.DataFrame(summary_rows).to_csv(out_dir / "gold_bucket_diagnostics_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(out_dir / "extra_gold_diagnostics.csv", index=False)

    missed_rows = []
    for group_id in bucket_groups["missed_by_old_and_beam"]:
        state = states[group_id]
        positive_seed_ids = getattr(state, "positive_seed_track_ids", getattr(state, "positive_seed_ids", []))
        negative_seed_ids = getattr(state, "negative_seed_track_ids", getattr(state, "negative_seed_ids", []))
        missed_rows.append(
            {
                "group_id": group_id,
                "session_id": state.session_id,
                "user_id": state.user_id,
                "turn_number": state.turn_number,
                "gold_track_id": state.gold_track_id,
                "current_user_query": state.current_user_query,
                "listener_goal": state.listener_goal,
                "category": state.category,
                "specificity": state.specificity,
                "num_positive_seed_tracks": len(positive_seed_ids),
                "num_negative_seed_tracks": len(negative_seed_ids),
            }
        )
    pd.DataFrame(missed_rows).to_csv(out_dir / "missed_gold_diagnostics.csv", index=False)
    print(f"wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
