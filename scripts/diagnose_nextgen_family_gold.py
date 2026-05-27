from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose which nextgen source families retrieve and rank gold tracks.")
    parser.add_argument("--features-pkl", required=True)
    parser.add_argument("--union-pred", default="goalflow_musiccrs/experiments/rerank_v2_independent_features/predictions_oof_union.jsonl")
    parser.add_argument("--nextgen-pred", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["session_id"]), str(row["user_id"]), int(row["turn_number"]))
            rows[key] = row
    return rows


def q(values: pd.Series, quantile: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.quantile(quantile))


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        ranks = pd.to_numeric(group["nextgen_rank"], errors="coerce")
        row.update(
            {
                "count": int(len(group)),
                "rank_p50": q(ranks, 0.5),
                "rank_p75": q(ranks, 0.75),
                "rank_p90": q(ranks, 0.9),
                "rank_le20": int((ranks <= 20).sum()),
                "rank_le50": int((ranks <= 50).sum()),
                "rank_le100": int((ranks <= 100).sum()),
                "rank_le300": int((ranks <= 300).sum()),
                "track_cf_pos_mean_p50": q(group.get("emb_track_cf_pos_mean", pd.Series(dtype=float)), 0.5),
                "audio_pos_mean_p50": q(group.get("emb_audio_pos_mean", pd.Series(dtype=float)), 0.5),
                "attributes_pos_mean_p50": q(group.get("emb_attributes_pos_mean", pd.Series(dtype=float)), 0.5),
                "weak_history_audio_mean_p50": q(group.get("emb_audio_weak_history_mean", pd.Series(dtype=float)), 0.5),
                "weak_history_attributes_mean_p50": q(group.get("emb_attributes_weak_history_mean", pd.Series(dtype=float)), 0.5),
                "current_metadata_jaccard_p50": q(group.get("current_metadata_jaccard", pd.Series(dtype=float)), 0.5),
                "goal_metadata_jaccard_p50": q(group.get("goal_metadata_jaccard", pd.Series(dtype=float)), 0.5),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["count"], ascending=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    union = load_jsonl(Path(args.union_pred))
    nextgen = load_jsonl(Path(args.nextgen_pred))

    features = pd.read_pickle(args.features_pkl)
    gold = features.loc[features["label"] == 1].copy()
    family_cols = sorted(col for col in gold.columns if col.startswith("nextgen_family_"))
    family_names = [col.removeprefix("nextgen_family_") for col in family_cols]

    rank_rows = []
    for key, next_row in nextgen.items():
        union_row = union.get(key)
        union_rank = None if union_row is None else union_row.get("gold_rank")
        next_rank = next_row.get("gold_rank")
        if union_rank is None and next_rank is not None:
            delta_type = "extra_gold_in_pool"
        elif union_rank is not None and next_rank is None:
            delta_type = "lost_gold_from_pool"
        elif union_rank is not None and next_rank is not None:
            delta_type = "both_in_pool"
        else:
            delta_type = "both_absent"
        rank_rows.append(
            {
                "session_id": key[0],
                "user_id": key[1],
                "turn_number": key[2],
                "union_rank": union_rank,
                "nextgen_rank": next_rank,
                "delta_type": delta_type,
            }
        )
    ranks = pd.DataFrame(rank_rows)
    gold = gold.merge(ranks, on=["session_id", "user_id", "turn_number"], how="left")

    def active_families(row: pd.Series) -> str:
        names = [name for name, col in zip(family_names, family_cols) if int(row.get(col, 0) or 0) > 0]
        return "+".join(names) if names else "base_union"

    gold["active_nextgen_families"] = gold.apply(active_families, axis=1)
    gold["nextgen_rank"] = pd.to_numeric(gold["nextgen_rank"], errors="coerce")
    gold["union_rank"] = pd.to_numeric(gold["union_rank"], errors="coerce")
    gold["nextgen_top20"] = (gold["nextgen_rank"] <= 20).astype(int)
    gold["union_top20"] = (gold["union_rank"] <= 20).astype(int)

    keep_cols = [
        "session_id",
        "user_id",
        "turn_number",
        "track_id",
        "delta_type",
        "union_rank",
        "nextgen_rank",
        "active_nextgen_families",
        "rrf_score",
        "source_count",
        "best_source_rank",
        "nextgen_source_count",
        "emb_track_cf_pos_mean",
        "emb_audio_pos_mean",
        "emb_attributes_pos_mean",
        "emb_track_cf_weak_history_mean",
        "emb_audio_weak_history_mean",
        "emb_attributes_weak_history_mean",
        "current_metadata_jaccard",
        "goal_metadata_jaccard",
        "tag_overlap_ratio",
    ]
    keep_cols = [col for col in keep_cols if col in gold.columns]
    gold[keep_cols].to_csv(out_dir / "gold_family_rank_diagnostics.csv", index=False)

    summary_family = summarize(gold, ["delta_type", "active_nextgen_families"])
    summary_family.to_csv(out_dir / "gold_family_rank_summary_by_family.csv", index=False)

    one_hot_rows = []
    for family, col in zip(family_names, family_cols):
        subset = gold.loc[gold[col].fillna(0).astype(int) > 0].copy()
        if subset.empty:
            continue
        subset["family"] = family
        one_hot_rows.append(subset)
    if one_hot_rows:
        exploded = pd.concat(one_hot_rows, ignore_index=True)
        summarize(exploded, ["delta_type", "family"]).to_csv(
            out_dir / "gold_family_rank_summary_exploded.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(out_dir / "gold_family_rank_summary_exploded.csv", index=False)

    print(f"wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
