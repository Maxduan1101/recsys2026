from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def dcg_at_20(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.sum() <= 0:
        return 0.0
    order = np.argsort(-scores)[:20]
    for rank, index in enumerate(order, start=1):
        if labels[index] > 0:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def mean_group_ndcg(df: pd.DataFrame, score_col: str, group_ids: set[int]) -> float:
    values = []
    for _group_id, group in df[df["group_id"].isin(group_ids)].groupby("group_id", sort=False):
        values.append(dcg_at_20(group["label"].to_numpy(), group[score_col].to_numpy()))
    return float(np.mean(values)) if values else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Probe LightGBM LambdaRank on exported GoalFlow LTR rows.")
    parser.add_argument("--input", required=True, help="JSONL from export_ltr_dataset.py.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--valid-mod", type=int, default=5, help="Use group_id %% valid_mod == 0 as validation.")
    parser.add_argument("--n-estimators", type=int, default=260)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            features = dict(item["features"])
            row = {
                "group_id": int(item["group_id"]),
                "session_id": item["session_id"],
                "turn_number": int(item["turn_number"]),
                "track_id": item["track_id"],
                "label": int(item["label"]),
            }
            row.update(features)
            rows.append(row)

    df = pd.DataFrame(rows)
    df["source_count"] = df.get("source_count", 0.0).fillna(0.0)
    forced_gold = (df["label"] == 1) & (df["source_count"] <= 0)
    df = df[~forced_gold].copy()

    categorical = [column for column in ["intent", "category", "specificity"] if column in df.columns]
    for column in categorical:
        df[column] = df[column].fillna("missing").astype("category")

    ignored = {"session_id", "track_id", "label"}
    feature_cols = [column for column in df.columns if column not in ignored and column != "group_id"]
    for column in feature_cols:
        if column in categorical:
            continue
        if column.startswith("rank_") or column == "best_source_rank":
            df[column] = df[column].fillna(9999.0)
        else:
            df[column] = df[column].fillna(0.0)

    all_groups = sorted(df["group_id"].unique())
    valid_groups = {group_id for group_id in all_groups if group_id % args.valid_mod == 0}
    train_groups = [group_id for group_id in all_groups if group_id not in valid_groups]

    train_df = df[df["group_id"].isin(train_groups)].copy()
    train_positive_groups = set(train_df.groupby("group_id")["label"].sum().loc[lambda s: s > 0].index)
    train_df = train_df[train_df["group_id"].isin(train_positive_groups)].copy()
    valid_df = df[df["group_id"].isin(valid_groups)].copy()

    train_df = train_df.sort_values("group_id")
    valid_df = valid_df.sort_values("group_id")
    train_group_sizes = train_df.groupby("group_id", sort=False).size().to_list()
    valid_group_sizes = valid_df.groupby("group_id", sort=False).size().to_list()

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.n_estimators,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=2026,
    )
    ranker.fit(
        train_df[feature_cols],
        train_df["label"],
        group=train_group_sizes,
        eval_set=[(valid_df[feature_cols], valid_df["label"])],
        eval_group=[valid_group_sizes],
        eval_at=[20],
        categorical_feature=categorical,
        callbacks=[lgb.log_evaluation(period=50)],
    )

    valid_df["model_score"] = ranker.predict(valid_df[feature_cols])
    valid_group_ids = set(valid_df["group_id"].unique())
    summary = {
        "input": args.input,
        "rows_after_dropping_forced_gold": int(len(df)),
        "forced_gold_dropped": int(forced_gold.sum()),
        "train_groups": int(len(train_positive_groups)),
        "valid_groups": int(len(valid_group_ids)),
        "valid_groups_with_positive": int(
            (valid_df.groupby("group_id")["label"].sum() > 0).sum()
        ),
        "baseline_rrf_ndcg20": mean_group_ndcg(valid_df, "rrf_score", valid_group_ids),
        "model_ndcg20": mean_group_ndcg(valid_df, "model_score", valid_group_ids),
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
