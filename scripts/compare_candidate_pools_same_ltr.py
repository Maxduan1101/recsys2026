from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from goalflow.pipeline import GoalFlowConfig
from goalflow.state import build_state_for_dev_turn
from run_ltr_rerank import prepare_features


META_COLUMNS = {"group_id", "session_id", "user_id", "turn_number", "track_id", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two candidate pools using the same fold-wise LTR model."
    )
    parser.add_argument("--project-root", default="goalflow_musiccrs")
    parser.add_argument("--old-pkl", required=True)
    parser.add_argument("--new-pkl", required=True)
    parser.add_argument("--tid", default="candidate_pool_same_ltr_compare")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--valid-mod", type=int, default=5)
    parser.add_argument("--mode", choices=["fold", "oof"], default="fold")
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--lambdarank-truncation-level", type=int, default=30)
    return parser.parse_args()


def dcg_at_20(gold_track_id: str | None, ranked_track_ids: list[str]) -> float:
    if not gold_track_id:
        return 0.0
    for rank, track_id in enumerate(ranked_track_ids[:20], start=1):
        if track_id == gold_track_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def load_gold_by_group(config: GoalFlowConfig) -> dict[int, str | None]:
    dataset = load_dataset(config.conversation_dataset_name, split="test")
    gold_by_group: dict[int, str | None] = {}
    group_id = 0
    for item in tqdm(dataset, desc="Load dev gold"):
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id:
                gold_by_group[group_id] = state.gold_track_id
                group_id += 1
    return gold_by_group


def add_pool_marker(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    df["candidate_pool"] = name
    df["candidate_pool_is_new"] = 1.0 if name == "new" else 0.0
    return df


def train_same_model(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    train_groups: set[int],
    args: argparse.Namespace,
) -> tuple[lgb.LGBMRanker, list[str], dict[str, list[str]], dict[str, int]]:
    train_df = pd.concat(
        [
            add_pool_marker(old_df[old_df["group_id"].isin(train_groups)], "old"),
            add_pool_marker(new_df[new_df["group_id"].isin(train_groups)], "new"),
        ],
        ignore_index=True,
        sort=False,
    )
    train_df = train_df.drop(columns=["candidate_pool"], errors="ignore")
    positive_groups = set(train_df.groupby("group_id")["label"].sum().loc[lambda s: s > 0].index)
    train_df = train_df[train_df["group_id"].isin(positive_groups)].copy()
    train_df, feature_cols, categorical, category_values = prepare_features(train_df)
    train_df = train_df.sort_values(["group_id", "candidate_pool"])
    group_sizes = train_df.groupby("group_id", sort=False).size().to_list()
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=1,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        lambdarank_truncation_level=args.lambdarank_truncation_level,
        random_state=2026,
        force_row_wise=True,
    )
    ranker.fit(
        train_df[feature_cols],
        train_df["label"],
        group=group_sizes,
        eval_at=[20],
        categorical_feature=categorical,
        callbacks=[lgb.log_evaluation(period=100)],
    )
    stats = {
        "train_groups_requested": len(train_groups),
        "train_groups_with_positive": len(positive_groups),
        "train_rows": len(train_df),
    }
    return ranker, feature_cols, category_values, stats


def score_pool(
    ranker: lgb.LGBMRanker,
    df: pd.DataFrame,
    pool_name: str,
    valid_groups: set[int],
    feature_cols: list[str],
    category_values: dict[str, list[str]],
    gold_by_group: dict[int, str | None],
) -> dict[str, float | int]:
    valid_df = add_pool_marker(df[df["group_id"].isin(valid_groups)], pool_name)
    valid_df = valid_df.drop(columns=["candidate_pool"], errors="ignore")
    valid_df, _, _, _ = prepare_features(
        valid_df,
        feature_cols=feature_cols,
        category_values=category_values,
    )
    valid_df = valid_df.copy()
    valid_df["model_score"] = ranker.predict(valid_df[feature_cols])
    scores = []
    hit20 = 0
    positive_groups = int(valid_df.groupby("group_id")["label"].sum().gt(0).sum())
    for group_id, group in valid_df.groupby("group_id"):
        ranked = list(group.sort_values("model_score", ascending=False)["track_id"])
        score = dcg_at_20(gold_by_group.get(int(group_id)), ranked)
        scores.append(score)
        hit20 += int(score > 0)
    return {
        "valid_groups": len(valid_groups),
        "valid_rows": len(valid_df),
        "valid_groups_with_positive_candidates": positive_groups,
        "ndcg@20": float(sum(scores) / len(scores)) if scores else 0.0,
        "hit@20": float(hit20 / len(scores)) if scores else 0.0,
    }


def main() -> None:
    args = parse_args()
    config = GoalFlowConfig(project_root=Path(args.project_root), tid=args.tid)
    out_dir = config.experiments_dir / "ltr"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading old pool: {args.old_pkl}")
    old_df = pd.read_pickle(args.old_pkl)
    print(f"Loading new pool: {args.new_pkl}")
    new_df = pd.read_pickle(args.new_pkl)
    gold_by_group = load_gold_by_group(config)
    all_groups = sorted(set(old_df["group_id"].astype(int)) | set(new_df["group_id"].astype(int)))

    folds = [args.fold] if args.mode == "fold" else list(range(args.valid_mod))
    summaries = []
    for fold in folds:
        valid_groups = {group_id for group_id in all_groups if group_id % args.valid_mod == fold}
        train_groups = {group_id for group_id in all_groups if group_id not in valid_groups}
        ranker, feature_cols, category_values, train_stats = train_same_model(
            old_df,
            new_df,
            train_groups,
            args,
        )
        old_scores = score_pool(
            ranker,
            old_df,
            "old",
            valid_groups,
            feature_cols,
            category_values,
            gold_by_group,
        )
        new_scores = score_pool(
            ranker,
            new_df,
            "new",
            valid_groups,
            feature_cols,
            category_values,
            gold_by_group,
        )
        summaries.append(
            {
                "fold": fold,
                "train": train_stats,
                "old": old_scores,
                "new": new_scores,
                "delta_new_minus_old_ndcg@20": new_scores["ndcg@20"] - old_scores["ndcg@20"],
            }
        )

    summary = {
        "mode": args.mode,
        "tid": args.tid,
        "old_pkl": args.old_pkl,
        "new_pkl": args.new_pkl,
        "old_rows": len(old_df),
        "new_rows": len(new_df),
        "folds": summaries,
    }
    if args.mode == "oof":
        summary["old_mean_ndcg@20"] = float(sum(item["old"]["ndcg@20"] for item in summaries) / len(summaries))
        summary["new_mean_ndcg@20"] = float(sum(item["new"]["ndcg@20"] for item in summaries) / len(summaries))
        summary["delta_new_minus_old_mean_ndcg@20"] = summary["new_mean_ndcg@20"] - summary["old_mean_ndcg@20"]
    out_path = out_dir / "same_ltr_candidate_pool_compare.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary={out_path}")


if __name__ == "__main__":
    main()
