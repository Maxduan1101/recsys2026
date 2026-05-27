from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from goalflow.pipeline import GoalFlowConfig
from goalflow.state import ConversationState
from run_rerank_v2 import (
    META_COLUMNS,
    SOURCE_PREFIXES,
    fold_for_state,
    load_dev_states,
    mean_ndcg_and_predictions,
    stable_json_hash,
)


DEFAULT_OUT_DIR = Path("goalflow_musiccrs/experiments/rerank_v2_independent_features")
DEFAULT_COMBOS = [
    "old300:all",
    "beam800:all",
    "union:all",
    "old300:source",
    "old300:independent",
    "union:source",
    "union:independent",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain rerank v2 from cached features with a global feature schema.")
    parser.add_argument("--project-root", default="goalflow_musiccrs")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--pools", default="old300,beam800,union")
    parser.add_argument("--combos", default=",".join(DEFAULT_COMBOS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--objective", default="lambdarank", choices=["lambdarank", "rank_xendcg"])
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--lambdarank-truncation-level", type=int, default=50)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    return parser.parse_args()


def load_feature_columns(out_dir: Path, pool: str) -> list[str]:
    meta_path = out_dir / f"features_{pool}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing feature metadata: {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return list(payload["columns"])


def is_source_feature(col: str) -> bool:
    if col in META_COLUMNS or col in {"pool_name", "intent", "category", "specificity", "session_fold"}:
        return False
    return col.startswith(SOURCE_PREFIXES)


def feature_columns_for_names(columns: list[str], feature_set: str) -> list[str]:
    ignored = set(META_COLUMNS) | {"pool_name", "session_fold"}
    if feature_set == "source":
        cols = [col for col in columns if is_source_feature(col)]
    elif feature_set == "independent":
        cols = [col for col in columns if col not in ignored and not is_source_feature(col)]
    elif feature_set == "all":
        cols = [col for col in columns if col not in ignored]
    else:
        raise ValueError(f"Unknown feature_set={feature_set}")
    return sorted(cols)


def global_feature_schema(pool_columns: dict[str, list[str]]) -> dict[str, list[str]]:
    all_columns: list[str] = []
    seen = set()
    for columns in pool_columns.values():
        for col in columns:
            if col not in seen:
                all_columns.append(col)
                seen.add(col)
    return {
        "source": feature_columns_for_names(all_columns, "source"),
        "independent": feature_columns_for_names(all_columns, "independent"),
        "all": feature_columns_for_names(all_columns, "all"),
    }


def prepare_features_global(
    df: pd.DataFrame,
    feature_cols: list[str],
    category_values: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = df.copy()
    missing_cols = [col for col in feature_cols if col not in out.columns]
    if missing_cols:
        additions: dict[str, Any] = {}
        for col in missing_cols:
            if col in {"intent", "category", "specificity"}:
                additions[col] = "missing"
            elif "rank" in col:
                additions[col] = np.float32(9999.0)
            elif "bucket" in col:
                additions[col] = np.float32(-1.0)
            else:
                additions[col] = np.float32(0.0)
        out = pd.concat([out, pd.DataFrame(additions, index=out.index)], axis=1)

    categorical = [col for col in ["intent", "category", "specificity"] if col in feature_cols]
    if category_values is None:
        category_values = {}
        for col in categorical:
            values = sorted(str(value) for value in out[col].fillna("missing").unique())
            category_values[col] = values or ["missing"]

    for col in categorical:
        out[col] = pd.Categorical(out[col].fillna("missing").astype(str), categories=category_values[col])

    for col in feature_cols:
        if col in categorical:
            continue
        if "rank" in col:
            out[col] = out[col].fillna(9999.0).astype(np.float32)
        elif "bucket" in col:
            out[col] = out[col].fillna(-1.0).astype(np.float32)
        else:
            out[col] = out[col].fillna(0.0).astype(np.float32)
    return out, category_values


def train_oof_global(
    df: pd.DataFrame,
    states: list[ConversationState],
    feature_cols: list[str],
    feature_set: str,
    args: argparse.Namespace,
    out_dir: Path,
    pool_name: str,
) -> dict[str, Any]:
    start = time.time()
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
        train_df, category_values = prepare_features_global(train_df, feature_cols)
        valid_df, _ = prepare_features_global(valid_df, feature_cols, category_values=category_values)
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
        del train_df, valid_df, ranker
        gc.collect()

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
    suffix = "all" if feature_set == "all" else feature_set
    fi.to_csv(out_dir / f"feature_importance_{pool_name}_{suffix}.csv", index=False)

    return {
        "method": "LTR_v2_global_schema",
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


def read_pool_summary(out_dir: Path) -> dict[str, dict[str, Any]]:
    path = out_dir / "pool_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing pool summary: {path}")
    df = pd.read_csv(path)
    return {str(row["pool"]): row.to_dict() for _, row in df.iterrows()}


def parse_combos(raw: str) -> list[tuple[str, str]]:
    combos = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Combo must be pool:feature_set, got {item}")
        pool, feature_set = item.split(":", 1)
        combos.append((pool.strip(), feature_set.strip()))
    return combos


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = GoalFlowConfig(project_root=Path(args.project_root), tid="rerank_v2_independent_features")
    states = load_dev_states(config)
    pools = [item.strip() for item in args.pools.split(",") if item.strip()]
    combos = parse_combos(args.combos)

    pool_columns = {pool: load_feature_columns(out_dir, pool) for pool in pools}
    schema = global_feature_schema(pool_columns)
    for feature_set, cols in schema.items():
        (out_dir / f"feature_columns_{feature_set}.txt").write_text("\n".join(cols) + "\n", encoding="utf-8")
    (out_dir / "feature_columns.txt").write_text("\n".join(schema["all"]) + "\n", encoding="utf-8")
    schema_payload = {
        "method": "global_schema_from_cached_features",
        "pools": pools,
        "feature_counts": {key: len(value) for key, value in schema.items()},
        "feature_hashes": {key: stable_json_hash(value) for key, value in schema.items()},
    }
    (out_dir / "feature_schema_manifest.json").write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")

    pool_summary = read_pool_summary(out_dir)
    metrics: list[dict[str, Any]] = []
    ablations: list[dict[str, Any]] = []
    params_payload = vars(args).copy()
    params_payload["global_schema_manifest"] = schema_payload
    (out_dir / "params.json").write_text(json.dumps(params_payload, indent=2), encoding="utf-8")

    combos_by_pool: dict[str, list[str]] = {}
    for pool, feature_set in combos:
        combos_by_pool.setdefault(pool, []).append(feature_set)

    for pool in pools:
        if pool not in combos_by_pool:
            continue
        feature_path = out_dir / f"features_{pool}.pkl"
        print(f"Loaded feature cache: {feature_path}")
        df = pd.read_pickle(feature_path)
        p_summary = pool_summary[pool]
        for feature_set in combos_by_pool[pool]:
            print(f"Training global-schema {pool} {feature_set} with {len(schema[feature_set])} features")
            result = train_oof_global(df, states, schema[feature_set], feature_set, args, out_dir, pool)
            row = {
                "method": result["method"],
                "pool": pool,
                "features": feature_set,
                "ranker": result["ranker"],
                "max_candidates": result["max_candidates"],
                "avg_group_size": float(p_summary["group_size_mean"]),
                "p95_group_size": float(p_summary["group_size_p95"]),
                "gold_in_pool": int(p_summary["gold_in_pool"]),
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
            ablations.append(row)
            pd.DataFrame(metrics).to_csv(out_dir / "metrics_summary.csv", index=False)
            pd.DataFrame(ablations).to_csv(out_dir / "ablation_summary.csv", index=False)
            (out_dir / f"folds_{pool}_{feature_set}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        del df
        gc.collect()

    readme = [
        "# Rerank V2 Independent Features",
        "",
        "This run retrains from cached candidate features with a single global feature schema.",
        "Candidate pools are compared under fixed split, fixed LightGBM parameters, and identical feature columns per feature set.",
        "",
        "RRF/source/rank features are auxiliary; independent lexical/history/metadata/embedding features are trained separately and together.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"done global-schema out_dir={out_dir}")


if __name__ == "__main__":
    main()
