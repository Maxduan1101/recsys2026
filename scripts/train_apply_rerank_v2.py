from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from datasets import load_dataset

from augment_rerank_features_v3 import augment_features
from goalflow.data import BLIND_A_DATASET, TrackCatalog
from goalflow.pipeline import GoalFlowConfig
from goalflow.response import generate_response
from goalflow.state import ConversationState, build_state_for_blind_item
from goalflow.validation import validate_predictions
from run_rerank_v2 import load_dev_states, stable_json_hash
from run_rerank_v2_global_schema import feature_columns_for_names, prepare_features_global


DEFAULT_MODEL_SPEC = (
    "main:1.0:"
    "objective=lambdarank,n_estimators=800,learning_rate=0.03,num_leaves=31,"
    "min_child_samples=100,reg_lambda=5,reg_alpha=0,subsample=0.8,"
    "colsample_bytree=0.8,lambdarank_truncation_level=300"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train rerank_v2 on a labeled dev feature cache and apply it to a blind/dev feature cache. "
            "This is the submission-side counterpart of the OOF rerank experiments."
        )
    )
    parser.add_argument("--project-root", default="goalflow_musiccrs")
    parser.add_argument("--blind-dataset-name", default=BLIND_A_DATASET)
    parser.add_argument("--train-feature-pkl", required=True)
    parser.add_argument("--apply-feature-pkl", required=True)
    parser.add_argument("--apply-mode", choices=["blind", "dev"], default="blind")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--feature-set", choices=["all", "source", "independent"], default="all")
    parser.add_argument("--model-specs", default=DEFAULT_MODEL_SPEC)
    parser.add_argument("--rank-blend", action="store_true", help="Blend multiple models by reciprocal-log rank per group.")
    parser.add_argument("--augment-v3", action="store_true", help="Add v3 semantic features in memory before training/apply.")
    parser.add_argument(
        "--response-style",
        choices=[
            "compact",
            "compact_broad",
            "concise",
            "setwise",
            "natural",
            "polished",
            "judge_v1",
            "judge_v2",
            "judge_v3",
            "judge_mix",
            "judge_brief",
            "judge_planned",
            "judge_compact_mix",
            "judge_clean_mix",
            "judge_balanced_mix",
            "judge_clean_mix_plus",
            "judge_clean_mix_safeplus",
            "judge_clean_mix_lexplus",
            "judge_clean_mix_lexplus_softened",
        ],
        default="judge_clean_mix_lexplus_softened",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def load_blind_states(config: GoalFlowConfig) -> list[ConversationState]:
    dataset = load_dataset(config.blind_dataset_name, split="test")
    return [build_state_for_blind_item(item) for item in dataset]


def parse_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in lowered for ch in [".", "e"]):
            return float(lowered)
        return int(lowered)
    except ValueError:
        return value


def parse_model_specs(raw: str) -> list[dict[str, Any]]:
    specs = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"model spec must be name:weight:key=value,... got {item!r}")
        name, weight_raw, params_raw = parts
        params: dict[str, Any] = {}
        for pair in params_raw.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(f"bad model param {pair!r} in {item!r}")
            key, value = pair.split("=", 1)
            params[key.strip()] = parse_value(value)
        specs.append({"name": name.strip(), "weight": float(weight_raw), "params": params})
    if not specs:
        raise ValueError("no model specs parsed")
    return specs


def lgb_params(spec_params: dict[str, Any]) -> dict[str, Any]:
    params = {
        "objective": spec_params.get("objective", "lambdarank"),
        "metric": "ndcg",
        "eval_at": [20],
        "n_estimators": int(spec_params.get("n_estimators", 800)),
        "learning_rate": float(spec_params.get("learning_rate", 0.03)),
        "num_leaves": int(spec_params.get("num_leaves", 31)),
        "min_child_samples": int(spec_params.get("min_child_samples", 100)),
        "reg_lambda": float(spec_params.get("reg_lambda", 5.0)),
        "reg_alpha": float(spec_params.get("reg_alpha", 0.0)),
        "subsample": float(spec_params.get("subsample", 0.8)),
        "subsample_freq": 1,
        "colsample_bytree": float(spec_params.get("colsample_bytree", 0.8)),
        "lambdarank_truncation_level": int(spec_params.get("lambdarank_truncation_level", 50)),
        "random_state": int(spec_params.get("random_state", 2026)),
        "force_row_wise": True,
    }
    return params


def train_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict[str, Any],
) -> tuple[lgb.LGBMRanker, dict[str, int]]:
    positive_groups = set(train_df.groupby("group_id")["label"].sum().loc[lambda s: s > 0].index)
    fit_df = train_df[train_df["group_id"].isin(positive_groups)].copy().sort_values("group_id")
    group_sizes = fit_df.groupby("group_id", sort=False).size().to_list()
    categorical = [col for col in ["intent", "category", "specificity"] if col in feature_cols]
    ranker = lgb.LGBMRanker(**params)
    ranker.fit(
        fit_df[feature_cols],
        fit_df["label"],
        group=group_sizes,
        eval_at=[20],
        categorical_feature=categorical,
        callbacks=[lgb.log_evaluation(period=100)],
    )
    stats = {
        "train_groups_total": int(train_df["group_id"].nunique()),
        "train_groups_with_positive": int(len(positive_groups)),
        "train_rows": int(len(fit_df)),
        "best_iteration": int(getattr(ranker, "best_iteration_", 0) or params["n_estimators"]),
    }
    return ranker, stats


def rank_blend_scores(df: pd.DataFrame, model_scores: list[tuple[str, float, np.ndarray]]) -> np.ndarray:
    blended = np.zeros(len(df), dtype=np.float32)
    group_ids = df["group_id"].to_numpy()
    for _name, weight, scores in model_scores:
        for group_id in np.unique(group_ids):
            positions = np.flatnonzero(group_ids == group_id)
            if len(positions) == 0:
                continue
            ordered = positions[np.argsort(-scores[positions])]
            ranks = np.empty(len(ordered), dtype=np.float32)
            ranks[np.arange(len(ordered))] = np.arange(1, len(ordered) + 1, dtype=np.float32)
            # Fill by ordered index to avoid a pandas groupby object for large frames.
            contrib = np.asarray([weight / math.log2(rank + 1.0) for rank in ranks], dtype=np.float32)
            blended[ordered] += contrib
    return blended


def top_tracks_for_group(group: pd.DataFrame, top_k: int, fallback: list[str]) -> list[str]:
    selected: list[str] = []
    seen = set()
    if "final_score" in group:
        ordered = group.sort_values("final_score", ascending=False)["track_id"].astype(str)
    else:
        ordered = group["track_id"].astype(str)
    for track_id in ordered:
        if track_id not in seen:
            selected.append(track_id)
            seen.add(track_id)
        if len(selected) >= top_k:
            return selected
    for track_id in fallback:
        if track_id not in seen:
            selected.append(track_id)
            seen.add(track_id)
        if len(selected) >= top_k:
            return selected
    return selected


def write_prediction(
    states: list[ConversationState],
    scored: pd.DataFrame,
    catalog: TrackCatalog,
    out_dir: Path,
    response_style: str,
    top_k: int,
    zip_output: bool,
) -> dict[str, str]:
    grouped = {int(group_id): group for group_id, group in scored.groupby("group_id", sort=False)}
    fallback = catalog.track_ids[: max(top_k, 20)]
    rows = []
    full_rank_path = out_dir / "ranked_apply_top100.jsonl"
    with full_rank_path.open("w", encoding="utf-8") as rf:
        for group_id, state in enumerate(states):
            group = grouped.get(group_id, pd.DataFrame())
            track_ids = top_tracks_for_group(group, top_k=top_k, fallback=fallback)
            rows.append(
                {
                    "session_id": state.session_id,
                    "user_id": state.user_id,
                    "turn_number": state.turn_number,
                    "predicted_track_ids": track_ids,
                    "predicted_response": generate_response(state, catalog, track_ids, style=response_style),
                }
            )
            if len(group):
                ordered = group.sort_values("final_score", ascending=False).head(100)
                rf.write(
                    json.dumps(
                        {
                            "group_id": group_id,
                            "session_id": state.session_id,
                            "turn_number": state.turn_number,
                            "track_ids": ordered["track_id"].astype(str).to_list(),
                            "scores": [float(x) for x in ordered["final_score"].to_list()],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    validation = validate_predictions(rows, catalog, expected_count=len(states))
    if not validation["ok"]:
        raise ValueError(f"Invalid predictions: {validation}")
    pred_path = out_dir / "prediction.json"
    pred_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    paths = {"prediction": str(pred_path), "ranked_apply_top100": str(full_rank_path)}
    if zip_output:
        zip_path = out_dir / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(pred_path, arcname="prediction.json")
        paths["submission_zip"] = str(zip_path)
    return paths


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = GoalFlowConfig(project_root=Path(args.project_root), tid="train_apply_rerank_v2")
    config.blind_dataset_name = args.blind_dataset_name

    catalog = TrackCatalog(config.track_metadata_name)
    states = load_blind_states(config) if args.apply_mode == "blind" else load_dev_states(config)

    train_df = pd.read_pickle(args.train_feature_pkl)
    apply_df = pd.read_pickle(args.apply_feature_pkl)
    if args.augment_v3:
        train_df, train_added = augment_features(train_df)
        apply_df, apply_added = augment_features(apply_df)
    else:
        train_added = []
        apply_added = []

    feature_cols = feature_columns_for_names(list(train_df.columns), args.feature_set)
    train_df, category_values = prepare_features_global(train_df, feature_cols)
    apply_df, _ = prepare_features_global(apply_df, feature_cols, category_values=category_values)
    train_df = train_df.sort_values("group_id").reset_index(drop=True)
    apply_df = apply_df.sort_values("group_id").reset_index(drop=True)

    specs = parse_model_specs(args.model_specs)
    model_outputs: list[tuple[str, float, np.ndarray]] = []
    model_summaries = []
    for spec in specs:
        params = lgb_params(spec["params"])
        print(f"training {spec['name']} weight={spec['weight']} params={params}")
        ranker, stats = train_model(train_df, feature_cols, params)
        scores = ranker.predict(apply_df[feature_cols]).astype(np.float32)
        model_outputs.append((spec["name"], float(spec["weight"]), scores))
        model_summaries.append(
            {
                "name": spec["name"],
                "weight": spec["weight"],
                "params": params,
                "params_hash": stable_json_hash(params),
                **stats,
            }
        )

    if len(model_outputs) == 1 and not args.rank_blend:
        apply_df["final_score"] = model_outputs[0][2]
    else:
        apply_df["final_score"] = rank_blend_scores(apply_df, model_outputs)

    paths = write_prediction(
        states=states,
        scored=apply_df,
        catalog=catalog,
        out_dir=out_dir,
        response_style=args.response_style,
        top_k=args.top_k,
        zip_output=not args.no_zip,
    )

    summary = {
        "train_feature_pkl": args.train_feature_pkl,
        "apply_feature_pkl": args.apply_feature_pkl,
        "apply_mode": args.apply_mode,
        "feature_set": args.feature_set,
        "feature_count": len(feature_cols),
        "feature_hash": stable_json_hash(feature_cols),
        "train_rows": int(len(train_df)),
        "apply_rows": int(len(apply_df)),
        "apply_groups": int(apply_df["group_id"].nunique()),
        "rank_blend": bool(args.rank_blend),
        "augment_v3": bool(args.augment_v3),
        "v3_train_added_columns": train_added,
        "v3_apply_added_columns": apply_added,
        "models": model_summaries,
        "outputs": paths,
    }
    (out_dir / "train_apply_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:4000])


if __name__ == "__main__":
    main()
