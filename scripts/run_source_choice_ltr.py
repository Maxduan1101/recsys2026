from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from export_ltr_dataset import candidate_features
from goalflow.data import TrackCatalog
from goalflow.fusion import CandidateScore
from goalflow.pipeline import GoalFlowConfig, default_index_weights, default_query_weights
from goalflow.response import generate_response
from goalflow.validation import validate_predictions
from run_ltr_rerank import (
    dcg_at_20,
    load_dev_states,
    merge_with_legacy,
    prepare_features,
    write_candidate_frame_cache,
    load_candidate_frame_cache,
)


META_COLUMNS = {"session_id", "user_id", "track_id", "label"}
CACHE_VERSION = "source_choice_ltr_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OOF LightGBM LTR on exported source-choice candidate matrix."
    )
    parser.add_argument(
        "--matrix-dir",
        default="goalflow_musiccrs/experiments/source_candidate_matrix_full_s20_k800/source_candidate_matrix",
    )
    parser.add_argument("--choice", required=True)
    parser.add_argument("--tid", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--mode", choices=["validate", "oof-dev"], default="validate")
    parser.add_argument("--max-candidates-per-group", type=int, default=300)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--valid-mod", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--lambdarank-truncation-level", type=int, default=30)
    parser.add_argument("--preserve-head-k", type=int, default=0)
    parser.add_argument("--preserve-grid", default="0,5,10,15,18,20")
    parser.add_argument("--response-style", default="judge_clean_mix")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-cache-write", action="store_true")
    parser.add_argument("--copy-to-official-evaluator", action="store_true")
    return parser.parse_args()


def read_meta(matrix_dir: Path) -> dict[str, object]:
    meta: dict[str, object] = {}
    with (matrix_dir / "meta.txt").open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            key, value = parts
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


def build_fused_candidates(
    candidates: np.memmap,
    counts: np.memmap,
    turn_index: int,
    choice: dict[int, int],
    source_names: list[str],
    track_ids: list[str],
    rrf_k: int,
) -> dict[str, CandidateScore]:
    fused: dict[str, CandidateScore] = {}
    for source_index, selected_k in choice.items():
        count = int(counts[turn_index, source_index])
        take = min(selected_k, count)
        if take <= 0:
            continue
        source_name = source_names[source_index]
        weight = source_weight(source_name)
        row = candidates[turn_index, source_index, :take]
        for rank, track_index in enumerate(row, start=1):
            track_index = int(track_index)
            if track_index < 0:
                continue
            track_id = track_ids[track_index]
            item = fused.setdefault(track_id, CandidateScore(track_id=track_id, score=0.0))
            item.score += weight / (rrf_k + rank)
            item.source_ranks[source_name] = rank
    return fused


def legacy_order_from_matrix(
    candidates: np.memmap,
    counts: np.memmap,
    turn_index: int,
    source_name_to_index: dict[str, int],
    track_ids: list[str],
) -> list[str]:
    source_index = source_name_to_index.get("legacy_metadata:legacy_history")
    if source_index is None:
        return []
    count = int(counts[turn_index, source_index])
    return [track_ids[int(item)] for item in candidates[turn_index, source_index, :count] if int(item) >= 0]


def candidate_cache_key(args: argparse.Namespace) -> str:
    payload = {
        "version": CACHE_VERSION,
        "matrix_dir": str(Path(args.matrix_dir).resolve()),
        "choice": str(Path(args.choice).resolve()),
        "max_candidates_per_group": args.max_candidates_per_group,
        "rrf_k": args.rrf_k,
    }
    import hashlib

    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"sourceChoice_{digest}"


def candidate_cache_paths(project_root: Path, key: str) -> tuple[Path, Path]:
    cache_dir = project_root / "cache" / "source_choice_ltr_frames"
    return cache_dir / f"{key}.pkl", cache_dir / f"{key}.json"


def load_cache(project_root: Path, key: str, states) -> tuple[pd.DataFrame, dict[int, list[str]]] | None:
    df_path, meta_path = candidate_cache_paths(project_root, key)
    if not df_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("state_count") != len(states):
        return None
    legacy = {int(group_id): track_ids for group_id, track_ids in meta["legacy_by_group"].items()}
    print(f"Loaded source-choice LTR cache: {df_path}")
    return pd.read_pickle(df_path), legacy


def write_cache(project_root: Path, key: str, states, df: pd.DataFrame, legacy_by_group: dict[int, list[str]]) -> None:
    df_path, meta_path = candidate_cache_paths(project_root, key)
    df_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(df_path)
    meta = {
        "state_count": len(states),
        "rows": len(df),
        "legacy_by_group": {str(group_id): track_ids for group_id, track_ids in legacy_by_group.items()},
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    print(f"Wrote source-choice LTR cache: {df_path}")


def build_candidate_frame(args, states, catalog: TrackCatalog) -> tuple[pd.DataFrame, dict[int, list[str]]]:
    matrix_dir = Path(args.matrix_dir)
    meta = read_meta(matrix_dir)
    num_turns = int(meta["num_turns"])
    num_sources = int(meta["num_sources"])
    max_k = int(meta["max_k"])
    if num_turns != len(states):
        raise ValueError(f"Matrix turns={num_turns} but states={len(states)}")
    candidates = np.memmap(matrix_dir / str(meta["candidate_file"]), dtype=np.int32, mode="r", shape=(num_turns, num_sources, max_k))
    counts = np.memmap(matrix_dir / str(meta["counts_file"]), dtype=np.uint16, mode="r", shape=(num_turns, num_sources))
    source_rows = read_tsv(matrix_dir / "sources.tsv")
    track_rows = read_tsv(matrix_dir / "track_ids.tsv")
    example_rows = read_tsv(matrix_dir / "examples.tsv")
    source_names = [row["source_name"] for row in source_rows]
    source_name_to_index = {row["source_name"]: int(row["source_index"]) for row in source_rows}
    track_ids = [row["track_id"] for row in track_rows]
    choice = read_choice(Path(args.choice), source_name_to_index)

    rows = []
    legacy_by_group: dict[int, list[str]] = {}
    groups_with_positive = 0
    for group_id, state in enumerate(tqdm(states, desc="Build source-choice LTR candidates")):
        example = example_rows[group_id]
        if example["session_id"] != state.session_id or int(example["turn_number"]) != state.turn_number:
            raise ValueError(f"Matrix example order mismatch at group {group_id}")
        fused = build_fused_candidates(
            candidates=candidates,
            counts=counts,
            turn_index=group_id,
            choice=choice,
            source_names=source_names,
            track_ids=track_ids,
            rrf_k=args.rrf_k,
        )
        legacy_by_group[group_id] = legacy_order_from_matrix(
            candidates,
            counts,
            group_id,
            source_name_to_index,
            track_ids,
        )
        candidate_items = sorted(fused.values(), key=lambda item: item.score, reverse=True)
        if args.max_candidates_per_group > 0:
            candidate_items = candidate_items[: args.max_candidates_per_group]
        if state.gold_track_id in {item.track_id for item in candidate_items}:
            groups_with_positive += 1
        for candidate in candidate_items:
            row = {
                "group_id": group_id,
                "session_id": state.session_id,
                "user_id": state.user_id,
                "turn_number": state.turn_number,
                "track_id": candidate.track_id,
                "label": 1 if candidate.track_id == state.gold_track_id else 0,
            }
            row.update(
                candidate_features(
                    state=state,
                    catalog=catalog,
                    track_id=candidate.track_id,
                    rrf_score=candidate.score,
                    source_ranks=dict(candidate.source_ranks),
                )
            )
            rows.append(row)
    df = pd.DataFrame(rows)
    print(
        json.dumps(
            {
                "rows": len(df),
                "groups": len(states),
                "groups_with_positive_candidates": groups_with_positive,
                "max_candidates_per_group": args.max_candidates_per_group,
            },
            indent=2,
        )
    )
    return df, legacy_by_group


def train_ltr(
    df: pd.DataFrame,
    train_groups,
    args: argparse.Namespace,
) -> tuple[lgb.LGBMRanker, list[str], list[str], dict[str, list[str]], dict[str, int]]:
    df, feature_cols, categorical, category_values = prepare_features(df)
    train_groups = set(train_groups)
    train_df = df[df["group_id"].isin(train_groups)].copy()
    positive_groups = set(train_df.groupby("group_id")["label"].sum().loc[lambda s: s > 0].index)
    train_df = train_df[train_df["group_id"].isin(positive_groups)].copy()
    train_df = train_df.sort_values("group_id")
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
    return ranker, feature_cols, categorical, category_values, stats


def mean_group_ndcg(df: pd.DataFrame, groups, state_by_group: dict[int, object], legacy_by_group: dict[int, list[str]], preserve_head_k: int) -> float:
    grouped = {group_id: group for group_id, group in df[df["group_id"].isin(groups)].groupby("group_id")}
    scores = []
    for group_id in groups:
        group = grouped.get(group_id)
        model_order = list(group.sort_values("model_score", ascending=False)["track_id"]) if group is not None else []
        ranked = merge_with_legacy(
            legacy_order=legacy_by_group.get(group_id, []),
            model_order=model_order,
            preserve_head_k=preserve_head_k,
        )
        scores.append(dcg_at_20(state_by_group[group_id].gold_track_id, ranked))
    return float(np.mean(scores)) if scores else 0.0


def write_predictions(config: GoalFlowConfig, states, df: pd.DataFrame, legacy_by_group, catalog: TrackCatalog, preserve_head_k: int) -> Path:
    grouped = {group_id: group for group_id, group in df.groupby("group_id")}
    predictions = []
    for group_id, state in enumerate(states):
        group = grouped.get(group_id)
        model_order = list(group.sort_values("model_score", ascending=False)["track_id"]) if group is not None else []
        track_ids = merge_with_legacy(
            legacy_order=legacy_by_group.get(group_id, []),
            model_order=model_order,
            preserve_head_k=preserve_head_k,
        )
        predictions.append(
            {
                "session_id": state.session_id,
                "user_id": state.user_id,
                "turn_number": state.turn_number,
                "predicted_track_ids": track_ids,
                "predicted_response": generate_response(state, catalog, track_ids, style=config.response_style),
            }
        )
    validation = validate_predictions(predictions, catalog, expected_count=len(states))
    if not validation["ok"]:
        raise ValueError(f"Invalid predictions: {validation}")
    out_dir = config.experiments_dir / "devset"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config.tid}.json"
    out_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    config = GoalFlowConfig(
        project_root=Path(args.project_root),
        tid=args.tid,
        response_style=args.response_style,
    )
    catalog = TrackCatalog(config.track_metadata_name)
    states = load_dev_states(config)
    state_by_group = {group_id: state for group_id, state in enumerate(states)}
    cache_key = candidate_cache_key(args)
    cached = None if args.rebuild_cache else load_cache(config.project_root, cache_key, states)
    if cached is None:
        df, legacy_by_group = build_candidate_frame(args, states, catalog)
        if not args.skip_cache_write:
            write_cache(config.project_root, cache_key, states, df, legacy_by_group)
    else:
        df, legacy_by_group = cached

    all_groups = sorted(int(group_id) for group_id in df["group_id"].unique())
    valid_groups_by_fold = [
        {group_id for group_id in all_groups if group_id % args.valid_mod == fold}
        for fold in range(args.valid_mod)
    ]

    if args.mode == "validate":
        valid_groups = valid_groups_by_fold[0]
        train_groups = [group_id for group_id in all_groups if group_id not in valid_groups]
        ranker, feature_cols, _categorical, category_values, train_stats = train_ltr(df, train_groups, args)
        valid_df, _, _, _ = prepare_features(
            df[df["group_id"].isin(valid_groups)].copy(),
            feature_cols=feature_cols,
            category_values=category_values,
        )
        valid_df = valid_df.copy()
        valid_df["model_score"] = ranker.predict(valid_df[feature_cols])
        preserve_values = [int(value) for value in args.preserve_grid.split(",") if value.strip()]
        grid = {
            f"head{head}": {
                "ndcg@20": mean_group_ndcg(valid_df, valid_groups, state_by_group, legacy_by_group, head),
                "valid_groups": len(valid_groups),
            }
            for head in preserve_values
        }
        summary = {
            "mode": "validate",
            "tid": args.tid,
            "choice": args.choice,
            "rows": len(df),
            "groups": len(all_groups),
            "max_candidates_per_group": args.max_candidates_per_group,
            "valid_fold": 0,
            "train": train_stats,
            "grid": grid,
        }
        out_dir = config.experiments_dir / "ltr"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "source_choice_ltr_validate_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"summary={out_path}")
        return

    scored_frames = []
    fold_summaries = []
    for fold, fold_valid_groups in enumerate(valid_groups_by_fold):
        fold_train_groups = [group_id for group_id in all_groups if group_id not in fold_valid_groups]
        ranker, feature_cols, _categorical, category_values, train_stats = train_ltr(df, fold_train_groups, args)
        fold_df, _, _, _ = prepare_features(
            df[df["group_id"].isin(fold_valid_groups)].copy(),
            feature_cols=feature_cols,
            category_values=category_values,
        )
        fold_df = fold_df.copy()
        fold_df["model_score"] = ranker.predict(fold_df[feature_cols])
        fold_ndcg = mean_group_ndcg(
            fold_df,
            fold_valid_groups,
            state_by_group,
            legacy_by_group,
            args.preserve_head_k,
        )
        fold_summaries.append({"fold": fold, "valid_groups": len(fold_valid_groups), "ndcg@20": fold_ndcg, "train": train_stats})
        scored_frames.append(fold_df)

    apply_df = pd.concat(scored_frames, ignore_index=False).sort_index()
    output = write_predictions(config, states, apply_df, legacy_by_group, catalog, args.preserve_head_k)
    if args.copy_to_official_evaluator:
        official = config.project_root / "music-crs-evaluator" / "exp" / "inference" / "devset"
        official.mkdir(parents=True, exist_ok=True)
        (official / f"{config.tid}.json").write_text(output.read_text(encoding="utf-8"), encoding="utf-8")
    overall = mean_group_ndcg(apply_df, all_groups, state_by_group, legacy_by_group, args.preserve_head_k)
    summary = {
        "mode": "oof-dev",
        "tid": args.tid,
        "choice": args.choice,
        "rows": len(df),
        "groups": len(all_groups),
        "max_candidates_per_group": args.max_candidates_per_group,
        "preserve_head_k": args.preserve_head_k,
        "oof_ndcg@20": overall,
        "output": str(output),
        "folds": fold_summaries,
    }
    out_dir = config.experiments_dir / "ltr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "source_choice_ltr_oof_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary={out_path}")


if __name__ == "__main__":
    main()
