from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from export_ltr_dataset import candidate_features
from goalflow.data import BLIND_A_DATASET, CONVERSATION_DATASET, TrackCatalog
from goalflow.embeddings import TRACK_EMBEDDING_CHANNELS, TrackEmbeddingStore, UserEmbeddingStore
from goalflow.fusion import CandidateScore, rrf_fuse
from goalflow.pipeline import (
    GoalFlowConfig,
    default_index_weights,
    default_query_weights,
    prepare_retriever,
    top_k_by_index,
    write_run_summary,
)
from goalflow.response import generate_response
from goalflow.state import build_state_for_blind_item, build_state_for_dev_turn, query_variants
from goalflow.validation import validate_predictions


LTR_CATEGORICAL = ["intent", "category", "specificity"]
META_COLUMNS = {"session_id", "user_id", "track_id", "label"}
CANDIDATE_CACHE_VERSION = "ltr_candidate_frames_v1"
DEFAULT_EMBEDDING_FEATURE_CHANNELS = "metadata,attributes,track_cf"


def dcg_at_20(gold_track_id: str | None, ranked_track_ids: list[str]) -> float:
    if not gold_track_id:
        return 0.0
    for rank, track_id in enumerate(ranked_track_ids[:20], start=1):
        if track_id == gold_track_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def load_dev_states(config: GoalFlowConfig):
    dataset = load_dataset(config.conversation_dataset_name, split="test")
    if config.dev_limit:
        dataset = dataset.select(range(min(config.dev_limit, len(dataset))))
    states = []
    for item in tqdm(dataset, desc="Build dev states"):
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id:
                states.append(state)
    return states


def load_extra_train_states(config: GoalFlowConfig, session_limit: int, seed: int):
    dataset = load_dataset(config.conversation_dataset_name, split="train")
    if session_limit and session_limit < len(dataset):
        rng = np.random.default_rng(seed)
        indices = sorted(int(index) for index in rng.choice(len(dataset), size=session_limit, replace=False))
        dataset = dataset.select(indices)
    states = []
    for item in tqdm(dataset, desc="Build extra train states"):
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id:
                states.append(state)
    return states


def load_blind_states(config: GoalFlowConfig):
    dataset = load_dataset(config.blind_dataset_name, split="test")
    return [build_state_for_blind_item(item) for item in tqdm(dataset, desc="Build blind states")]


def legacy_order_from_sources(sources) -> list[str]:
    for source_name, _index_name, _weight, results in sources:
        if source_name == "legacy_metadata:legacy_history":
            return [result.track_id for result in results]
    return []


def build_candidate_frames(
    *,
    states,
    config: GoalFlowConfig,
    catalog: TrackCatalog,
    retriever,
    max_candidates_per_group: int,
) -> tuple[pd.DataFrame, dict[int, list[str]], dict[int, object]]:
    variants = [query_variants(state, catalog) for state in states]
    source_rows = retriever.batch_search(
        query_variants_per_state=variants,
        top_k_by_index=top_k_by_index(config),
        query_weights=default_query_weights(),
        index_weights=default_index_weights(),
    )

    rows = []
    legacy_by_group: dict[int, list[str]] = {}
    state_by_group: dict[int, object] = {}
    for group_id, (state, sources) in enumerate(
        tqdm(list(zip(states, source_rows)), desc="Build LTR candidates")
    ):
        legacy_by_group[group_id] = legacy_order_from_sources(sources)
        state_by_group[group_id] = state
        fused = rrf_fuse(sources, rrf_k=config.rrf_k)
        if len(fused) > config.rerank_pool_size:
            fused = dict(
                sorted(fused.items(), key=lambda item: item[1].score, reverse=True)[
                    : config.rerank_pool_size
                ]
            )
        candidates = sorted(fused.values(), key=lambda item: item.score, reverse=True)[
            :max_candidates_per_group
        ]
        for candidate in candidates:
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
    return pd.DataFrame(rows), legacy_by_group, state_by_group


def candidate_cache_key(
    *,
    split: str,
    config: GoalFlowConfig,
    max_candidates_per_group: int,
) -> str:
    payload = {
        "version": CANDIDATE_CACHE_VERSION,
        "split": split,
        "conversation_dataset_name": config.conversation_dataset_name,
        "blind_dataset_name": config.blind_dataset_name,
        "use_train_augmentation": config.use_train_augmentation,
        "retrieval_top_k": config.retrieval_top_k,
        "rerank_pool_size": config.rerank_pool_size,
        "rrf_k": config.rrf_k,
        "dev_limit": config.dev_limit,
        "max_candidates_per_group": max_candidates_per_group,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{split}_{digest}"


def candidate_cache_paths(config: GoalFlowConfig, cache_key: str) -> tuple[Path, Path]:
    cache_dir = config.cache_dir / "ltr_candidate_frames"
    return cache_dir / f"{cache_key}.pkl", cache_dir / f"{cache_key}.json"


def load_candidate_frame_cache(
    *,
    states,
    config: GoalFlowConfig,
    cache_key: str,
) -> tuple[pd.DataFrame, dict[int, list[str]], dict[int, object]] | None:
    df_path, meta_path = candidate_cache_paths(config, cache_key)
    if not df_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("state_count") != len(states):
        return None
    df = pd.read_pickle(df_path)
    legacy_by_group = {
        int(group_id): list(track_ids)
        for group_id, track_ids in meta.get("legacy_by_group", {}).items()
    }
    state_by_group = {group_id: state for group_id, state in enumerate(states)}
    print(f"Loaded LTR candidate cache: {df_path}")
    return df, legacy_by_group, state_by_group


def write_candidate_frame_cache(
    *,
    states,
    config: GoalFlowConfig,
    cache_key: str,
    df: pd.DataFrame,
    legacy_by_group: dict[int, list[str]],
) -> None:
    df_path, meta_path = candidate_cache_paths(config, cache_key)
    df_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(df_path)
    meta = {
        "version": CANDIDATE_CACHE_VERSION,
        "state_count": len(states),
        "row_count": len(df),
        "legacy_by_group": {str(group_id): track_ids for group_id, track_ids in legacy_by_group.items()},
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    print(f"Wrote LTR candidate cache: {df_path}")


def build_or_load_candidate_frames(
    *,
    split: str,
    states,
    config: GoalFlowConfig,
    catalog: TrackCatalog,
    retriever,
    max_candidates_per_group: int,
) -> tuple[pd.DataFrame, dict[int, list[str]], dict[int, object]]:
    cache_key = candidate_cache_key(
        split=split,
        config=config,
        max_candidates_per_group=max_candidates_per_group,
    )
    if not config.rebuild_cache:
        cached = load_candidate_frame_cache(states=states, config=config, cache_key=cache_key)
        if cached is not None:
            return cached
    df, legacy_by_group, state_by_group = build_candidate_frames(
        states=states,
        config=config,
        catalog=catalog,
        retriever=retriever,
        max_candidates_per_group=max_candidates_per_group,
    )
    write_candidate_frame_cache(
        states=states,
        config=config,
        cache_key=cache_key,
        df=df,
        legacy_by_group=legacy_by_group,
    )
    return df, legacy_by_group, state_by_group


def parse_embedding_channels(value: str) -> list[str]:
    channels = []
    for raw in value.split(","):
        channel = raw.strip()
        if not channel:
            continue
        if channel not in TRACK_EMBEDDING_CHANNELS:
            allowed = ", ".join(sorted(TRACK_EMBEDDING_CHANNELS))
            raise ValueError(f"Unknown embedding channel {channel!r}; allowed: {allowed}")
        if channel not in channels:
            channels.append(channel)
    return channels


def _valid_embedding_indices(
    track_store: TrackEmbeddingStore,
    channel: str,
    track_ids: Iterable[str],
) -> list[int]:
    matrix = track_store.matrices[channel]
    indices = []
    for track_id in track_ids:
        index = track_store.track_index.get(track_id)
        if index is not None and matrix.valid[index]:
            indices.append(index)
    return indices


def _rank_percentiles(scores: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rank = np.zeros_like(scores, dtype=np.float32)
    pct = np.zeros_like(scores, dtype=np.float32)
    valid_positions = np.flatnonzero(valid & np.isfinite(scores))
    if not len(valid_positions):
        return rank, pct
    ordered_positions = valid_positions[np.argsort(-scores[valid_positions])]
    rank[ordered_positions] = np.arange(1, len(ordered_positions) + 1, dtype=np.float32)
    if len(ordered_positions) == 1:
        pct[ordered_positions] = 1.0
    else:
        pct[ordered_positions] = 1.0 - (rank[ordered_positions] - 1.0) / (len(ordered_positions) - 1.0)
    return rank, pct


def _zscore(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    valid_scores = scores[valid & np.isfinite(scores)]
    if len(valid_scores) < 2:
        return out
    std = float(valid_scores.std())
    if not np.isfinite(std) or std <= 1e-12:
        return out
    mean = float(valid_scores.mean())
    out[valid] = (scores[valid] - mean) / std
    return out


def add_embedding_features(
    df: pd.DataFrame,
    *,
    state_by_group: dict[int, object],
    track_store: TrackEmbeddingStore,
    channels: list[str],
    user_store: UserEmbeddingStore | None,
) -> pd.DataFrame:
    """Append optional seed/user embedding features without rebuilding BM25 candidates."""
    if df.empty:
        return df
    out = df.copy()
    candidate_indices = np.array(
        [track_store.track_index.get(track_id, -1) for track_id in out["track_id"]],
        dtype=np.int32,
    )
    out["_embedding_track_index"] = candidate_indices

    for channel in channels:
        prefix = f"emb_{channel}"
        out[f"{prefix}_candidate_valid"] = 0.0
        out[f"{prefix}_pos_max"] = 0.0
        out[f"{prefix}_neg_max"] = 0.0
        out[f"{prefix}_pos_minus_neg"] = 0.0
        out[f"{prefix}_pos_seed_count"] = 0.0
        out[f"{prefix}_neg_seed_count"] = 0.0
        out[f"{prefix}_pos_rank_pct"] = 0.0

    if user_store is not None:
        out["emb_user_cf_has_user"] = 0.0
        out["emb_user_cf_candidate_valid"] = 0.0
        out["emb_user_cf_raw"] = 0.0
        out["emb_user_cf_z"] = 0.0
        out["emb_user_cf_rank"] = 0.0
        out["emb_user_cf_rank_pct"] = 0.0

    grouped = out.groupby("group_id", sort=False)
    for group_id, group in tqdm(grouped, desc="Add embedding LTR features"):
        state = state_by_group[int(group_id)]
        row_index = group.index
        group_positions = out.index.get_indexer(row_index)
        group_embedding_indices = candidate_indices[group_positions]

        for channel in channels:
            matrix = track_store.matrices[channel]
            prefix = f"emb_{channel}"
            valid_candidates = (
                (group_embedding_indices >= 0)
                & matrix.valid[np.clip(group_embedding_indices, 0, len(matrix.valid) - 1)]
            )
            out.loc[row_index, f"{prefix}_candidate_valid"] = valid_candidates.astype(np.float32)

            pos_indices = _valid_embedding_indices(track_store, channel, state.positive_seed_ids[-4:])
            neg_indices = _valid_embedding_indices(track_store, channel, state.negative_seed_ids[-4:])
            out.loc[row_index, f"{prefix}_pos_seed_count"] = float(len(pos_indices))
            out.loc[row_index, f"{prefix}_neg_seed_count"] = float(len(neg_indices))

            pos_scores = np.zeros(len(group), dtype=np.float32)
            neg_scores = np.zeros(len(group), dtype=np.float32)
            if valid_candidates.any() and pos_indices:
                candidate_vectors = matrix.normalized[group_embedding_indices[valid_candidates]]
                seed_vectors = matrix.normalized[pos_indices]
                pos_scores[valid_candidates] = (candidate_vectors @ seed_vectors.T).max(axis=1)
                out.loc[row_index, f"{prefix}_pos_max"] = pos_scores
                _rank, pct = _rank_percentiles(pos_scores, valid_candidates)
                out.loc[row_index, f"{prefix}_pos_rank_pct"] = pct
            if valid_candidates.any() and neg_indices:
                candidate_vectors = matrix.normalized[group_embedding_indices[valid_candidates]]
                seed_vectors = matrix.normalized[neg_indices]
                neg_scores[valid_candidates] = (candidate_vectors @ seed_vectors.T).max(axis=1)
                out.loc[row_index, f"{prefix}_neg_max"] = neg_scores
            out.loc[row_index, f"{prefix}_pos_minus_neg"] = pos_scores - neg_scores

        if user_store is not None and "track_cf" in track_store.matrices:
            user_vector = user_store.user_vectors.get(state.user_id)
            matrix = track_store.matrices["track_cf"]
            valid_candidates = (
                (group_embedding_indices >= 0)
                & matrix.valid[np.clip(group_embedding_indices, 0, len(matrix.valid) - 1)]
            )
            out.loc[row_index, "emb_user_cf_candidate_valid"] = valid_candidates.astype(np.float32)
            if user_vector is not None and matrix.raw.shape[1] == user_vector.shape[0]:
                scores = np.zeros(len(group), dtype=np.float32)
                if valid_candidates.any():
                    scores[valid_candidates] = matrix.raw[group_embedding_indices[valid_candidates]] @ user_vector
                rank, pct = _rank_percentiles(scores, valid_candidates)
                out.loc[row_index, "emb_user_cf_has_user"] = 1.0
                out.loc[row_index, "emb_user_cf_raw"] = scores
                out.loc[row_index, "emb_user_cf_z"] = _zscore(scores, valid_candidates)
                out.loc[row_index, "emb_user_cf_rank"] = rank
                out.loc[row_index, "emb_user_cf_rank_pct"] = pct

    return out.drop(columns=["_embedding_track_index"])


def prepare_features(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    category_values: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, list[str]]]:
    df = df.copy()
    if "source_count" not in df:
        df["source_count"] = 0.0

    categorical = [column for column in LTR_CATEGORICAL if column in df.columns]
    if category_values is None:
        category_values = {}
        for column in categorical:
            values = sorted(str(value) for value in df[column].fillna("missing").unique())
            category_values[column] = values or ["missing"]

    if feature_cols is None:
        ignored = set(META_COLUMNS) | {"group_id"}
        feature_cols = [column for column in df.columns if column not in ignored]
    else:
        for column in feature_cols:
            if column not in df:
                df[column] = np.nan

    for column in categorical:
        values = category_values.get(column, ["missing"])
        df[column] = pd.Categorical(df[column].fillna("missing").astype(str), categories=values)

    for column in feature_cols:
        if column in categorical:
            continue
        if column.startswith("rank_") or column == "best_source_rank":
            df[column] = df[column].fillna(9999.0)
        else:
            df[column] = df[column].fillna(0.0)
    return df, feature_cols, categorical, category_values


def train_ltr(
    df: pd.DataFrame,
    train_groups: Iterable[int],
    *,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    subsample: float,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
    lambdarank_truncation_level: int,
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
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        lambdarank_truncation_level=lambdarank_truncation_level,
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


def ltr_order_for_group(group: pd.DataFrame) -> list[str]:
    return list(group.sort_values("model_score", ascending=False)["track_id"])


def merge_with_legacy(
    *,
    legacy_order: list[str],
    model_order: list[str],
    preserve_head_k: int,
    top_k: int = 20,
) -> list[str]:
    selected: list[str] = []
    seen = set()
    for track_id in legacy_order[: max(0, preserve_head_k)]:
        if track_id not in seen:
            selected.append(track_id)
            seen.add(track_id)
    for track_id in model_order:
        if len(selected) >= top_k:
            break
        if track_id not in seen:
            selected.append(track_id)
            seen.add(track_id)
    for track_id in legacy_order:
        if len(selected) >= top_k:
            break
        if track_id not in seen:
            selected.append(track_id)
            seen.add(track_id)
    return selected[:top_k]


def evaluate_preserve_grid(
    *,
    df: pd.DataFrame,
    groups: Iterable[int],
    legacy_by_group: dict[int, list[str]],
    state_by_group: dict[int, object],
    preserve_values: list[int],
) -> dict[str, dict[str, float]]:
    groups = list(groups)
    results: dict[str, dict[str, float]] = {}
    grouped = {group_id: group for group_id, group in df[df["group_id"].isin(groups)].groupby("group_id")}
    for preserve_head_k in preserve_values:
        scores = []
        changed = 0
        for group_id in groups:
            state = state_by_group[group_id]
            legacy_order = legacy_by_group.get(group_id, [])
            model_order = ltr_order_for_group(grouped[group_id]) if group_id in grouped else []
            ranked = merge_with_legacy(
                legacy_order=legacy_order,
                model_order=model_order,
                preserve_head_k=preserve_head_k,
            )
            scores.append(dcg_at_20(state.gold_track_id, ranked))
            if ranked != legacy_order[:20]:
                changed += 1
        key = f"head{preserve_head_k}"
        results[key] = {
            "ndcg@20": float(np.mean(scores)) if scores else 0.0,
            "changed_groups": changed,
        }
    baseline = [
        dcg_at_20(state_by_group[group_id].gold_track_id, legacy_by_group.get(group_id, [])[:20])
        for group_id in groups
    ]
    results["legacy_head20"] = {
        "ndcg@20": float(np.mean(baseline)) if baseline else 0.0,
        "changed_groups": 0,
    }
    return results


def write_predictions(
    *,
    config: GoalFlowConfig,
    mode: str,
    states,
    df: pd.DataFrame,
    legacy_by_group: dict[int, list[str]],
    catalog: TrackCatalog,
    preserve_head_k: int,
    zip_submission: bool,
) -> Path:
    grouped = {group_id: group for group_id, group in df.groupby("group_id")}
    predictions = []
    for group_id, state in enumerate(states):
        model_order = ltr_order_for_group(grouped[group_id]) if group_id in grouped else []
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

    if mode == "blind":
        out_dir = config.experiments_dir / "blindset_A"
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = out_dir / "prediction.json"
        pred_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
        if zip_submission:
            zip_path = out_dir / "submission.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(pred_path, arcname="prediction.json")
            return zip_path
        return pred_path

    out_dir = config.experiments_dir / "devset"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config.tid}.json"
    out_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    return out_path


def copy_dev_prediction_to_official(config: GoalFlowConfig, output: Path) -> None:
    if config.dev_limit:
        return
    official = config.project_root.parent / "music-crs-evaluator" / "exp" / "inference" / "devset"
    official.mkdir(parents=True, exist_ok=True)
    official_path = official / f"{config.tid}.json"
    official_path.write_text(output.read_text(encoding="utf-8"), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and apply a protected LightGBM LTR reranker.")
    parser.add_argument("--mode", choices=["validate", "oof-dev", "dev", "blind"], required=True)
    parser.add_argument("--tid", default="goalflow_ltr_probe")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--conversation-dataset-name", default=CONVERSATION_DATASET)
    parser.add_argument("--blind-dataset-name", default=BLIND_A_DATASET)
    parser.add_argument("--retrieval-top-k", type=int, default=260)
    parser.add_argument("--rerank-pool-size", type=int, default=1200)
    parser.add_argument("--max-candidates-per-group", type=int, default=300)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--valid-mod", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=260)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=0.0)
    parser.add_argument("--lambdarank-truncation-level", type=int, default=30)
    parser.add_argument(
        "--estimator-grid",
        default="",
        help="Validate-mode comma-separated n_estimators values trained on the same candidate frame.",
    )
    parser.add_argument(
        "--learning-rate-grid",
        default="",
        help="Validate-mode comma-separated learning_rate values trained on the same candidate frame.",
    )
    parser.add_argument(
        "--num-leaves-grid",
        default="",
        help="Validate-mode comma-separated num_leaves values trained on the same candidate frame.",
    )
    parser.add_argument(
        "--min-child-samples-grid",
        default="",
        help="Validate-mode comma-separated min_child_samples values trained on the same candidate frame.",
    )
    parser.add_argument(
        "--subsample-grid",
        default="",
        help="Validate-mode comma-separated subsample values trained on the same candidate frame.",
    )
    parser.add_argument(
        "--colsample-bytree-grid",
        default="",
        help="Validate-mode comma-separated colsample_bytree values trained on the same candidate frame.",
    )
    parser.add_argument(
        "--reg-alpha-grid",
        default="",
        help="Validate-mode comma-separated L1 regularization values trained on the same candidate frame.",
    )
    parser.add_argument(
        "--reg-lambda-grid",
        default="",
        help="Validate-mode comma-separated L2 regularization values trained on the same candidate frame.",
    )
    parser.add_argument("--preserve-head-k", type=int, default=18)
    parser.add_argument(
        "--preserve-grid",
        default="0,1,3,5,10,15,18,19,20",
        help="Comma-separated head sizes for validate mode.",
    )
    parser.add_argument(
        "--response-style",
        choices=[
            "compact", "compact_broad", "concise", "setwise", "natural", "polished",
            "judge_v1", "judge_v2", "judge_v3", "judge_mix", "judge_brief",
            "judge_planned", "judge_compact_mix", "judge_clean_mix", "judge_balanced_mix",
            "judge_clean_mix_plus", "judge_clean_mix_safeplus", "judge_clean_mix_lexplus",
        ],
        default="compact_broad",
    )
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument(
        "--extra-train-sessions",
        type=int,
        default=0,
        help="Deterministically sample this many labeled train-split sessions as extra LTR training data.",
    )
    parser.add_argument("--extra-train-seed", type=int, default=2026)
    parser.add_argument(
        "--extra-train-max-candidates",
        type=int,
        default=160,
        help="Candidates per extra train-split group. Lower than dev keeps memory bounded.",
    )
    parser.add_argument(
        "--embedding-features",
        action="store_true",
        help="Append seed/user official-embedding features to LTR candidates after loading the BM25 cache.",
    )
    parser.add_argument(
        "--embedding-feature-channels",
        default=DEFAULT_EMBEDDING_FEATURE_CHANNELS,
        help="Comma-separated official track embedding channels for seed similarity features.",
    )
    parser.add_argument(
        "--no-user-cf-feature",
        action="store_true",
        help="Disable user_cf_bpr candidate features when --embedding-features is enabled.",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-train-augmentation", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def parse_int_grid(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_float_grid(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def model_key(
    *,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    subsample: float,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
) -> str:
    return (
        f"n{n_estimators}_lr{learning_rate:g}_"
        f"leaves{num_leaves}_minchild{min_child_samples}_"
        f"sub{subsample:g}_col{colsample_bytree:g}_"
        f"alpha{reg_alpha:g}_lambda{reg_lambda:g}"
    )


def main() -> None:
    args = parse_args()
    config = GoalFlowConfig(
        project_root=Path(args.project_root),
        tid=args.tid,
        conversation_dataset_name=args.conversation_dataset_name,
        blind_dataset_name=args.blind_dataset_name,
        use_train_augmentation=not args.no_train_augmentation,
        rebuild_cache=args.rebuild_cache,
        retrieval_top_k=args.retrieval_top_k,
        rerank_pool_size=args.rerank_pool_size,
        rrf_k=args.rrf_k,
        response_style=args.response_style,
        dev_limit=args.dev_limit,
    )
    catalog = TrackCatalog(config.track_metadata_name)
    retriever = prepare_retriever(config, catalog)

    dev_states = load_dev_states(config)
    dev_df, dev_legacy, dev_state_by_group = build_or_load_candidate_frames(
        split="dev",
        states=dev_states,
        config=config,
        catalog=catalog,
        retriever=retriever,
        max_candidates_per_group=args.max_candidates_per_group,
    )
    embedding_channels: list[str] = []
    track_embedding_store: TrackEmbeddingStore | None = None
    user_embedding_store: UserEmbeddingStore | None = None
    embedding_feature_stats = {
        "enabled": bool(args.embedding_features),
        "channels": [],
        "user_cf": False,
    }
    if args.embedding_features:
        embedding_channels = parse_embedding_channels(args.embedding_feature_channels)
        if not args.no_user_cf_feature and "track_cf" not in embedding_channels:
            embedding_channels.append("track_cf")
        track_embedding_store = TrackEmbeddingStore(
            channels={channel: TRACK_EMBEDDING_CHANNELS[channel] for channel in embedding_channels}
        )
        if not args.no_user_cf_feature:
            user_embedding_store = UserEmbeddingStore()
        embedding_feature_stats = {
            "enabled": True,
            "channels": embedding_channels,
            "user_cf": user_embedding_store is not None,
        }
        dev_df = add_embedding_features(
            dev_df,
            state_by_group=dev_state_by_group,
            track_store=track_embedding_store,
            channels=embedding_channels,
            user_store=user_embedding_store,
        )
    all_groups = sorted(dev_df["group_id"].unique())
    valid_groups = {group_id for group_id in all_groups if group_id % args.valid_mod == 0}
    train_groups = [group_id for group_id in all_groups if group_id not in valid_groups]
    extra_train_df = None
    extra_train_groups: list[int] = []
    extra_train_stats = {
        "sessions": args.extra_train_sessions,
        "states": 0,
        "rows": 0,
        "groups_with_positive_candidates": 0,
        "max_candidates_per_group": args.extra_train_max_candidates,
        "seed": args.extra_train_seed,
    }
    if args.extra_train_sessions > 0:
        extra_states = load_extra_train_states(
            config,
            session_limit=args.extra_train_sessions,
            seed=args.extra_train_seed,
        )
        raw_extra_df, _extra_legacy, _extra_state_by_group = build_or_load_candidate_frames(
            split=f"trainExtra{args.extra_train_sessions}_seed{args.extra_train_seed}",
            states=extra_states,
            config=config,
            catalog=catalog,
            retriever=retriever,
            max_candidates_per_group=args.extra_train_max_candidates,
        )
        if track_embedding_store is not None:
            raw_extra_df = add_embedding_features(
                raw_extra_df,
                state_by_group=_extra_state_by_group,
                track_store=track_embedding_store,
                channels=embedding_channels,
                user_store=user_embedding_store,
            )
        group_offset = int(max(all_groups) + 1) if all_groups else 0
        extra_train_df = raw_extra_df.copy()
        extra_train_df["group_id"] = extra_train_df["group_id"] + group_offset
        extra_train_groups = sorted(int(group_id) for group_id in extra_train_df["group_id"].unique())
        extra_train_stats.update(
            {
                "states": len(extra_states),
                "rows": len(extra_train_df),
                "groups_with_positive_candidates": int(
                    (extra_train_df.groupby("group_id")["label"].sum() > 0).sum()
                ),
            }
        )
    train_pool_df = (
        pd.concat([dev_df, extra_train_df], ignore_index=True)
        if extra_train_df is not None
        else dev_df
    )

    if args.mode == "validate":
        preserve_values = [int(value) for value in args.preserve_grid.split(",") if value.strip()]
        estimator_values = parse_int_grid(args.estimator_grid) or [args.n_estimators]
        learning_rate_values = parse_float_grid(args.learning_rate_grid) or [args.learning_rate]
        num_leaves_values = parse_int_grid(args.num_leaves_grid) or [args.num_leaves]
        min_child_values = parse_int_grid(args.min_child_samples_grid) or [args.min_child_samples]
        subsample_values = parse_float_grid(args.subsample_grid) or [args.subsample]
        colsample_values = parse_float_grid(args.colsample_bytree_grid) or [args.colsample_bytree]
        reg_alpha_values = parse_float_grid(args.reg_alpha_grid) or [args.reg_alpha]
        reg_lambda_values = parse_float_grid(args.reg_lambda_grid) or [args.reg_lambda]
        estimator_results = {}
        first_grid = None
        first_train_stats = None
        first_valid_df = None
        for (
            n_estimators,
            learning_rate,
            num_leaves,
            min_child_samples,
            subsample,
            colsample_bytree,
            reg_alpha,
            reg_lambda,
        ) in product(
            estimator_values,
            learning_rate_values,
            num_leaves_values,
            min_child_values,
            subsample_values,
            colsample_values,
            reg_alpha_values,
            reg_lambda_values,
        ):
            ranker, feature_cols, _categorical, category_values, train_stats = train_ltr(
                train_pool_df,
                list(train_groups) + extra_train_groups,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                lambdarank_truncation_level=args.lambdarank_truncation_level,
            )
            valid_df, _, _, _ = prepare_features(
                dev_df[dev_df["group_id"].isin(valid_groups)].copy(),
                feature_cols=feature_cols,
                category_values=category_values,
            )
            valid_df = valid_df.copy()
            valid_df["model_score"] = ranker.predict(valid_df[feature_cols])
            grid = evaluate_preserve_grid(
                df=valid_df,
                groups=valid_groups,
                legacy_by_group=dev_legacy,
                state_by_group=dev_state_by_group,
                preserve_values=preserve_values,
            )
            key = model_key(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
            )
            estimator_results[key] = {
                "params": {
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "num_leaves": num_leaves,
                    "min_child_samples": min_child_samples,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "reg_alpha": reg_alpha,
                    "reg_lambda": reg_lambda,
                    "lambdarank_truncation_level": args.lambdarank_truncation_level,
                },
                "train": train_stats,
                "grid": grid,
            }
            if first_grid is None:
                first_grid = grid
                first_train_stats = train_stats
                first_valid_df = valid_df
        if first_grid is None or first_train_stats is None or first_valid_df is None:
            raise ValueError("No validate estimator values were provided.")
        best_estimator = max(
            estimator_results,
            key=lambda key: estimator_results[key]["grid"].get("head0", {}).get("ndcg@20", -1.0),
        )
        best_params = estimator_results[best_estimator]["params"]
        summary = {
            "mode": "validate",
            "tid": args.tid,
            "valid_mod": args.valid_mod,
            "max_candidates_per_group": args.max_candidates_per_group,
            "n_estimators": estimator_values[0],
            "learning_rate": learning_rate_values[0],
            "num_leaves": num_leaves_values[0],
            "min_child_samples": min_child_values[0],
            "subsample": subsample_values[0],
            "colsample_bytree": colsample_values[0],
            "reg_alpha": reg_alpha_values[0],
            "reg_lambda": reg_lambda_values[0],
            "lambdarank_truncation_level": args.lambdarank_truncation_level,
            "extra_train": extra_train_stats,
            "embedding_features": embedding_feature_stats,
            "best_ltr_config_by_head0": best_params,
            "train": first_train_stats,
            "valid_groups": len(valid_groups),
            "valid_groups_with_positive_candidates": int(
                (first_valid_df.groupby("group_id")["label"].sum() > 0).sum()
            ),
            "grid": first_grid,
            "estimator_grid": estimator_results,
        }
        out_dir = config.experiments_dir / "ltr"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "ltr_validate_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"summary={out_path}")
        return

    if args.mode == "oof-dev":
        scored_frames = []
        fold_stats = []
        for fold in range(args.valid_mod):
            fold_valid_groups = {group_id for group_id in all_groups if group_id % args.valid_mod == fold}
            fold_train_groups = [group_id for group_id in all_groups if group_id not in fold_valid_groups]
            ranker, feature_cols, _categorical, category_values, train_stats = train_ltr(
                train_pool_df,
                list(fold_train_groups) + extra_train_groups,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                num_leaves=args.num_leaves,
                min_child_samples=args.min_child_samples,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                reg_alpha=args.reg_alpha,
                reg_lambda=args.reg_lambda,
                lambdarank_truncation_level=args.lambdarank_truncation_level,
            )
            fold_df, _, _, _ = prepare_features(
                dev_df[dev_df["group_id"].isin(fold_valid_groups)].copy(),
                feature_cols=feature_cols,
                category_values=category_values,
            )
            fold_df = fold_df.copy()
            fold_df["model_score"] = ranker.predict(fold_df[feature_cols])
            fold_scores = evaluate_preserve_grid(
                df=fold_df,
                groups=fold_valid_groups,
                legacy_by_group=dev_legacy,
                state_by_group=dev_state_by_group,
                preserve_values=[args.preserve_head_k],
            )
            fold_stats.append(
                {
                    "fold": fold,
                    "valid_groups": len(fold_valid_groups),
                    "valid_groups_with_positive_candidates": int(
                        (fold_df.groupby("group_id")["label"].sum() > 0).sum()
                    ),
                    "train": train_stats,
                    "scores": fold_scores,
                }
            )
            scored_frames.append(fold_df)

        apply_df = pd.concat(scored_frames, ignore_index=False).sort_index()
        output = write_predictions(
            config=config,
            mode="dev",
            states=dev_states,
            df=apply_df,
            legacy_by_group=dev_legacy,
            catalog=catalog,
            preserve_head_k=args.preserve_head_k,
            zip_submission=False,
        )
        copy_dev_prediction_to_official(config, output)
        summary = {
            "mode": "oof-dev",
            "tid": args.tid,
            "valid_mod": args.valid_mod,
            "max_candidates_per_group": args.max_candidates_per_group,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_child_samples": args.min_child_samples,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "lambdarank_truncation_level": args.lambdarank_truncation_level,
            "preserve_head_k": args.preserve_head_k,
            "extra_train": extra_train_stats,
            "embedding_features": embedding_feature_stats,
            "output": str(output),
            "folds": fold_stats,
        }
        out_dir = config.experiments_dir / "ltr"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "ltr_oof_summary.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"summary={out_path}")
        return

    train_groups_for_output = all_groups if args.mode == "blind" else train_groups
    ranker, feature_cols, _categorical, category_values, train_stats = train_ltr(
        train_pool_df,
        list(train_groups_for_output) + extra_train_groups,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        lambdarank_truncation_level=args.lambdarank_truncation_level,
    )

    if args.mode == "blind":
        states = load_blind_states(config)
        apply_df, legacy_by_group, _state_by_group = build_or_load_candidate_frames(
            split="blindA",
            states=states,
            config=config,
            catalog=catalog,
            retriever=retriever,
            max_candidates_per_group=args.max_candidates_per_group,
        )
        if track_embedding_store is not None:
            apply_df = add_embedding_features(
                apply_df,
                state_by_group=_state_by_group,
                track_store=track_embedding_store,
                channels=embedding_channels,
                user_store=user_embedding_store,
            )
    else:
        states = dev_states
        apply_df = dev_df
        legacy_by_group = dev_legacy

    apply_df, _, _, _ = prepare_features(
        apply_df,
        feature_cols=feature_cols,
        category_values=category_values,
    )
    apply_df = apply_df.copy()
    apply_df["model_score"] = ranker.predict(apply_df[feature_cols])
    output = write_predictions(
        config=config,
        mode=args.mode,
        states=states,
        df=apply_df,
        legacy_by_group=legacy_by_group,
        catalog=catalog,
        preserve_head_k=args.preserve_head_k,
        zip_submission=not args.no_zip,
    )
    if args.mode == "dev":
        copy_dev_prediction_to_official(config, output)
    summary = write_run_summary(config, output, args.mode)
    print(
        json.dumps(
            {
                "output": str(output),
                "summary": str(summary),
                "train": train_stats,
                "extra_train": extra_train_stats,
                "embedding_features": embedding_feature_stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
