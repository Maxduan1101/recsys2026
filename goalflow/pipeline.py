from __future__ import annotations

import json
import os
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from .bm25_retrieval import MultiBM25Retriever
from .data import BLIND_A_DATASET, CONVERSATION_DATASET, TRACK_METADATA, TrackCatalog, load_conversations
from .documents import build_documents, build_train_augmentation
from .fusion import diversify_tail, rerank_candidates, rerank_candidates_gated, rrf_fuse
from .response import generate_response
from .state import build_state_for_blind_item, build_state_for_dev_turn, query_variants
from .validation import validate_predictions


@dataclass
class GoalFlowConfig:
    project_root: Path
    tid: str = "goalflow_bm25_aug_v1"
    track_metadata_name: str = TRACK_METADATA
    conversation_dataset_name: str = CONVERSATION_DATASET
    blind_dataset_name: str = BLIND_A_DATASET
    use_train_augmentation: bool = True
    rebuild_cache: bool = False
    retrieval_top_k: int = 260
    rerank_pool_size: int = 1200
    legacy_head_k: int = 20
    fusion_mode: str = "standard"
    tail_diversity_start: int = 20
    global_repeat_penalty: float = 0.0
    rrf_k: int = 60
    response_style: str = "compact"
    dev_limit: int | None = None

    @property
    def cache_dir(self) -> Path:
        return self.project_root / "cache"

    @property
    def experiments_dir(self) -> Path:
        return self.project_root / "experiments" / self.tid


def default_query_weights() -> dict[str, float]:
    return {
        "legacy_history": 3.5,
        "current": 1.35,
        "goal": 1.15,
        "current_goal": 1.45,
        "seed_current": 0.85,
        "quoted_entities": 2.0,
    }


def default_index_weights() -> dict[str, float]:
    return {
        "legacy_metadata": 2.4,
        "metadata_all": 1.0,
        "title_artist": 1.25,
        "album_artist": 0.75,
        "tags": 0.7,
        "enriched": 1.45,
    }


def top_k_by_index(config: GoalFlowConfig) -> dict[str, int]:
    return {
        "legacy_metadata": config.retrieval_top_k,
        "metadata_all": config.retrieval_top_k,
        "title_artist": config.retrieval_top_k,
        "album_artist": max(100, config.retrieval_top_k // 2),
        "tags": max(120, config.retrieval_top_k // 2),
        "enriched": config.retrieval_top_k,
    }


def prepare_retriever(config: GoalFlowConfig, catalog: TrackCatalog) -> MultiBM25Retriever:
    train_aug = {}
    if config.use_train_augmentation:
        train_dataset = load_conversations(config.conversation_dataset_name, split="train")
        aug_path = config.cache_dir / "train_track_augmentation_v1.json"
        train_aug = build_train_augmentation(
            train_dataset=train_dataset,
            catalog=catalog,
            cache_path=str(aug_path),
            rebuild=config.rebuild_cache,
        )
    docs = build_documents(catalog, train_augmentation=train_aug)
    retriever = MultiBM25Retriever(cache_dir=str(config.cache_dir), rebuild=config.rebuild_cache)
    retriever.build(track_ids=docs.track_ids, documents_by_index=docs.documents_by_index)
    return retriever


def _predict_states(config: GoalFlowConfig, states, catalog: TrackCatalog, retriever: MultiBM25Retriever) -> list[dict]:
    if config.fusion_mode not in {"standard", "gated"}:
        raise ValueError(f"Unsupported fusion_mode={config.fusion_mode!r}")

    variants = [query_variants(state, catalog) for state in states]
    source_rows = retriever.batch_search(
        query_variants_per_state=variants,
        top_k_by_index=top_k_by_index(config),
        query_weights=default_query_weights(),
        index_weights=default_index_weights(),
    )

    predictions = []
    global_counts: Counter[str] = Counter()
    for state, sources in tqdm(list(zip(states, source_rows)), desc="Fuse/rerank states"):
        legacy_order: list[str] = []
        for source_name, _index_name, _weight, results in sources:
            if source_name == "legacy_metadata:legacy_history":
                legacy_order = [result.track_id for result in results]
                break
        fused = rrf_fuse(sources, rrf_k=config.rrf_k)
        if len(fused) > config.rerank_pool_size:
            fused = dict(
                sorted(fused.items(), key=lambda item: item[1].score, reverse=True)[: config.rerank_pool_size]
            )
        rerank_top_k = 80 if config.tail_diversity_start < 20 else 20
        if config.fusion_mode == "gated":
            track_ids = rerank_candidates_gated(
                state,
                catalog,
                fused,
                legacy_order=legacy_order,
                top_k=rerank_top_k,
                global_counts=global_counts,
                protect_head_k=config.legacy_head_k,
            )
        else:
            track_ids = rerank_candidates(state, catalog, fused, top_k=rerank_top_k, global_counts=global_counts)
        if config.fusion_mode == "standard" and config.legacy_head_k and legacy_order:
            anchored = legacy_order[: config.legacy_head_k]
            merged = anchored + [track_id for track_id in track_ids if track_id not in anchored]
            merged += [track_id for track_id in legacy_order if track_id not in merged]
            track_ids = merged
        if config.tail_diversity_start < 20:
            track_ids = diversify_tail(
                track_ids,
                state,
                catalog,
                global_counts=global_counts,
                top_k=20,
                preserve_head_k=config.tail_diversity_start,
                repeat_penalty=config.global_repeat_penalty,
            )
        else:
            track_ids = track_ids[:20]
        global_counts.update(track_ids)
        predictions.append(
            {
                "session_id": state.session_id,
                "user_id": state.user_id,
                "turn_number": state.turn_number,
                "predicted_track_ids": track_ids,
                "predicted_response": generate_response(state, catalog, track_ids, style=config.response_style),
            }
        )
    return predictions


def run_dev(config: GoalFlowConfig, copy_to_official_evaluator: bool = True) -> Path:
    catalog = TrackCatalog(config.track_metadata_name)
    retriever = prepare_retriever(config, catalog)
    dataset = load_dataset(config.conversation_dataset_name, split="test")
    if config.dev_limit:
        dataset = dataset.select(range(min(config.dev_limit, len(dataset))))

    states = []
    for item in tqdm(dataset, desc="Build dev states"):
        for turn_number in range(1, 9):
            states.append(build_state_for_dev_turn(item, turn_number))

    predictions = _predict_states(config, states, catalog, retriever)
    validation = validate_predictions(predictions, catalog, expected_count=len(states))
    if not validation["ok"]:
        raise ValueError(f"Invalid predictions: {validation}")

    out_dir = config.experiments_dir / "devset"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config.tid}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    if copy_to_official_evaluator and not config.dev_limit:
        official = config.project_root.parent / "music-crs-evaluator" / "exp" / "inference" / "devset"
        official.mkdir(parents=True, exist_ok=True)
        with open(official / f"{config.tid}.json", "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False)
    return out_path


def run_blind(config: GoalFlowConfig, zip_submission: bool = True) -> Path:
    catalog = TrackCatalog(config.track_metadata_name)
    retriever = prepare_retriever(config, catalog)
    dataset = load_dataset(config.blind_dataset_name, split="test")
    states = [build_state_for_blind_item(item) for item in tqdm(dataset, desc="Build blind states")]
    predictions = _predict_states(config, states, catalog, retriever)
    validation = validate_predictions(predictions, catalog, expected_count=len(states))
    if not validation["ok"]:
        raise ValueError(f"Invalid predictions: {validation}")

    out_dir = config.experiments_dir / "blindset_A"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "prediction.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    if zip_submission:
        zip_path = out_dir / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(pred_path, arcname="prediction.json")
        return zip_path
    return pred_path


def write_run_summary(config: GoalFlowConfig, output_path: Path, mode: str) -> Path:
    summary = {
        "tid": config.tid,
        "mode": mode,
        "output_path": str(output_path),
        "use_train_augmentation": config.use_train_augmentation,
        "retrieval_top_k": config.retrieval_top_k,
        "rerank_pool_size": config.rerank_pool_size,
        "legacy_head_k": config.legacy_head_k,
        "fusion_mode": config.fusion_mode,
        "tail_diversity_start": config.tail_diversity_start,
        "global_repeat_penalty": config.global_repeat_penalty,
        "rrf_k": config.rrf_k,
        "response_style": config.response_style,
        "query_weights": default_query_weights(),
        "index_weights": default_index_weights(),
    }
    config.experiments_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.experiments_dir / f"{mode}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path
