from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from goalflow.data import TrackCatalog
from goalflow.fusion import CandidateScore, infer_intent, rrf_fuse, score_candidate_boost
from goalflow.pipeline import (
    GoalFlowConfig,
    default_index_weights,
    default_query_weights,
    prepare_retriever,
    top_k_by_index,
)
from goalflow.state import ConversationState, build_state_for_dev_turn, query_variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GoalFlow candidate features for a future LightGBM/CatBoost ranker."
    )
    parser.add_argument("--tid", default="ltr_export")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--retrieval-top-k", type=int, default=260)
    parser.add_argument("--rerank-pool-size", type=int, default=1200)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--max-candidates-per-group", type=int, default=300)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-train-augmentation", action="store_true")
    parser.add_argument("--include-missed-gold", action="store_true")
    return parser.parse_args()


def sanitize_feature_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
    return value.strip("_").lower()


def build_dev_states(config: GoalFlowConfig) -> list[ConversationState]:
    dataset = load_dataset(config.conversation_dataset_name, split="test")
    if config.dev_limit:
        dataset = dataset.select(range(min(config.dev_limit, len(dataset))))

    states = []
    for item in tqdm(dataset, desc="Build LTR states"):
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id:
                states.append(state)
    return states


def safe_log1p(value: float | int | None) -> float:
    return math.log1p(max(float(value or 0.0), 0.0))


def rank_features(source_ranks: dict[str, int]) -> dict[str, float]:
    features: dict[str, float] = {
        "source_count": float(len(source_ranks)),
        "best_source_rank": float(min(source_ranks.values())) if source_ranks else 9999.0,
    }
    for source_name, rank in source_ranks.items():
        key = sanitize_feature_name(f"rank_{source_name}")
        features[key] = float(rank)
        features[f"hit20_{key}"] = 1.0 if rank <= 20 else 0.0
        features[f"hit100_{key}"] = 1.0 if rank <= 100 else 0.0
    return features


def candidate_features(
    state: ConversationState,
    catalog: TrackCatalog,
    track_id: str,
    rrf_score: float,
    source_ranks: dict[str, int],
) -> dict[str, float | str | int]:
    view = catalog.view(track_id)
    artist = catalog.normalized_field(track_id, "artist_name")
    album = catalog.normalized_field(track_id, "album_name")
    positive_artists = {
        catalog.normalized_field(seed_id, "artist_name")
        for seed_id in state.positive_seed_ids
        if catalog.has_track(seed_id)
    }
    positive_albums = {
        catalog.normalized_field(seed_id, "album_name")
        for seed_id in state.positive_seed_ids
        if catalog.has_track(seed_id)
    }
    negative_artists = {
        catalog.normalized_field(seed_id, "artist_name")
        for seed_id in state.negative_seed_ids
        if catalog.has_track(seed_id)
    }
    negative_albums = {
        catalog.normalized_field(seed_id, "album_name")
        for seed_id in state.negative_seed_ids
        if catalog.has_track(seed_id)
    }
    release_year = catalog.release_year(track_id)
    features: dict[str, float | str | int] = {
        "rrf_score": float(rrf_score),
        "rule_boost": float(score_candidate_boost(state, catalog, track_id)),
        "popularity": float(view.popularity),
        "log_popularity": safe_log1p(view.popularity),
        "release_year": release_year or 0,
        "duration": view.duration or 0,
        "previously_recommended": 1 if track_id in state.previous_music_track_ids else 0,
        "same_artist_as_positive_seed": 1 if artist in positive_artists else 0,
        "same_album_as_positive_seed": 1 if album in positive_albums else 0,
        "same_artist_as_negative_seed": 1 if artist in negative_artists else 0,
        "same_album_as_negative_seed": 1 if album in negative_albums else 0,
        "num_positive_seeds": len(state.positive_seed_ids),
        "num_negative_seeds": len(state.negative_seed_ids),
        "turn_number": state.turn_number,
        "intent": infer_intent(state),
        "category": state.category,
        "specificity": state.specificity,
    }
    features.update(rank_features(source_ranks))
    return features


def main() -> None:
    args = parse_args()
    config = GoalFlowConfig(
        project_root=Path(args.project_root),
        tid=args.tid,
        use_train_augmentation=not args.no_train_augmentation,
        rebuild_cache=args.rebuild_cache,
        retrieval_top_k=args.retrieval_top_k,
        rerank_pool_size=args.rerank_pool_size,
        rrf_k=args.rrf_k,
        dev_limit=args.dev_limit,
    )
    catalog = TrackCatalog(config.track_metadata_name)
    retriever = prepare_retriever(config, catalog)
    states = build_dev_states(config)
    variants = [query_variants(state, catalog) for state in states]
    source_rows = retriever.batch_search(
        query_variants_per_state=variants,
        top_k_by_index=top_k_by_index(config),
        query_weights=default_query_weights(),
        index_weights=default_index_weights(),
    )

    out_dir = config.experiments_dir / "ltr"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "dev_ltr_candidates.jsonl"
    summary = {
        "states": len(states),
        "rows": 0,
        "groups_with_positive": 0,
        "groups_missing_positive": 0,
        "max_candidates_per_group": args.max_candidates_per_group,
    }

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for group_index, (state, sources) in enumerate(
            tqdm(list(zip(states, source_rows)), desc="Export LTR rows")
        ):
            gold = state.gold_track_id
            fused = rrf_fuse(sources, rrf_k=config.rrf_k)
            candidates = sorted(fused.values(), key=lambda item: item.score, reverse=True)[
                : args.max_candidates_per_group
            ]
            if gold and gold not in {candidate.track_id for candidate in candidates}:
                summary["groups_missing_positive"] += 1
                if args.include_missed_gold and catalog.has_track(gold):
                    candidates.append(CandidateScore(track_id=gold, score=0.0))

            has_positive = False
            for candidate in candidates:
                label = 1 if candidate.track_id == gold else 0
                has_positive = has_positive or bool(label)
                row = {
                    "group_id": group_index,
                    "session_id": state.session_id,
                    "user_id": state.user_id,
                    "turn_number": state.turn_number,
                    "track_id": candidate.track_id,
                    "label": label,
                    "features": candidate_features(
                        state=state,
                        catalog=catalog,
                        track_id=candidate.track_id,
                        rrf_score=candidate.score,
                        source_ranks=dict(candidate.source_ranks),
                    ),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary["rows"] += 1
            if has_positive:
                summary["groups_with_positive"] += 1

    summary_path = out_dir / "dev_ltr_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"jsonl={jsonl_path}")
    print(f"summary={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
