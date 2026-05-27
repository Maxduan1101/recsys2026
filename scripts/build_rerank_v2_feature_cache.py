from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from augment_rerank_features_v3 import augment_features
from goalflow.data import BLIND_A_DATASET, TrackCatalog
from goalflow.embeddings import TRACK_EMBEDDING_CHANNELS, TrackEmbeddingStore, UserEmbeddingStore
from goalflow.pipeline import GoalFlowConfig
from goalflow.state import ConversationState, build_state_for_blind_item
from run_rerank_v2 import (
    TrackTextCache,
    build_feature_frame,
    load_dev_states,
    normalize_pool_base,
    stable_json_hash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a rerank_v2 feature cache for a saved candidate pool. "
            "Unlike run_rerank_v2.py this works for both dev and blind pools."
        )
    )
    parser.add_argument("--project-root", default="goalflow_musiccrs")
    parser.add_argument("--mode", choices=["dev", "blind"], default="dev")
    parser.add_argument("--blind-dataset-name", default=BLIND_A_DATASET)
    parser.add_argument("--pool-name", default="nextgen")
    parser.add_argument("--pool-pkl", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--meta-json", default="")
    parser.add_argument(
        "--embedding-channels",
        default="metadata,attributes,lyrics,audio,image,track_cf",
        help="Comma-separated channel names from goalflow.embeddings, or empty to disable.",
    )
    parser.add_argument("--disable-user-cf", action="store_true")
    parser.add_argument(
        "--augment-v3",
        action="store_true",
        help="Add semantic interaction features from augment_rerank_features_v3.py before writing.",
    )
    return parser.parse_args()


def load_blind_states(config: GoalFlowConfig) -> list[ConversationState]:
    dataset = load_dataset(config.blind_dataset_name, split="test")
    return [build_state_for_blind_item(item) for item in dataset]


def normalize_apply_pool(df: pd.DataFrame, states: list[ConversationState]) -> pd.DataFrame:
    out = df.copy()
    if "group_id" not in out:
        raise ValueError("Candidate pool must contain group_id")
    if "track_id" not in out:
        raise ValueError("Candidate pool must contain track_id")
    out["group_id"] = out["group_id"].astype(int)
    out["track_id"] = out["track_id"].astype(str)

    def state_value(group_id: Any, attr: str) -> Any:
        state = states[int(group_id)]
        return getattr(state, attr)

    for column, attr in [
        ("session_id", "session_id"),
        ("user_id", "user_id"),
        ("turn_number", "turn_number"),
    ]:
        if column not in out:
            out[column] = [state_value(group_id, attr) for group_id in out["group_id"]]
    if "gold_track_id" not in out:
        out["gold_track_id"] = [states[int(group_id)].gold_track_id or "" for group_id in out["group_id"]]
    else:
        out["gold_track_id"] = out["gold_track_id"].fillna("").astype(str)
    if "label" not in out:
        out["label"] = (out["track_id"] == out["gold_track_id"]).astype(int)
    else:
        out["label"] = out["label"].fillna(0).astype(int)

    # This string column is useful in pool diagnostics but cannot be used as a numeric LTR feature.
    out = out.drop(columns=["nextgen_family_names"], errors="ignore")
    return normalize_pool_base(out)


def main() -> None:
    args = parse_args()
    config = GoalFlowConfig(project_root=Path(args.project_root), tid="build_rerank_v2_feature_cache")
    config.blind_dataset_name = args.blind_dataset_name

    states = load_dev_states(config) if args.mode == "dev" else load_blind_states(config)
    pool = pd.read_pickle(args.pool_pkl)
    pool = normalize_apply_pool(pool, states)

    catalog = TrackCatalog(config.track_metadata_name)
    text_cache = TrackTextCache(catalog)
    embedding_channels = [item.strip() for item in args.embedding_channels.split(",") if item.strip()]
    track_embeddings = None
    user_embeddings = None
    if embedding_channels:
        channel_map = {name: TRACK_EMBEDDING_CHANNELS[name] for name in embedding_channels}
        track_embeddings = TrackEmbeddingStore(channels=channel_map)
        if not args.disable_user_cf and "track_cf" in embedding_channels:
            user_embeddings = UserEmbeddingStore()

    features = build_feature_frame(
        pool_name=args.pool_name,
        base_df=pool,
        states=states,
        catalog=catalog,
        text_cache=text_cache,
        track_embeddings=track_embeddings,
        user_embeddings=user_embeddings,
        embedding_channels=embedding_channels,
    )
    added_v3: list[str] = []
    if args.augment_v3:
        features, added_v3 = augment_features(features)

    output = Path(args.output_pkl)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_pickle(output)
    meta_path = Path(args.meta_json) if args.meta_json else output.with_suffix(".json")
    meta = {
        "mode": args.mode,
        "pool_name": args.pool_name,
        "pool_pkl": args.pool_pkl,
        "rows": int(len(features)),
        "groups": int(features["group_id"].nunique()),
        "columns": list(features.columns),
        "feature_hash": stable_json_hash(list(features.columns)),
        "embedding_channels": embedding_channels,
        "augment_v3": bool(args.augment_v3),
        "v3_added_columns": added_v3,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote features={output} rows={len(features)} groups={features['group_id'].nunique()} cols={len(features.columns)}")


if __name__ == "__main__":
    main()
