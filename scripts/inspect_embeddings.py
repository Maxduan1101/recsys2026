from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import get_dataset_split_names, load_dataset

from goalflow.data import TRACK_EMBEDDINGS, USER_EMBEDDINGS, TrackCatalog
from goalflow.embeddings import TRACK_EMBEDDING_CHANNELS, TrackEmbeddingStore, UserEmbeddingStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect official Challenge embedding datasets.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--load-store", action="store_true")
    return parser.parse_args()


def vector_length(value) -> int | None:
    return len(value) if isinstance(value, list) else None


def dataset_summary(name: str, first_split: str) -> dict:
    dataset = load_dataset(name, split=first_split)
    row = dataset[0]
    return {
        "name": name,
        "splits": get_dataset_split_names(name),
        "first_split": first_split,
        "num_rows_first_split": len(dataset),
        "columns": dataset.column_names,
        "vector_lengths": {
            key: vector_length(value)
            for key, value in row.items()
            if vector_length(value) is not None
        },
    }


def main() -> None:
    args = parse_args()
    catalog = TrackCatalog()
    track_summary = dataset_summary(TRACK_EMBEDDINGS, "all_tracks")
    user_summary = dataset_summary(USER_EMBEDDINGS, "train")
    track_dataset = load_dataset(TRACK_EMBEDDINGS, split="all_tracks")
    test_dataset = load_dataset(TRACK_EMBEDDINGS, split="test_tracks")
    all_track_ids = set(track_dataset["track_id"])
    test_track_ids = set(test_dataset["track_id"])
    summary = {
        "track_embeddings": track_summary,
        "user_embeddings": user_summary,
        "track_channels": TRACK_EMBEDDING_CHANNELS,
        "catalog_tracks": len(catalog.track_ids),
        "embedding_all_tracks": len(all_track_ids),
        "embedding_test_tracks": len(test_track_ids),
        "catalog_overlap_all_tracks": sum(1 for track_id in catalog.track_ids if track_id in all_track_ids),
        "catalog_overlap_test_tracks": sum(1 for track_id in catalog.track_ids if track_id in test_track_ids),
    }
    if args.load_store:
        track_store = TrackEmbeddingStore()
        user_store = UserEmbeddingStore()
        summary["loaded_store"] = {
            "track_count": len(track_store.track_ids),
            "user_count": len(user_store.user_vectors),
            "channels": {
                name: {
                    "shape": list(channel.raw.shape),
                    "valid": int(channel.valid.sum()),
                }
                for name, channel in track_store.matrices.items()
            },
        }

    out_dir = Path(args.project_root) / "experiments" / "embedding_schema"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"summary={out_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
