from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from goalflow.data import TRACK_METADATA, TrackCatalog


def distinct_n(responses: list[str], n: int = 2) -> float:
    ngrams = set()
    total = 0
    for response in responses:
        tokens = (response or "").lower().split()
        for index in range(len(tokens) - n + 1):
            ngrams.add(tuple(tokens[index : index + n]))
            total += 1
    return len(ngrams) / total if total else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize prediction diversity without gold labels.")
    parser.add_argument("prediction_json")
    parser.add_argument("--track-metadata-name", default=TRACK_METADATA)
    parser.add_argument("--top-repeat", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.prediction_json)
    rows = json.loads(path.read_text(encoding="utf-8"))
    catalog = TrackCatalog(args.track_metadata_name)
    track_ids = [track_id for row in rows for track_id in row["predicted_track_ids"]]
    responses = [row.get("predicted_response", "") for row in rows]
    counts = Counter(track_ids)
    max_unique = len(rows) * 20
    summary = {
        "path": str(path),
        "rows": len(rows),
        "recommended_slots": len(track_ids),
        "unique_tracks": len(counts),
        "max_unique_tracks": max_unique,
        "unique_slot_ratio": len(counts) / max_unique if max_unique else 0.0,
        "catalog_diversity": len(counts) / len(catalog),
        "catalog_diversity_ceiling": max_unique / len(catalog) if len(catalog) else 0.0,
        "distinct_2": distinct_n(responses, n=2),
        "top_repeated_tracks": [
            {
                "track_id": track_id,
                "count": count,
                "track": catalog.view(track_id).track_name if catalog.has_track(track_id) else "",
                "artist": catalog.view(track_id).artist_name if catalog.has_track(track_id) else "",
            }
            for track_id, count in counts.most_common(args.top_repeat)
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
