from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge a protected legacy head with a GoalFlow prediction file.")
    parser.add_argument("--legacy-json", required=True)
    parser.add_argument("--goalflow-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--head-k", type=int, default=10)
    args = parser.parse_args()

    with open(args.legacy_json, "r", encoding="utf-8") as f:
        legacy = json.load(f)
    with open(args.goalflow_json, "r", encoding="utf-8") as f:
        goalflow = json.load(f)

    legacy_by_key = {(row["session_id"], row["turn_number"]): row for row in legacy}
    merged = []
    for row in goalflow:
        key = (row["session_id"], row["turn_number"])
        legacy_tracks = legacy_by_key[key]["predicted_track_ids"]
        head = legacy_tracks[: args.head_k]
        tracks = head + [track_id for track_id in row["predicted_track_ids"] if track_id not in head]
        tracks += [track_id for track_id in legacy_tracks if track_id not in tracks]
        merged.append({**row, "predicted_track_ids": tracks[:20]})

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    print(output)


if __name__ == "__main__":
    main()
