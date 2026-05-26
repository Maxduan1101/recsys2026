from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute stable hashes for prediction ranking and response content.")
    parser.add_argument("prediction_json", nargs="+")
    return parser.parse_args()


def summarize(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    ranking_payload = [
        {
            "session_id": row.get("session_id"),
            "user_id": row.get("user_id"),
            "turn_number": row.get("turn_number"),
            "predicted_track_ids": row.get("predicted_track_ids", []),
        }
        for row in rows
    ]
    response_payload = [
        {
            "session_id": row.get("session_id"),
            "turn_number": row.get("turn_number"),
            "predicted_response": row.get("predicted_response", ""),
        }
        for row in rows
    ]
    all_track_ids = [
        track_id
        for row in rows
        for track_id in row.get("predicted_track_ids", [])
    ]
    return {
        "path": str(path),
        "rows": len(rows),
        "slots": len(all_track_ids),
        "unique_tracks": len(set(all_track_ids)),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "ranking_sha256": stable_hash(ranking_payload),
        "response_sha256": stable_hash(response_payload),
    }


def main() -> None:
    summaries = [summarize(Path(raw)) for raw in parse_args().prediction_json]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
