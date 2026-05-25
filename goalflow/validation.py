from __future__ import annotations

from collections import Counter

from .data import TrackCatalog


def validate_predictions(predictions: list[dict], catalog: TrackCatalog, expected_count: int | None = None) -> dict:
    errors = []
    if expected_count is not None and len(predictions) != expected_count:
        errors.append(f"expected {expected_count} predictions, found {len(predictions)}")

    seen_keys = Counter()
    for index, item in enumerate(predictions):
        for field in ["session_id", "user_id", "turn_number", "predicted_track_ids", "predicted_response"]:
            if field not in item:
                errors.append(f"row {index} missing field {field}")
        key = (item.get("session_id"), item.get("turn_number"))
        seen_keys[key] += 1
        track_ids = item.get("predicted_track_ids", [])
        if len(track_ids) != 20:
            errors.append(f"row {index} has {len(track_ids)} tracks, expected 20")
        if len(track_ids) != len(set(track_ids)):
            errors.append(f"row {index} has duplicate track ids")
        invalid = [track_id for track_id in track_ids if not catalog.has_track(track_id)]
        if invalid:
            errors.append(f"row {index} has invalid track ids: {invalid[:3]}")

    duplicates = [key for key, count in seen_keys.items() if count > 1]
    if duplicates:
        errors.append(f"duplicate session/turn keys: {duplicates[:5]}")

    return {"ok": not errors, "errors": errors[:50], "num_errors": len(errors)}
