from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from goalflow.data import TrackCatalog
from goalflow.validation import validate_predictions


NOISY_PHRASES = (
    "albums i own",
    "cds i own",
    "funk tag",
    "hip hop tag",
    "lastfm",
    "lobpreis",
    "not metal",
    "seen live",
    "sexist metal",
    "songsof",
)

MECHANICAL_PATTERNS = (
    "album-clue lead",
    "artist-path opener",
    "backups:",
    "catalog signal:",
    "closest catalog answer",
    "exact-track lead",
    "first answer:",
    "first test for the",
    "grounding:",
    "lead pick:",
    "likely specific-song match",
    "mood/discovery lead",
    "reason:",
    "record-context pick",
    "verified cue:",
    "visual-clue anchor",
)

GENERIC_PATTERNS = (
    "a few tracks that should fit",
    "direction you described",
    "songs you might enjoy",
    "you might like",
)


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranking_payload(rows: list[dict]) -> list[dict]:
    return [
        {
            "session_id": row.get("session_id"),
            "user_id": row.get("user_id"),
            "turn_number": row.get("turn_number"),
            "predicted_track_ids": row.get("predicted_track_ids", []),
        }
        for row in rows
    ]


def response_payload(rows: list[dict]) -> list[dict]:
    return [
        {
            "session_id": row.get("session_id"),
            "turn_number": row.get("turn_number"),
            "predicted_response": row.get("predicted_response", ""),
        }
        for row in rows
    ]


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def distinct_n(responses: list[str], n: int = 2) -> float:
    seen = set()
    total = 0
    for response in responses:
        row_tokens = tokens(response)
        for index in range(len(row_tokens) - n + 1):
            seen.add(tuple(row_tokens[index : index + n]))
            total += 1
    return len(seen) / total if total else 0.0


def normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def mention_variants(value: object) -> list[str]:
    raw = str(value or "")
    pieces = [raw]
    pieces.extend(part.strip() for part in raw.split(","))
    variants = []
    seen = set()
    for piece in pieces:
        text = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", piece.lower())
        text = re.sub(
            r"\b(remaster(?:ed)?|explicit|album|version|edit|mix|original|feat(?:uring)?|live)\b",
            " ",
            text,
        )
        key = normalized(text)
        if len(key) >= 2 and key not in seen:
            variants.append(key)
            seen.add(key)
    return variants


def response_mentions_any(response_norm: str, candidates: list[str]) -> bool:
    return any(candidate and candidate in response_norm for candidate in candidates)


def top1_mention_misses(rows: list[dict], catalog: TrackCatalog, limit: int) -> list[dict]:
    misses = []
    for index, row in enumerate(rows):
        track_ids = row.get("predicted_track_ids") or []
        if not track_ids:
            continue
        track_id = track_ids[0]
        if not catalog.has_track(track_id):
            continue
        view = catalog.view(track_id)
        response_norm = normalized(row.get("predicted_response", ""))
        title_hit = response_mentions_any(response_norm, mention_variants(view.track_name))
        artist_hit = response_mentions_any(response_norm, mention_variants(view.artist_name))
        if not (title_hit and artist_hit):
            misses.append(
                {
                    "row": index,
                    "session_id": row.get("session_id"),
                    "turn_number": row.get("turn_number"),
                    "track_id": track_id,
                    "track": view.track_name,
                    "artist": view.artist_name,
                    "title_hit": title_hit,
                    "artist_hit": artist_hit,
                    "response": row.get("predicted_response", ""),
                }
            )
            if len(misses) >= limit:
                break
    return misses


def response_risks(rows: list[dict], catalog: TrackCatalog, top: int) -> dict:
    responses = [re.sub(r"\s+", " ", row.get("predicted_response", "") or "").strip() for row in rows]
    word_counts = [len(tokens(response)) for response in responses]
    openings = Counter(" ".join(tokens(response)[:4]) for response in responses if response)

    noisy_hits = []
    mechanical_hits = []
    generic_hits = []
    for index, response in enumerate(responses):
        lower = response.lower()
        noisy = [phrase for phrase in NOISY_PHRASES if phrase in lower]
        mechanical = [phrase for phrase in MECHANICAL_PATTERNS if phrase in lower]
        generic = [phrase for phrase in GENERIC_PATTERNS if phrase in lower]
        if noisy:
            noisy_hits.append(
                {
                    "row": index,
                    "session_id": rows[index].get("session_id"),
                    "turn_number": rows[index].get("turn_number"),
                    "hits": noisy,
                    "response": response,
                }
            )
        if mechanical:
            mechanical_hits.append(
                {
                    "row": index,
                    "session_id": rows[index].get("session_id"),
                    "turn_number": rows[index].get("turn_number"),
                    "hits": mechanical,
                    "response": response,
                }
            )
        if generic:
            generic_hits.append(
                {
                    "row": index,
                    "session_id": rows[index].get("session_id"),
                    "turn_number": rows[index].get("turn_number"),
                    "hits": generic,
                    "response": response,
                }
            )

    long_rows = [
        {
            "row": index,
            "session_id": rows[index].get("session_id"),
            "turn_number": rows[index].get("turn_number"),
            "words": count,
            "response": responses[index],
        }
        for index, count in enumerate(word_counts)
        if count > 115
    ]
    short_rows = [
        {
            "row": index,
            "session_id": rows[index].get("session_id"),
            "turn_number": rows[index].get("turn_number"),
            "words": count,
            "response": responses[index],
        }
        for index, count in enumerate(word_counts)
        if count < 24
    ]

    return {
        "distinct_2": distinct_n(responses, 2),
        "word_count": {
            "min": min(word_counts) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
            "avg": sum(word_counts) / len(word_counts) if word_counts else 0.0,
        },
        "top_openings": openings.most_common(top),
        "noisy_hit_count": len(noisy_hits),
        "mechanical_hit_count": len(mechanical_hits),
        "generic_hit_count": len(generic_hits),
        "long_count": len(long_rows),
        "short_count": len(short_rows),
        "top1_mention_misses": top1_mention_misses(rows, catalog, top),
        "noisy_hits": noisy_hits[:top],
        "mechanical_hits": mechanical_hits[:top],
        "generic_hits": generic_hits[:top],
        "long_rows": long_rows[:top],
        "short_rows": short_rows[:top],
    }


def zip_summary(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        return {"path": str(path), "exists": False, "ok": False, "entries": []}
    with zipfile.ZipFile(path) as zf:
        entries = zf.namelist()
    return {
        "path": str(path),
        "exists": True,
        "ok": entries == ["prediction.json"],
        "entries": entries,
        "sha256": file_hash(path),
    }


def artifact_summary(
    *,
    label: str,
    prediction_path: Path,
    zip_path: Path | None,
    catalog: TrackCatalog,
    expected_count: int | None,
    top: int,
) -> dict:
    rows = load_rows(prediction_path)
    track_ids = [track_id for row in rows for track_id in row.get("predicted_track_ids", [])]
    track_counts = Counter(track_ids)
    validation = validate_predictions(rows, catalog, expected_count=expected_count)
    return {
        "label": label,
        "prediction_path": str(prediction_path),
        "prediction_sha256": file_hash(prediction_path),
        "zip": zip_summary(zip_path),
        "validation": validation,
        "rows": len(rows),
        "recommended_slots": len(track_ids),
        "unique_tracks": len(track_counts),
        "unique_slot_ratio": len(track_counts) / len(track_ids) if track_ids else 0.0,
        "catalog_diversity": len(track_counts) / len(catalog) if len(catalog) else 0.0,
        "ranking_sha256": stable_hash(ranking_payload(rows)),
        "response_sha256": stable_hash(response_payload(rows)),
        "response_risks": response_risks(rows, catalog, top),
    }


def parse_artifact(value: str) -> tuple[str, Path, Path | None]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("artifact must be label=prediction.json[:submission.zip]")
    label, paths = parts
    path_parts = paths.split(":", 1)
    prediction_path = Path(path_parts[0])
    zip_path = Path(path_parts[1]) if len(path_parts) == 2 and path_parts[1] else None
    return label, prediction_path, zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final frozen-artifact audit for RecSys Music-CRS submissions.")
    parser.add_argument(
        "--artifact",
        action="append",
        type=parse_artifact,
        required=True,
        help="label=prediction.json[:submission.zip]. Repeat for primary/backups.",
    )
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = TrackCatalog()
    summaries = [
        artifact_summary(
            label=label,
            prediction_path=prediction_path,
            zip_path=zip_path,
            catalog=catalog,
            expected_count=args.expected_count,
            top=args.top,
        )
        for label, prediction_path, zip_path in args.artifact
    ]
    primary_ranking_hash = summaries[0]["ranking_sha256"] if summaries else ""
    for summary in summaries:
        summary["same_ranking_as_primary"] = summary["ranking_sha256"] == primary_ranking_hash

    output = {"artifacts": summaries}
    text = json.dumps(output, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")

    severe = []
    for summary in summaries:
        if not summary["validation"]["ok"]:
            severe.append(f"{summary['label']}: validation failed")
        if summary["zip"] and not summary["zip"]["ok"]:
            severe.append(f"{summary['label']}: zip entries are not exactly prediction.json")
        risks = summary["response_risks"]
        if risks["noisy_hit_count"] or risks["long_count"] or risks["short_count"]:
            severe.append(
                f"{summary['label']}: response risk noisy={risks['noisy_hit_count']} "
                f"long={risks['long_count']} short={risks['short_count']}"
            )
        if risks["top1_mention_misses"]:
            severe.append(f"{summary['label']}: top1 title/artist mention misses present")
    if severe:
        raise SystemExit("Final audit found severe issues:\n" + "\n".join(severe))


if __name__ == "__main__":
    main()
