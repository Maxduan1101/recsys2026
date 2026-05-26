from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


NOISY_PHRASES = (
    "albums i own",
    "cds i own",
    "favourites",
    "favorites",
    "funk tag",
    "hip hop tag",
    "lastfm",
    "lobpreis",
    "not metal",
    "playlist",
    "seen live",
    "sexist metal",
    "songsof",
)


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def opening_key(response: str, words: int = 4) -> str:
    tokens = re.findall(r"[A-Za-z0-9']+", response.lower())
    return " ".join(tokens[:words])


def distinct_n(responses: list[str], n: int = 2) -> float:
    total = 0
    seen = set()
    for response in responses:
        tokens = re.findall(r"[A-Za-z0-9']+", response.lower())
        for index in range(len(tokens) - n + 1):
            seen.add(tuple(tokens[index : index + n]))
            total += 1
    return len(seen) / total if total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gold-free response text audit for submission candidates.")
    parser.add_argument("prediction_json")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--long-words", type=int, default=115)
    parser.add_argument("--short-words", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(Path(args.prediction_json).read_text(encoding="utf-8"))
    responses = [normalized_text(row.get("predicted_response", "")) for row in rows]
    word_counts = [len(re.findall(r"[A-Za-z0-9']+", response)) for response in responses]
    openings = Counter(opening_key(response) for response in responses if response)
    noisy_hits = []
    for index, response in enumerate(responses):
        lower = response.lower()
        hits = [phrase for phrase in NOISY_PHRASES if phrase in lower]
        if hits:
            row = rows[index]
            noisy_hits.append(
                {
                    "row": index,
                    "session_id": row.get("session_id"),
                    "turn_number": row.get("turn_number"),
                    "hits": hits,
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
        if count > args.long_words
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
        if count < args.short_words
    ]

    summary = {
        "path": args.prediction_json,
        "rows": len(rows),
        "distinct_2": distinct_n(responses, 2),
        "word_count": {
            "min": min(word_counts) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
            "avg": sum(word_counts) / len(word_counts) if word_counts else 0.0,
        },
        "noisy_hit_count": len(noisy_hits),
        "long_count": len(long_rows),
        "short_count": len(short_rows),
        "top_openings": openings.most_common(args.top),
        "noisy_hits": noisy_hits[: args.top],
        "long_rows": long_rows[: args.top],
        "short_rows": short_rows[: args.top],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
