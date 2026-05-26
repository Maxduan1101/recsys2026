from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def row_key(row: dict) -> str:
    return f"{row.get('session_id')}:{row.get('turn_number')}"


def opening_key(response: str, words: int = 4) -> str:
    tokens = re.findall(r"[A-Za-z0-9']+", response.lower())
    return " ".join(tokens[:words])


def word_count(response: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", response))


def hash_fraction(row: dict, salt: str) -> float:
    digest = hashlib.md5(f"{row_key(row)}:{salt}".encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12)


def choose_secondary(
    *,
    strategy: str,
    primary: list[dict],
    ratio: float,
    opening_cap: int,
    max_secondary_words: int,
    salt: str,
) -> set[int]:
    chosen: set[int] = set()
    if strategy == "ratio":
        for index, row in enumerate(primary):
            if hash_fraction(row, salt) < ratio:
                chosen.add(index)
        return chosen

    if strategy == "repeated_openings":
        seen = defaultdict(int)
        for index, row in enumerate(primary):
            key = opening_key(row.get("predicted_response", ""))
            seen[key] += 1
            if seen[key] > opening_cap:
                chosen.add(index)
        return chosen

    if strategy == "hybrid":
        openings = Counter(opening_key(row.get("predicted_response", "")) for row in primary)
        seen = defaultdict(int)
        for index, row in enumerate(primary):
            response = row.get("predicted_response", "")
            key = opening_key(response)
            seen[key] += 1
            repeated = openings[key] > opening_cap and seen[key] > max(1, opening_cap // 2)
            longish = word_count(response) > max_secondary_words
            sampled = hash_fraction(row, salt) < ratio
            if repeated or longish or sampled:
                chosen.add(index)
        return chosen

    raise ValueError(f"Unknown strategy: {strategy}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend responses from two prediction files without changing ranking.")
    parser.add_argument("--primary", required=True)
    parser.add_argument("--secondary", required=True)
    parser.add_argument("--tid", required=True)
    parser.add_argument("--mode", choices=["dev", "blind"], required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--strategy", choices=["ratio", "repeated_openings", "hybrid"], default="ratio")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--opening-cap", type=int, default=12)
    parser.add_argument("--max-secondary-words", type=int, default=88)
    parser.add_argument("--salt", default="blend")
    parser.add_argument("--copy-to-official-evaluator", action="store_true")
    parser.add_argument("--zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    primary = load_rows(Path(args.primary))
    secondary = load_rows(Path(args.secondary))
    if len(primary) != len(secondary):
        raise ValueError(f"Row count mismatch: {len(primary)} vs {len(secondary)}")
    if stable_hash(ranking_payload(primary)) != stable_hash(ranking_payload(secondary)):
        raise ValueError("Primary and secondary ranking payloads differ; refusing response-only blend.")

    secondary_by_key = {row_key(row): row for row in secondary}
    chosen = choose_secondary(
        strategy=args.strategy,
        primary=primary,
        ratio=args.ratio,
        opening_cap=args.opening_cap,
        max_secondary_words=args.max_secondary_words,
        salt=args.salt,
    )

    output_rows = []
    for index, row in enumerate(primary):
        out = dict(row)
        other = secondary_by_key[row_key(row)]
        if index in chosen:
            out["predicted_response"] = other.get("predicted_response", "")
        output_rows.append(out)

    if args.mode == "dev":
        output = project_root / "experiments" / args.tid / "devset" / f"{args.tid}.json"
    else:
        output = project_root / "experiments" / args.tid / "blindset_A" / "prediction.json"
    write_rows(output, output_rows)

    if args.copy_to_official_evaluator and args.mode == "dev":
        official = project_root.parent / "music-crs-evaluator" / "exp" / "inference" / "devset"
        official.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output, official / f"{args.tid}.json")

    zip_path = None
    if args.zip and args.mode == "blind":
        zip_path = output.parent / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(output, arcname="prediction.json")

    summary = {
        "output": str(output),
        "zip": str(zip_path) if zip_path else None,
        "rows": len(output_rows),
        "chosen_secondary": len(chosen),
        "chosen_secondary_rate": len(chosen) / len(output_rows) if output_rows else 0.0,
        "ranking_sha256": stable_hash(ranking_payload(output_rows)),
        "strategy": args.strategy,
        "ratio": args.ratio,
        "opening_cap": args.opening_cap,
        "max_secondary_words": args.max_secondary_words,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
