from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend ranked prediction JSONL files with weighted reciprocal-log rank scores.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Prediction input as name:path:weight. Can be repeated.",
    )
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def parse_input(value: str) -> tuple[str, Path, float]:
    parts = value.split(":")
    if len(parts) < 3:
        raise ValueError(f"Expected name:path:weight, got {value}")
    name = parts[0]
    weight = float(parts[-1])
    path = Path(":".join(parts[1:-1]))
    return name, path, weight


def load_jsonl(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["session_id"]), str(row["user_id"]), int(row["turn_number"]))
            rows[key] = row
    return rows


def main() -> None:
    args = parse_args()
    specs = [parse_input(value) for value in args.input]
    loaded = [(name, load_jsonl(path), weight, str(path)) for name, path, weight in specs]
    keys = sorted(set.intersection(*(set(rows) for _, rows, _, _ in loaded)))
    if not keys:
        raise RuntimeError("No shared prediction keys across inputs.")

    predictions = []
    hit20 = 0
    ndcg20 = 0.0
    missing_gold = 0
    for key in keys:
        score: dict[str, float] = {}
        first_row = loaded[0][1][key]
        gold = first_row.get("gold_track_id")
        for _, rows, weight, _ in loaded:
            for rank, track_id in enumerate(rows[key]["predicted_track_ids"], start=1):
                score[track_id] = score.get(track_id, 0.0) + weight / math.log2(rank + 1)
        ranked = [track_id for track_id, _ in sorted(score.items(), key=lambda item: item[1], reverse=True)[: args.top_k]]
        gold_rank = None
        if gold:
            for rank, track_id in enumerate(ranked, start=1):
                if track_id == gold:
                    gold_rank = rank
                    break
            if gold_rank is not None and gold_rank <= 20:
                hit20 += 1
                ndcg20 += 1.0 / math.log2(gold_rank + 1)
        else:
            missing_gold += 1
        predictions.append(
            {
                "session_id": key[0],
                "user_id": key[1],
                "turn_number": key[2],
                "gold_track_id": gold,
                "gold_rank": gold_rank,
                "predicted_track_ids": ranked,
                "ensemble_inputs": [
                    {"name": name, "weight": weight, "path": path}
                    for name, _, weight, path in loaded
                ],
            }
        )

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "method": "rank_list_ensemble",
        "inputs": json.dumps(
            [{"name": name, "weight": weight, "path": path} for name, _, weight, path in loaded],
            ensure_ascii=False,
        ),
        "shared_groups": len(keys),
        "hit20": hit20,
        "ndcg20": ndcg20 / len(keys),
        "missing_gold": missing_gold,
        "top_k": args.top_k,
    }
    metrics_csv = Path(args.metrics_csv)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
