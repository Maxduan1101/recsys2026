from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare union and nextgen OOF prediction ranks.")
    parser.add_argument("--union-pred", default="goalflow_musiccrs/experiments/rerank_v2_independent_features/predictions_oof_union.jsonl")
    parser.add_argument("--nextgen-pred", default="goalflow_musiccrs/experiments/rerank_v2_nextgen_v1/predictions_oof_nextgen.jsonl")
    parser.add_argument("--out-dir", default="goalflow_musiccrs/experiments/rerank_v2_nextgen_v1")
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["session_id"]), str(row["user_id"]), int(row["turn_number"]))
            rows[key] = row
    return rows


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "absent"
    if rank <= 20:
        return "<=20"
    if rank <= 50:
        return "21-50"
    if rank <= 100:
        return "51-100"
    if rank <= 300:
        return "101-300"
    if rank <= 800:
        return "301-800"
    return ">800"


def summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [row["nextgen_rank"] for row in rows if row["nextgen_rank"] is not None]
    return {
        "bucket": name,
        "count": len(rows),
        "nextgen_rank_mean": mean(ranks) if ranks else "",
        "nextgen_rank_p50": median(ranks) if ranks else "",
        "nextgen_rank_le20": sum(1 for rank in ranks if rank <= 20),
        "nextgen_rank_le50": sum(1 for rank in ranks if rank <= 50),
        "nextgen_rank_le100": sum(1 for rank in ranks if rank <= 100),
        "nextgen_rank_le300": sum(1 for rank in ranks if rank <= 300),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    union = load_jsonl(Path(args.union_pred))
    nextgen = load_jsonl(Path(args.nextgen_pred))
    rows = []
    for key, next_row in sorted(nextgen.items()):
        union_row = union.get(key)
        if union_row is None:
            continue
        union_rank = union_row.get("gold_rank")
        next_rank = next_row.get("gold_rank")
        if union_rank is None and next_rank is not None:
            delta_type = "extra_gold_in_pool"
        elif union_rank is not None and next_rank is None:
            delta_type = "lost_gold_from_pool"
        elif union_rank is not None and next_rank is not None:
            delta_type = "both_in_pool"
        else:
            delta_type = "both_absent"
        rows.append(
            {
                "session_id": key[0],
                "user_id": key[1],
                "turn_number": key[2],
                "gold_track_id": next_row.get("gold_track_id"),
                "union_rank": union_rank,
                "nextgen_rank": next_rank,
                "union_rank_bucket": rank_bucket(union_rank),
                "nextgen_rank_bucket": rank_bucket(next_rank),
                "delta_type": delta_type,
                "rank_delta_next_minus_union": (next_rank - union_rank) if union_rank is not None and next_rank is not None else "",
                "union_top20_hit": int(union_rank is not None and union_rank <= 20),
                "nextgen_top20_hit": int(next_rank is not None and next_rank <= 20),
            }
        )

    detail_path = out_dir / "nextgen_vs_union_rank_delta.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = {
        "extra_gold_in_pool": [row for row in rows if row["delta_type"] == "extra_gold_in_pool"],
        "both_in_pool": [row for row in rows if row["delta_type"] == "both_in_pool"],
        "union_top20_lost": [
            row for row in rows if row["union_top20_hit"] == 1 and row["nextgen_top20_hit"] == 0
        ],
        "nextgen_top20_gained": [
            row for row in rows if row["union_top20_hit"] == 0 and row["nextgen_top20_hit"] == 1
        ],
        "both_absent": [row for row in rows if row["delta_type"] == "both_absent"],
    }
    summary_rows = [summarize(name, group_rows) for name, group_rows in groups.items()]
    summary_path = out_dir / "nextgen_vs_union_rank_delta_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"wrote {detail_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
