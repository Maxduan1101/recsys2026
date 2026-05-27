from __future__ import annotations

import argparse
import csv
from pathlib import Path


CHECKPOINTS = (20, 50, 100, 200, 300, 500, 800, 1200, 2000)
INTERVALS = ((20, 100), (100, 300), (300, 500), (500, 800), (800, 1200), (1200, 2000))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize per-source topK hit curves from evaluate_recall_pool.py output."
    )
    parser.add_argument(
        "--summary-csv",
        default="goalflow_musiccrs/experiments/recall_pool_top1000/recall_pool/recall_pool_summary.csv",
    )
    parser.add_argument("--group", default="overall")
    parser.add_argument(
        "--out-dir",
        default="goalflow_musiccrs/experiments/recall_pool_top1000/recall_pool/source_topk_curves",
    )
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def is_real_source(source: str) -> bool:
    return not (
        source.startswith("__")
        or source.startswith("index_any=")
        or source.startswith("query_any=")
    )


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else 0.0


def load_rows(summary_csv: Path, group: str) -> list[dict[str, object]]:
    rows = []
    with summary_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source = row["source"]
            if row["group"] != group or not is_real_source(source):
                continue
            out: dict[str, object] = {
                "source": source,
                "coverage": as_float(row, "coverage"),
                "mean_found_rank": as_float(row, "mean_found_rank"),
                "mrr": as_float(row, "mrr"),
            }
            for k in CHECKPOINTS:
                out[f"hit@{k}"] = as_float(row, f"hit@{k}")
            for start, end in INTERVALS:
                out[f"gain_{start}_{end}"] = out[f"hit@{end}"] - out[f"hit@{start}"]
                out[f"gain_per_100_{start}_{end}"] = 100.0 * (out[f"hit@{end}"] - out[f"hit@{start}"]) / (end - start)
            rows.append(out)
    return rows


def recommended_k(row: dict[str, object]) -> int:
    # Simple elbow heuristic: keep increasing K while the interval adds at least
    # two percentage points of hit rate. This is only a first-pass diagnostic.
    last_good = 20
    for start, end in INTERVALS:
        if float(row[f"gain_{start}_{end}"]) >= 0.02:
            last_good = end
    return last_good


def write_csv(rows: list[dict[str, object]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "source_topk_curves.csv"
    fieldnames = [
        "source",
        "recommended_k",
        "coverage",
        "mean_found_rank",
        "mrr",
        *[f"hit@{k}" for k in CHECKPOINTS],
        *[f"gain_{start}_{end}" for start, end in INTERVALS],
        *[f"gain_per_100_{start}_{end}" for start, end in INTERVALS],
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["recommended_k"] = recommended_k(row)
            writer.writerow(out)
    return path


def write_report(rows: list[dict[str, object]], out_dir: Path, top_n: int) -> Path:
    path = out_dir / "source_topk_gain_report.md"
    sections = [
        ("coverage", "最大覆盖率最高的 source"),
        ("gain_20_100", "top20 到 top100 收益最大的 source"),
        ("gain_100_300", "top100 到 top300 收益最大的 source"),
        ("gain_300_500", "top300 到 top500 收益最大的 source"),
        ("gain_500_800", "top500 到 top800 收益最大的 source"),
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Source TopK Curves\n\n")
        f.write("这份报告只看每个 source 自己的 hit@K 曲线，不看 source 之间的重叠。\n\n")
        f.write("它回答的问题是：某个 source 的 K 继续放大时，单独还能多找到多少 gold。\n\n")
        for key, title in sections:
            f.write(f"## {title}\n\n")
            f.write("| source | recommended_k | coverage | hit@20 | hit@100 | hit@300 | hit@500 | hit@800 | gain |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in sorted(rows, key=lambda item: float(item[key]), reverse=True)[:top_n]:
                f.write(
                    f"| {row['source']} | {recommended_k(row)} | {float(row['coverage']):.4f} | "
                    f"{float(row['hit@20']):.4f} | {float(row['hit@100']):.4f} | "
                    f"{float(row['hit@300']):.4f} | {float(row['hit@500']):.4f} | "
                    f"{float(row['hit@800']):.4f} | {float(row[key]):.4f} |\n"
                )
            f.write("\n")
    return path


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.summary_csv), args.group)
    rows.sort(key=lambda row: float(row["coverage"]), reverse=True)
    out_dir = Path(args.out_dir)
    csv_path = write_csv(rows, out_dir)
    report_path = write_report(rows, out_dir, args.top_n)
    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"report={report_path}")
    for row in rows[: min(10, len(rows))]:
        print(
            f"{row['source']}: recommended_k={recommended_k(row)} "
            f"coverage={float(row['coverage']):.4f} "
            f"hit@100={float(row['hit@100']):.4f} hit@500={float(row['hit@500']):.4f}"
        )


if __name__ == "__main__":
    main()

