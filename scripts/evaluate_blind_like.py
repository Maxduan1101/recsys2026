from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from goalflow.data import BLIND_A_DATASET, CONVERSATION_DATASET
from goalflow.state import build_state_for_dev_turn, build_state_for_blind_item


def dcg_at(gold_track_id: str, ranked_track_ids: list[str], k: int) -> float:
    for index, track_id in enumerate(ranked_track_ids[:k], start=1):
        if track_id == gold_track_id:
            return 1.0 / math.log2(index + 1)
    return 0.0


def load_prediction_map(path: Path) -> dict[tuple[str, int], list[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (row["session_id"], int(row["turn_number"])): list(row["predicted_track_ids"])
        for row in rows
    }


def blind_strata(blind_dataset_name: str) -> Counter[tuple[int, str, str]]:
    dataset = load_dataset(blind_dataset_name, split="test")
    counts: Counter[tuple[int, str, str]] = Counter()
    for item in dataset:
        state = build_state_for_blind_item(item)
        counts[(state.turn_number, state.category, state.specificity)] += 1
    return counts


def dev_rows(conversation_dataset_name: str) -> list[dict]:
    dataset = load_dataset(conversation_dataset_name, split="test")
    rows = []
    for item in dataset:
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id:
                rows.append(
                    {
                        "key": (state.session_id, state.turn_number),
                        "session_id": state.session_id,
                        "turn_number": state.turn_number,
                        "category": state.category,
                        "specificity": state.specificity,
                        "gold_track_id": state.gold_track_id,
                    }
                )
    return rows


def sample_panel(
    rows_by_stratum: dict[tuple[int, str, str], list[dict]],
    targets: Counter[tuple[int, str, str]],
    rng: random.Random,
) -> list[dict]:
    panel = []
    for stratum, count in targets.items():
        candidates = rows_by_stratum.get(stratum, [])
        if len(candidates) >= count:
            panel.extend(rng.sample(candidates, count))
        elif candidates:
            panel.extend(rng.choices(candidates, k=count))
    return panel


def panel_metrics(panel: list[dict], prediction_map: dict[tuple[str, int], list[str]]) -> dict[str, float]:
    rows = []
    for row in panel:
        ranked = prediction_map.get(row["key"])
        if ranked:
            rows.append((row["gold_track_id"], ranked))
    if not rows:
        return {"rows": 0.0, "ndcg@1": 0.0, "ndcg@10": 0.0, "ndcg@20": 0.0, "hit@20": 0.0}
    return {
        "rows": float(len(rows)),
        "ndcg@1": sum(dcg_at(gold, ranked, 1) for gold, ranked in rows) / len(rows),
        "ndcg@10": sum(dcg_at(gold, ranked, 10) for gold, ranked in rows) / len(rows),
        "ndcg@20": sum(dcg_at(gold, ranked, 20) for gold, ranked in rows) / len(rows),
        "hit@20": sum(1.0 if dcg_at(gold, ranked, 20) > 0 else 0.0 for gold, ranked in rows) / len(rows),
    }


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "median": 0.0}
    ordered = sorted(values)
    return {
        "mean": sum(values) / len(values),
        "p10": ordered[max(0, min(len(ordered) - 1, int(0.10 * (len(ordered) - 1))))],
        "median": ordered[len(ordered) // 2],
    }


def parse_prediction_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    return label, Path(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dev predictions on Blind-A-shaped sampled panels.")
    parser.add_argument("--prediction", action="append", required=True, help="label=path, repeatable.")
    parser.add_argument("--baseline-label", default=None)
    parser.add_argument("--panels", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--conversation-dataset-name", default=CONVERSATION_DATASET)
    parser.add_argument("--blind-dataset-name", default=BLIND_A_DATASET)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    prediction_args = [parse_prediction_arg(value) for value in args.prediction]
    prediction_maps = {label: load_prediction_map(path) for label, path in prediction_args}
    baseline_label = args.baseline_label or prediction_args[0][0]

    targets = blind_strata(args.blind_dataset_name)
    rows = dev_rows(args.conversation_dataset_name)
    rows_by_stratum: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_stratum[(row["turn_number"], row["category"], row["specificity"])].append(row)

    rng = random.Random(args.seed)
    per_panel = []
    metrics_by_label: dict[str, list[dict[str, float]]] = {label: [] for label in prediction_maps}
    delta_by_label: dict[str, list[float]] = {label: [] for label in prediction_maps if label != baseline_label}

    for panel_index in range(args.panels):
        panel = sample_panel(rows_by_stratum, targets, rng)
        row = {"panel": panel_index, "rows": len(panel)}
        panel_metrics_by_label = {}
        for label, prediction_map in prediction_maps.items():
            metrics = panel_metrics(panel, prediction_map)
            panel_metrics_by_label[label] = metrics
            metrics_by_label[label].append(metrics)
            row[f"{label}_ndcg20"] = metrics["ndcg@20"]
        baseline_ndcg = panel_metrics_by_label[baseline_label]["ndcg@20"]
        for label, metrics in panel_metrics_by_label.items():
            if label == baseline_label:
                continue
            delta = metrics["ndcg@20"] - baseline_ndcg
            delta_by_label[label].append(delta)
            row[f"{label}_delta_ndcg20"] = delta
        per_panel.append(row)

    summary = {
        "target_strata": {str(key): count for key, count in sorted(targets.items())},
        "panels": args.panels,
        "seed": args.seed,
        "baseline_label": baseline_label,
        "labels": {},
    }
    for label, metric_rows in metrics_by_label.items():
        summary["labels"][label] = {
            metric: summarize([row[metric] for row in metric_rows])
            for metric in ["ndcg@1", "ndcg@10", "ndcg@20", "hit@20"]
        }
        if label in delta_by_label:
            summary["labels"][label]["delta_ndcg@20_vs_baseline"] = summarize(delta_by_label[label])

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "blind_like_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with open(out_dir / "blind_like_panels.csv", "w", encoding="utf-8", newline="") as f:
            fieldnames = sorted({key for row in per_panel for key in row})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_panel)


if __name__ == "__main__":
    main()
