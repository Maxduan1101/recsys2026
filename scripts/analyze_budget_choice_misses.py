from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

from goalflow.data import TRACK_METADATA, as_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more source/K choices and summarize missed gold-track features."
    )
    parser.add_argument(
        "--matrix-dir",
        default="goalflow_musiccrs/experiments/source_candidate_matrix_full_s20_k800/source_candidate_matrix",
    )
    parser.add_argument(
        "--choice",
        action="append",
        required=True,
        help="Named choice in the form name:path/to/best_choice.tsv. Can be repeated.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--track-metadata-name", default=TRACK_METADATA)
    return parser.parse_args()


def read_meta(matrix_dir: Path) -> dict[str, object]:
    meta: dict[str, object] = {}
    with (matrix_dir / "meta.txt").open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            key, value = parts
            if key in {"num_turns", "num_sources", "num_tracks", "max_k"}:
                meta[key] = int(value)
            elif key == "k_values":
                meta[key] = [int(item) for item in value.split()]
            else:
                meta[key] = value
    return meta


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse_choice(value: str) -> tuple[str, Path]:
    if ":" not in value:
        path = Path(value)
        return path.parent.name, path
    name, path = value.split(":", 1)
    return name, Path(path)


def read_choice(path: Path, source_name_to_index: dict[str, int]) -> dict[int, int]:
    choice: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            source_name = row["source_name"]
            source_index = source_name_to_index[source_name]
            choice[source_index] = int(row["selected_k"])
    return choice


def pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    index = min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))
    return values[index]


def year_bucket(value: object) -> str:
    match = re.search(r"\b(\d{4})\b", as_text(value))
    if not match:
        return "(unknown)"
    year = int(match.group(1))
    if year < 1960:
        return "<1960"
    if year >= 2020:
        return "2020s"
    decade = (year // 10) * 10
    return f"{decade}s"


def popularity_bucket(value: object) -> str:
    try:
        popularity = float(value or 0)
    except Exception:
        popularity = 0.0
    if popularity <= 0:
        return "0"
    if popularity < 10:
        return "1-9"
    if popularity < 25:
        return "10-24"
    if popularity < 50:
        return "25-49"
    if popularity < 75:
        return "50-74"
    return "75+"


def split_tags(value: object) -> list[str]:
    text = as_text(value)
    parts = re.split(r"[,;/|]+", text)
    return [part.strip().lower() for part in parts if part.strip()]


def evaluate_choice(
    choice: dict[int, int],
    candidates: np.memmap,
    counts: np.memmap,
    gold: np.ndarray,
) -> tuple[list[bool], list[int], list[set[int]]]:
    num_turns, _num_sources, _max_k = candidates.shape
    hit = []
    union_sizes = []
    unions: list[set[int]] = []
    for turn_index in range(num_turns):
        union: set[int] = set()
        for source_index, selected_k in choice.items():
            count = int(counts[turn_index, source_index])
            take = min(selected_k, count)
            if take <= 0:
                continue
            row = candidates[turn_index, source_index, :take]
            union.update(int(item) for item in row if int(item) >= 0)
        unions.append(union)
        union_sizes.append(len(union))
        hit.append(int(gold[turn_index]) in union)
    return hit, union_sizes, unions


def best_source_for_turn(
    turn_index: int,
    gold_id: int,
    candidates: np.memmap,
    counts: np.memmap,
    source_names: list[str],
) -> tuple[str, int] | tuple[str, None]:
    best_name = ""
    best_rank: int | None = None
    for source_index, source_name in enumerate(source_names):
        count = int(counts[turn_index, source_index])
        if count <= 0:
            continue
        row = candidates[turn_index, source_index, :count]
        positions = np.where(row == gold_id)[0]
        if len(positions) == 0:
            continue
        rank = int(positions[0]) + 1
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_name = source_name
    return best_name, best_rank


def metadata_rows(track_metadata_name: str) -> dict[str, dict[str, object]]:
    dataset = load_dataset(track_metadata_name, split="all_tracks")
    return {item["track_id"]: item for item in dataset}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_counter_rows(rows: list[dict[str, object]], choice_name: str, field: str, counter: Counter, total: int) -> None:
    for value, count in counter.most_common(40):
        rows.append(
            {
                "choice": choice_name,
                "field": field,
                "value": value,
                "miss_count": count,
                "miss_share": count / total if total else 0.0,
            }
        )


def main() -> None:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    out_dir = Path(args.out_dir) if args.out_dir else matrix_dir / "miss_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_meta(matrix_dir)
    num_turns = int(meta["num_turns"])
    num_sources = int(meta["num_sources"])
    max_k = int(meta["max_k"])
    candidates = np.memmap(matrix_dir / str(meta["candidate_file"]), dtype=np.int32, mode="r", shape=(num_turns, num_sources, max_k))
    counts = np.memmap(matrix_dir / str(meta["counts_file"]), dtype=np.uint16, mode="r", shape=(num_turns, num_sources))
    gold = np.fromfile(matrix_dir / str(meta["gold_file"]), dtype=np.int32)

    examples = read_tsv(matrix_dir / "examples.tsv")
    sources = read_tsv(matrix_dir / "sources.tsv")
    tracks = read_tsv(matrix_dir / "track_ids.tsv")
    source_names = [row["source_name"] for row in sources]
    source_name_to_index = {row["source_name"]: int(row["source_index"]) for row in sources}
    track_index_to_id = {int(row["track_index"]): row["track_id"] for row in tracks}
    track_rows = metadata_rows(args.track_metadata_name)

    choice_specs = [parse_choice(item) for item in args.choice]
    eval_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    recovery_rows: list[dict[str, object]] = []
    hit_by_choice: dict[str, list[bool]] = {}

    for choice_name, choice_path in choice_specs:
        choice = read_choice(choice_path, source_name_to_index)
        hit, union_sizes, _unions = evaluate_choice(choice, candidates, counts, gold)
        hit_by_choice[choice_name] = hit
        misses = [index for index, ok in enumerate(hit) if not ok]

        eval_rows.append(
            {
                "choice": choice_name,
                "hit": sum(hit),
                "coverage": sum(hit) / num_turns,
                "misses": len(misses),
                "avg_union_size": sum(union_sizes) / num_turns,
                "median_union_size": pct(union_sizes, 0.50),
                "p90_union_size": pct(union_sizes, 0.90),
                "p95_union_size": pct(union_sizes, 0.95),
                "max_union_size": max(union_sizes) if union_sizes else 0,
                "choice_path": str(choice_path),
            }
        )

        miss_detail_rows: list[dict[str, object]] = []
        counters: dict[str, Counter] = defaultdict(Counter)
        recovery_counter: Counter = Counter()
        recovery_source_counter: Counter = Counter()
        selected_source_counter: Counter = Counter()

        for turn_index in misses:
            example = examples[turn_index]
            gold_index = int(gold[turn_index])
            gold_track_id = track_index_to_id.get(gold_index, "")
            track = track_rows.get(gold_track_id, {})
            best_source, best_rank = best_source_for_turn(turn_index, gold_index, candidates, counts, source_names)
            if best_rank is None:
                recovery_type = "not_in_any_exported_source_top800"
                selected_k = 0
            else:
                best_source_index = source_name_to_index[best_source]
                selected_k = choice.get(best_source_index, 0)
                if selected_k <= 0:
                    recovery_type = "in_unselected_source_top800"
                elif best_rank > selected_k:
                    recovery_type = "in_selected_source_but_rank_above_selected_k"
                else:
                    recovery_type = "unexpected_miss_check_logic"
                recovery_source_counter[best_source] += 1
                if selected_k > 0:
                    selected_source_counter[best_source] += 1
            recovery_counter[recovery_type] += 1

            tags = split_tags(track.get("tag_list"))
            counters["turn_number"][example["turn_number"]] += 1
            counters["intent"][example["intent"]] += 1
            counters["category"][example["category"]] += 1
            counters["specificity"][example["specificity"]] += 1
            counters["gold_artist_name"][as_text(track.get("artist_name")) or "(unknown)"] += 1
            counters["gold_year_bucket"][year_bucket(track.get("release_date"))] += 1
            counters["gold_popularity_bucket"][popularity_bucket(track.get("popularity"))] += 1
            for tag in tags[:20]:
                counters["gold_tag"][tag] += 1

            miss_detail_rows.append(
                {
                    "turn_index": turn_index,
                    "sample_id": example["sample_id"],
                    "session_id": example["session_id"],
                    "user_id": example["user_id"],
                    "turn_number": example["turn_number"],
                    "intent": example["intent"],
                    "category": example["category"],
                    "specificity": example["specificity"],
                    "union_size": union_sizes[turn_index],
                    "gold_track_id": gold_track_id,
                    "gold_track_name": as_text(track.get("track_name")),
                    "gold_artist_name": as_text(track.get("artist_name")),
                    "gold_album_name": as_text(track.get("album_name")),
                    "gold_release_date": as_text(track.get("release_date")),
                    "gold_popularity": as_text(track.get("popularity")),
                    "gold_tags": as_text(track.get("tag_list")),
                    "best_exported_source": best_source,
                    "best_exported_rank": best_rank if best_rank is not None else "",
                    "selected_k_for_best_source": selected_k,
                    "recovery_type": recovery_type,
                    "current_user_query": example["current_user_query"],
                    "conversation_goal": example["conversation_goal"],
                }
            )

        write_csv(
            out_dir / f"misses_{choice_name}.csv",
            miss_detail_rows,
            [
                "turn_index",
                "sample_id",
                "session_id",
                "user_id",
                "turn_number",
                "intent",
                "category",
                "specificity",
                "union_size",
                "gold_track_id",
                "gold_track_name",
                "gold_artist_name",
                "gold_album_name",
                "gold_release_date",
                "gold_popularity",
                "gold_tags",
                "best_exported_source",
                "best_exported_rank",
                "selected_k_for_best_source",
                "recovery_type",
                "current_user_query",
                "conversation_goal",
            ],
        )

        total_misses = len(misses)
        for field, counter in counters.items():
            add_counter_rows(feature_rows, choice_name, field, counter, total_misses)
        add_counter_rows(feature_rows, choice_name, "recovery_type", recovery_counter, total_misses)
        add_counter_rows(feature_rows, choice_name, "best_exported_source_for_miss", recovery_source_counter, total_misses)
        add_counter_rows(feature_rows, choice_name, "selected_best_source_for_miss", selected_source_counter, total_misses)
        for value, count in recovery_counter.most_common():
            recovery_rows.append(
                {
                    "choice": choice_name,
                    "recovery_type": value,
                    "miss_count": count,
                    "miss_share": count / total_misses if total_misses else 0.0,
                }
            )

    write_csv(
        out_dir / "choice_eval_summary.csv",
        eval_rows,
        [
            "choice",
            "hit",
            "coverage",
            "misses",
            "avg_union_size",
            "median_union_size",
            "p90_union_size",
            "p95_union_size",
            "max_union_size",
            "choice_path",
        ],
    )
    write_csv(out_dir / "miss_feature_summary.csv", feature_rows, ["choice", "field", "value", "miss_count", "miss_share"])
    write_csv(out_dir / "miss_recovery_summary.csv", recovery_rows, ["choice", "recovery_type", "miss_count", "miss_share"])

    if len(choice_specs) >= 2:
        comparison_rows: list[dict[str, object]] = []
        names = [name for name, _path in choice_specs]
        for i, name_a in enumerate(names):
            for name_b in names[i + 1 :]:
                hit_a = hit_by_choice[name_a]
                hit_b = hit_by_choice[name_b]
                rescued = [idx for idx in range(num_turns) if not hit_a[idx] and hit_b[idx]]
                lost = [idx for idx in range(num_turns) if hit_a[idx] and not hit_b[idx]]
                both_miss = [idx for idx in range(num_turns) if not hit_a[idx] and not hit_b[idx]]
                comparison_rows.append(
                    {
                        "from_choice": name_a,
                        "to_choice": name_b,
                        "rescued_by_to": len(rescued),
                        "lost_by_to": len(lost),
                        "both_miss": len(both_miss),
                    }
                )
        write_csv(out_dir / "choice_pair_comparison.csv", comparison_rows, ["from_choice", "to_choice", "rescued_by_to", "lost_by_to", "both_miss"])

    report = out_dir / "miss_analysis_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Budget Choice Miss Analysis\n\n")
        f.write("## Choice Summary\n\n")
        f.write("| choice | hit | coverage | avg_union_size | p95_union_size | misses |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in eval_rows:
            f.write(
                f"| {row['choice']} | {row['hit']} | {float(row['coverage']):.4f} | "
                f"{float(row['avg_union_size']):.3f} | {row['p95_union_size']} | {row['misses']} |\n"
            )
        f.write("\n## Recovery Type\n\n")
        f.write("| choice | recovery_type | miss_count | miss_share |\n")
        f.write("|---|---|---:|---:|\n")
        for row in recovery_rows:
            f.write(
                f"| {row['choice']} | {row['recovery_type']} | {row['miss_count']} | "
                f"{float(row['miss_share']):.4f} |\n"
            )
        f.write("\n更多细节见 `miss_feature_summary.csv` 和 `misses_<choice>.csv`。\n")

    print(f"out_dir={out_dir}")
    for row in eval_rows:
        print(
            f"{row['choice']}: hit={row['hit']} coverage={float(row['coverage']):.4f} "
            f"avg={float(row['avg_union_size']):.1f} p95={row['p95_union_size']} misses={row['misses']}"
        )


if __name__ == "__main__":
    main()
