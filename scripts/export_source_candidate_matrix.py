from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from goalflow.data import BLIND_A_DATASET, TrackCatalog
from goalflow.fusion import infer_intent
from goalflow.pipeline import GoalFlowConfig, prepare_retriever
from goalflow.state import build_state_for_blind_item, build_state_for_dev_turn, query_variants


DEFAULT_K_VALUES = "0,50,100,200,400,800"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-turn/per-source candidate track lists for true union-size source-K search. "
            "The exported binary matrix is consumed by cpp/source_budget_beam.cpp."
        )
    )
    parser.add_argument("--tid", default="source_candidate_matrix_top800")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--mode", choices=["dev", "blind"], default="dev")
    parser.add_argument("--blind-dataset-name", default=BLIND_A_DATASET)
    parser.add_argument("--k-values", default=DEFAULT_K_VALUES)
    parser.add_argument("--max-k", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-train-augmentation", action="store_true")
    parser.add_argument(
        "--include-blank-query-results",
        action="store_true",
        help="Mimic the old batch_search behavior for absent query variants. Off by default.",
    )
    parser.add_argument(
        "--source-limit",
        type=int,
        default=0,
        help="Keep only the top N sources by an existing source_topk_curves.csv. 0 means all sources.",
    )
    parser.add_argument(
        "--source-curve-csv",
        default=(
            "goalflow_musiccrs/experiments/recall_pool_top1000/recall_pool/"
            "source_topk_curves/source_topk_curves.csv"
        ),
    )
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    out = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not out or out[0] != 0:
        out = [0, *out]
    return out


def safe_tsv(value: object) -> str:
    return re.sub(r"[\t\r\n]+", " ", "" if value is None else str(value)).strip()


def build_dev_states(config: GoalFlowConfig):
    dataset = load_dataset(config.conversation_dataset_name, split="test")
    if config.dev_limit:
        dataset = dataset.select(range(min(config.dev_limit, len(dataset))))

    states = []
    for item in tqdm(dataset, desc="Build dev matrix states"):
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id:
                states.append(state)
    return states


def build_blind_states(config: GoalFlowConfig):
    dataset = load_dataset(config.blind_dataset_name, split="test")
    if config.dev_limit:
        dataset = dataset.select(range(min(config.dev_limit, len(dataset))))
    return [build_state_for_blind_item(item) for item in tqdm(dataset, desc="Build blind matrix states")]


def source_priority_from_curve(path: Path) -> list[str]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row.get("coverage") or 0.0),
            float(row.get("hit@100") or 0.0),
            float(row.get("hit@500") or 0.0),
        ),
        reverse=True,
    )
    return [row["source"] for row in rows if row.get("source")]


def choose_sources(
    index_names: list[str],
    variant_names: list[str],
    *,
    source_limit: int,
    curve_csv: Path,
) -> list[tuple[str, str, str]]:
    all_sources = [(f"{index_name}:{variant_name}", index_name, variant_name) for index_name in index_names for variant_name in variant_names]
    if source_limit <= 0:
        return all_sources

    by_name = {name: (name, index_name, variant_name) for name, index_name, variant_name in all_sources}
    priority = [name for name in source_priority_from_curve(curve_csv) if name in by_name]
    selected = [by_name[name] for name in priority[:source_limit]]
    if len(selected) < source_limit:
        already = {name for name, _index_name, _variant_name in selected}
        for source in all_sources:
            if source[0] not in already:
                selected.append(source)
                already.add(source[0])
            if len(selected) >= source_limit:
                break
    return selected


def write_meta(
    out_dir: Path,
    *,
    num_turns: int,
    num_sources: int,
    num_tracks: int,
    max_k: int,
    k_values: list[int],
) -> None:
    with (out_dir / "meta.txt").open("w", encoding="utf-8") as f:
        f.write(f"num_turns\t{num_turns}\n")
        f.write(f"num_sources\t{num_sources}\n")
        f.write(f"num_tracks\t{num_tracks}\n")
        f.write(f"max_k\t{max_k}\n")
        f.write("k_values\t" + " ".join(str(k) for k in k_values) + "\n")
        f.write("candidate_file\tcandidates.i32\n")
        f.write("counts_file\tcounts.u16\n")
        f.write("gold_file\tgold.i32\n")
        f.write("sources_file\tsources.tsv\n")
        f.write("examples_file\texamples.tsv\n")
        f.write("track_ids_file\ttrack_ids.tsv\n")


def write_sources(out_dir: Path, sources: list[tuple[str, str, str]]) -> None:
    with (out_dir / "sources.tsv").open("w", encoding="utf-8") as f:
        f.write("source_index\tsource_name\tindex_name\tquery_variant\n")
        for i, (source_name, index_name, variant_name) in enumerate(sources):
            f.write(f"{i}\t{source_name}\t{index_name}\t{variant_name}\n")


def write_tracks(out_dir: Path, track_ids: list[str]) -> None:
    with (out_dir / "track_ids.tsv").open("w", encoding="utf-8") as f:
        f.write("track_index\ttrack_id\n")
        for i, track_id in enumerate(track_ids):
            f.write(f"{i}\t{track_id}\n")


def write_examples(out_dir: Path, states, catalog: TrackCatalog, track_to_int: dict[str, int]) -> np.ndarray:
    gold = np.full((len(states),), -1, dtype=np.int32)
    with (out_dir / "examples.tsv").open("w", encoding="utf-8") as f:
        f.write(
            "turn_index\tsample_id\tsession_id\tuser_id\tturn_number\tgold_track_id\tgold_track_index\t"
            "intent\tcategory\tspecificity\tcurrent_user_query\tconversation_goal\n"
        )
        for i, state in enumerate(states):
            gold_id = state.gold_track_id or ""
            gold_index = track_to_int.get(gold_id, -1)
            gold[i] = gold_index
            sample_id = f"{state.session_id}::{state.user_id}::{state.turn_number}"
            f.write(
                "\t".join(
                    [
                        str(i),
                        safe_tsv(sample_id),
                        safe_tsv(state.session_id),
                        safe_tsv(state.user_id),
                        str(state.turn_number),
                        safe_tsv(gold_id),
                        str(gold_index),
                        safe_tsv(infer_intent(state)),
                        safe_tsv(state.category),
                        safe_tsv(state.specificity),
                        safe_tsv(state.current_user_query),
                        safe_tsv(state.listener_goal),
                    ]
                )
                + "\n"
            )
    gold.tofile(out_dir / "gold.i32")
    return gold


def main() -> None:
    args = parse_args()
    k_values = parse_int_list(args.k_values)
    max_k = args.max_k or max(k_values)
    if max(k_values) > max_k:
        raise ValueError(f"max(k_values)={max(k_values)} exceeds --max-k={max_k}")

    config = GoalFlowConfig(
        project_root=Path(args.project_root),
        tid=args.tid,
        blind_dataset_name=args.blind_dataset_name,
        use_train_augmentation=not args.no_train_augmentation,
        rebuild_cache=args.rebuild_cache,
        retrieval_top_k=max_k,
        dev_limit=args.dev_limit,
    )
    out_dir = Path(args.out_dir) if args.out_dir else config.experiments_dir / "source_candidate_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = TrackCatalog(config.track_metadata_name)
    track_to_int = {track_id: index for index, track_id in enumerate(catalog.track_ids)}
    retriever = prepare_retriever(config, catalog)
    states = build_dev_states(config) if args.mode == "dev" else build_blind_states(config)
    variants_by_state = [query_variants(state, catalog) for state in states]
    variant_names = sorted({name for variants in variants_by_state for name in variants})
    index_names = sorted(retriever.indices)
    sources = choose_sources(
        index_names,
        variant_names,
        source_limit=args.source_limit,
        curve_csv=Path(args.source_curve_csv),
    )

    write_meta(
        out_dir,
        num_turns=len(states),
        num_sources=len(sources),
        num_tracks=len(catalog.track_ids),
        max_k=max_k,
        k_values=k_values,
    )
    write_sources(out_dir, sources)
    write_tracks(out_dir, catalog.track_ids)
    gold = write_examples(out_dir, states, catalog, track_to_int)

    candidates = np.memmap(out_dir / "candidates.i32", dtype=np.int32, mode="w+", shape=(len(states), len(sources), max_k))
    counts = np.memmap(out_dir / "counts.u16", dtype=np.uint16, mode="w+", shape=(len(states), len(sources)))
    nonzero_k_values = [k for k in k_values if k > 0]
    hit_matrix = np.zeros((len(nonzero_k_values), len(states), len(sources)), dtype=np.uint8)
    candidates[:] = -1
    counts[:] = 0
    candidates.flush()
    counts.flush()

    stats_rows: list[dict[str, object]] = []
    for source_index, (source_name, index_name, variant_name) in enumerate(tqdm(sources, desc="Export source candidates")):
        index = retriever.indices[index_name]
        if args.include_blank_query_results:
            query_indices = list(range(len(states)))
            queries = [variants.get(variant_name, "") for variants in variants_by_state]
        else:
            query_indices = [i for i, variants in enumerate(variants_by_state) if variants.get(variant_name, "").strip()]
            queries = [variants_by_state[i][variant_name] for i in query_indices]

        source_hits = {k: 0 for k in k_values if k > 0}
        if queries:
            rows = index.search_many(queries, top_k=max_k)
            for state_index, row in zip(query_indices, rows):
                track_indices = [track_to_int[result.track_id] for result in row[:max_k]]
                count = len(track_indices)
                if count:
                    candidates[state_index, source_index, :count] = np.array(track_indices, dtype=np.int32)
                    counts[state_index, source_index] = count
                    gold_index = int(gold[state_index])
                    for k_pos, k in enumerate(nonzero_k_values):
                        if gold_index >= 0 and gold_index in track_indices[: min(k, count)]:
                            source_hits[k] += 1
                            hit_matrix[k_pos, state_index, source_index] = 1

        for k in nonzero_k_values:
            stats_rows.append(
                {
                    "source_index": source_index,
                    "source": source_name,
                    "k": k,
                    "hit_turns": source_hits[k],
                    "hit_rate": source_hits[k] / len(states) if states else 0.0,
                }
            )
        candidates.flush()
        counts.flush()

    with (out_dir / "source_single_hit_stats.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_index", "source", "k", "hit_turns", "hit_rate"])
        writer.writeheader()
        writer.writerows(stats_rows)

    pair_rows = []
    for k_pos, k in enumerate(nonzero_k_values):
        hits_at_k = hit_matrix[k_pos]
        source_hit_counts = hits_at_k.sum(axis=0)
        for source_a in range(len(sources)):
            hits_a = int(source_hit_counts[source_a])
            if hits_a == 0:
                continue
            col_a = hits_at_k[:, source_a]
            for source_b in range(source_a + 1, len(sources)):
                hits_b = int(source_hit_counts[source_b])
                if hits_b == 0:
                    continue
                col_b = hits_at_k[:, source_b]
                intersection = int(np.logical_and(col_a, col_b).sum())
                union = int(np.logical_or(col_a, col_b).sum())
                pair_rows.append(
                    {
                        "k": k,
                        "source_a_index": source_a,
                        "source_a": sources[source_a][0],
                        "source_b_index": source_b,
                        "source_b": sources[source_b][0],
                        "hits_a": hits_a,
                        "hits_b": hits_b,
                        "intersection_hits": intersection,
                        "union_hits": union,
                        "hit_jaccard": intersection / union if union else 0.0,
                        "overlap_coef": intersection / min(hits_a, hits_b),
                        "a_covered_by_b": intersection / hits_a,
                        "b_covered_by_a": intersection / hits_b,
                        "exclusive_a": hits_a - intersection,
                        "exclusive_b": hits_b - intersection,
                    }
                )
    pair_rows.sort(key=lambda row: (row["k"], -row["overlap_coef"], -row["hit_jaccard"], -row["intersection_hits"]))
    with (out_dir / "source_pair_hit_overlap.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "k",
                "source_a_index",
                "source_a",
                "source_b_index",
                "source_b",
                "hits_a",
                "hits_b",
                "intersection_hits",
                "union_hits",
                "hit_jaccard",
                "overlap_coef",
                "a_covered_by_b",
                "b_covered_by_a",
                "exclusive_a",
                "exclusive_b",
            ],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    with (out_dir / "export_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_turns": len(states),
                "num_sources": len(sources),
                "num_tracks": len(catalog.track_ids),
                "max_k": max_k,
                "k_values": k_values,
                "include_blank_query_results": args.include_blank_query_results,
                "source_limit": args.source_limit,
                "mode": args.mode,
                "out_dir": str(out_dir),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"out_dir={out_dir}")
    print(f"turns={len(states)} sources={len(sources)} tracks={len(catalog.track_ids)} max_k={max_k}")
    print(f"candidate_matrix={(out_dir / 'candidates.i32').stat().st_size / (1024 * 1024):.1f} MiB")


if __name__ == "__main__":
    main()
