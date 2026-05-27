from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from goalflow.data import TrackCatalog
from goalflow.fusion import infer_intent
from goalflow.pipeline import (
    GoalFlowConfig,
    default_index_weights,
    default_query_weights,
    prepare_retriever,
)
from goalflow.state import ConversationState, build_state_for_dev_turn, query_variants


DEFAULT_KS = (20, 50, 100, 200, 300, 500, 800)


@dataclass(frozen=True)
class ExampleMeta:
    index: int
    session_id: str
    user_id: str
    turn_number: int
    gold_track_id: str
    intent: str
    category: str
    specificity: str
    current_user_query: str
    conversation_goal: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure hit overlap between retrieval sources and search source/topK budgets "
            "for a fixed coarse-recall budget."
        )
    )
    parser.add_argument("--tid", default="source_overlap_budget_top800")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--max-source-k", type=int, default=800)
    parser.add_argument("--ks", default="20,50,100,200,300,500,800")
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--beam-size", type=int, default=40)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-train-augmentation", action="store_true")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return sorted({int(part.strip()) for part in value.split(",") if part.strip()})


def build_dev_states(config: GoalFlowConfig) -> list[ConversationState]:
    dataset = load_dataset(config.conversation_dataset_name, split="test")
    if config.dev_limit:
        dataset = dataset.select(range(min(config.dev_limit, len(dataset))))

    states = []
    for item in tqdm(dataset, desc="Build dev overlap states"):
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id:
                states.append(state)
    return states


def gold_rank(track_id: str, results) -> int | None:
    for result in results:
        if result.track_id == track_id:
            return int(result.rank)
    return None


def bitset_from_hits(hits: set[int]) -> int:
    bits = 0
    for index in hits:
        bits |= 1 << index
    return bits


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def source_hit_rows(
    sources: list[str],
    ks: list[int],
    hit_sets: dict[tuple[str, int], set[int]],
    example_count: int,
) -> list[dict[str, object]]:
    rows = []
    for source in sources:
        previous_hits = 0
        previous_k = 0
        for k in ks:
            hits = len(hit_sets[(source, k)])
            delta_hits = hits - previous_hits
            delta_k = k - previous_k
            rows.append(
                {
                    "source": source,
                    "k": k,
                    "hits": hits,
                    "coverage": hits / example_count if example_count else 0.0,
                    "delta_hits_from_previous_k": delta_hits,
                    "delta_k": delta_k,
                    "delta_hits_per_100_k": (100.0 * delta_hits / delta_k) if delta_k else 0.0,
                }
            )
            previous_hits = hits
            previous_k = k
    return rows


def pair_overlap_rows(
    sources: list[str],
    ks: list[int],
    hit_sets: dict[tuple[str, int], set[int]],
    example_count: int,
) -> list[dict[str, object]]:
    rows = []
    for k in ks:
        for i, source_a in enumerate(sources):
            hits_a = hit_sets[(source_a, k)]
            if not hits_a:
                continue
            for source_b in sources[i + 1 :]:
                hits_b = hit_sets[(source_b, k)]
                if not hits_b:
                    continue
                inter = hits_a & hits_b
                union = hits_a | hits_b
                rows.append(
                    {
                        "k": k,
                        "source_a": source_a,
                        "source_b": source_b,
                        "hits_a": len(hits_a),
                        "hits_b": len(hits_b),
                        "intersection_hits": len(inter),
                        "union_hits": len(union),
                        "union_coverage": len(union) / example_count if example_count else 0.0,
                        "jaccard": len(inter) / len(union) if union else 0.0,
                        "overlap_coef": len(inter) / min(len(hits_a), len(hits_b)),
                        "a_covered_by_b": len(inter) / len(hits_a),
                        "b_covered_by_a": len(inter) / len(hits_b),
                        "exclusive_a": len(hits_a - hits_b),
                        "exclusive_b": len(hits_b - hits_a),
                    }
                )
    rows.sort(key=lambda row: (int(row["k"]), -float(row["jaccard"]), -int(row["intersection_hits"])))
    return rows


def stable_pair_rows(pair_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in pair_rows:
        grouped[(str(row["source_a"]), str(row["source_b"]))].append(row)

    rows = []
    for (source_a, source_b), group in grouped.items():
        jaccards = [float(row["jaccard"]) for row in group]
        overlaps = [float(row["overlap_coef"]) for row in group]
        rows.append(
            {
                "source_a": source_a,
                "source_b": source_b,
                "ks_seen": ",".join(str(row["k"]) for row in group),
                "avg_jaccard": sum(jaccards) / len(jaccards),
                "max_jaccard": max(jaccards),
                "avg_overlap_coef": sum(overlaps) / len(overlaps),
                "max_overlap_coef": max(overlaps),
                "high_jaccard_k_count": sum(value >= 0.75 for value in jaccards),
                "high_overlap_k_count": sum(value >= 0.90 for value in overlaps),
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["high_overlap_k_count"]),
            -float(row["avg_overlap_coef"]),
            -float(row["avg_jaccard"]),
        )
    )
    return rows


def greedy_budget(
    sources: list[str],
    ks: list[int],
    hit_sets: dict[tuple[str, int], set[int]],
    budget: int,
    example_count: int,
) -> tuple[list[dict[str, object]], dict[str, int], set[int]]:
    selected_k = {source: 0 for source in sources}
    covered: set[int] = set()
    budget_used = 0
    steps = []
    step = 0

    while budget_used < budget:
        best: tuple[float, int, int, str, int, set[int], set[int]] | None = None
        for source in sources:
            current_k = selected_k[source]
            next_levels = [k for k in ks if k > current_k and budget_used + (k - current_k) <= budget]
            if not next_levels:
                continue
            # Consider every feasible upgrade, not only the next checkpoint.
            for next_k in next_levels:
                new_hits = hit_sets[(source, next_k)] - covered
                delta_k = next_k - current_k
                gain = len(new_hits)
                score = gain / delta_k if delta_k else 0.0
                candidate = (score, gain, -delta_k, source, next_k, new_hits, hit_sets[(source, next_k)])
                if best is None or candidate > best:
                    best = candidate
        if best is None or best[1] <= 0:
            break

        score, gain, neg_delta_k, source, next_k, new_hits, source_hits = best
        old_k = selected_k[source]
        delta_k = -neg_delta_k
        already_covered = len(source_hits & covered)
        step += 1
        budget_used += delta_k
        selected_k[source] = next_k
        covered |= source_hits
        steps.append(
            {
                "step": step,
                "source": source,
                "from_k": old_k,
                "to_k": next_k,
                "delta_k": delta_k,
                "budget_used": budget_used,
                "new_hits": gain,
                "covered_hits": len(covered),
                "covered_rate": len(covered) / example_count if example_count else 0.0,
                "new_hits_per_100_k": 100.0 * score,
                "source_hits_at_to_k": len(source_hits),
                "source_hits_already_covered_before_step": already_covered,
                "source_overlap_before_step": already_covered / len(source_hits) if source_hits else 0.0,
            }
        )

    return steps, selected_k, covered


def beam_budget(
    sources: list[str],
    ks: list[int],
    hit_bits: dict[tuple[str, int], int],
    budget: int,
    beam_size: int,
) -> tuple[dict[str, int], int, int]:
    # One choice per source: 0, 20, 50, ... K. This is an approximate beam
    # search because exact set coverage with a budget is combinatorial.
    states_by_budget: dict[int, list[tuple[int, dict[str, int]]]] = {0: [(0, {})]}
    for source in tqdm(sources, desc="Beam search source budget"):
        next_by_budget: dict[int, dict[int, tuple[int, dict[str, int]]]] = defaultdict(dict)
        options = [0, *ks]
        for used_budget, states in states_by_budget.items():
            for bits, choices in states:
                for k in options:
                    new_budget = used_budget + k
                    if new_budget > budget:
                        continue
                    new_bits = bits | (hit_bits[(source, k)] if k else 0)
                    new_choices = choices if k == 0 else {**choices, source: k}
                    bucket = next_by_budget[new_budget]
                    old = bucket.get(new_bits)
                    if old is None or len(new_choices) < len(old[1]):
                        bucket[new_bits] = (new_bits, new_choices)

        states_by_budget = {}
        for used_budget, by_bits in next_by_budget.items():
            candidates = sorted(
                by_bits.values(),
                key=lambda item: (item[0].bit_count(), -len(item[1])),
                reverse=True,
            )[:beam_size]
            states_by_budget[used_budget] = candidates

    best_bits = 0
    best_budget = 0
    best_choices: dict[str, int] = {}
    for used_budget, states in states_by_budget.items():
        for bits, choices in states:
            candidate = (bits.bit_count(), -used_budget, -len(choices))
            best = (best_bits.bit_count(), -best_budget, -len(best_choices))
            if candidate > best:
                best_bits = bits
                best_budget = used_budget
                best_choices = choices
    return best_choices, best_bits, best_budget


def selection_rows(
    choices: dict[str, int],
    hit_sets: dict[tuple[str, int], set[int]],
    covered: set[int],
    example_count: int,
) -> list[dict[str, object]]:
    rows = []
    for source, k in sorted(choices.items(), key=lambda item: (-len(hit_sets[(item[0], item[1])]), item[0])):
        hits = hit_sets[(source, k)]
        rows.append(
            {
                "source": source,
                "selected_k": k,
                "source_hits": len(hits),
                "source_coverage": len(hits) / example_count if example_count else 0.0,
                "hits_inside_final_union": len(hits & covered),
                "hits_outside_final_union": len(hits - covered),
            }
        )
    return rows


def write_remaining_misses(path: Path, examples: list[ExampleMeta], covered: set[int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in examples:
            if item.index in covered:
                continue
            f.write(
                json.dumps(
                    {
                        "session_id": item.session_id,
                        "user_id": item.user_id,
                        "turn_number": item.turn_number,
                        "gold_track_id": item.gold_track_id,
                        "intent": item.intent,
                        "category": item.category,
                        "specificity": item.specificity,
                        "current_user_query": item.current_user_query,
                        "conversation_goal": item.conversation_goal,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def count_field(examples: list[ExampleMeta], missed: set[int], field: str) -> list[dict[str, object]]:
    counter = Counter(getattr(examples[index], field) or "(empty)" for index in missed)
    return [{"field": field, "value": value, "miss_count": count} for value, count in counter.most_common()]


def write_report(
    path: Path,
    example_count: int,
    budget: int,
    source_rows: list[dict[str, object]],
    stable_rows: list[dict[str, object]],
    greedy_steps: list[dict[str, object]],
    beam_choices: dict[str, int],
    beam_bits: int,
    beam_budget_used: int,
    beam_selection: list[dict[str, object]],
    missed_counts: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Source Overlap And Budget Report\n\n")
        f.write("这份报告回答三个问题：\n\n")
        f.write("1. 每个 source 在不同 topK 下能命中多少 gold。\n")
        f.write("2. source 两两之间命中的 gold 是否高度重复。\n")
        f.write(f"3. 在 sum(K) <= {budget} 的粗排预算下，哪些 source/K 组合能覆盖最多 gold。\n\n")
        f.write("注意：这里的 hit 只表示 gold 被放进候选池，不表示最终排序已经排对。\n\n")

        f.write("## 单 source 前 15\n\n")
        f.write("| source | k | hits | coverage | delta_hits | delta_hits_per_100_k |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        best_source_rows = sorted(source_rows, key=lambda row: (int(row["hits"]), float(row["delta_hits_per_100_k"])), reverse=True)
        for row in best_source_rows[:15]:
            f.write(
                f"| {row['source']} | {row['k']} | {row['hits']} | "
                f"{float(row['coverage']):.4f} | {row['delta_hits_from_previous_k']} | "
                f"{float(row['delta_hits_per_100_k']):.2f} |\n"
            )

        f.write("\n## 长期高度重复的 source pair 前 15\n\n")
        f.write("| source_a | source_b | avg_overlap_coef | avg_jaccard | high_overlap_k_count |\n")
        f.write("|---|---|---:|---:|---:|\n")
        for row in stable_rows[:15]:
            f.write(
                f"| {row['source_a']} | {row['source_b']} | "
                f"{float(row['avg_overlap_coef']):.4f} | {float(row['avg_jaccard']):.4f} | "
                f"{row['high_overlap_k_count']} |\n"
            )

        if greedy_steps:
            final = greedy_steps[-1]
            f.write("\n## 贪心 sumK 组合\n\n")
            f.write(
                f"贪心最终 budget_used={final['budget_used']}，covered_hits={final['covered_hits']}/{example_count}，"
                f"coverage={float(final['covered_rate']):.4f}。\n\n"
            )
            f.write("| step | source | from_k | to_k | new_hits | budget_used | coverage |\n")
            f.write("|---:|---|---:|---:|---:|---:|---:|\n")
            for row in greedy_steps[:25]:
                f.write(
                    f"| {row['step']} | {row['source']} | {row['from_k']} | {row['to_k']} | "
                    f"{row['new_hits']} | {row['budget_used']} | {float(row['covered_rate']):.4f} |\n"
                )

        f.write("\n## Beam Search sumK 组合\n\n")
        f.write(
            f"Beam search 最终 budget_used={beam_budget_used}，covered_hits={beam_bits.bit_count()}/{example_count}，"
            f"coverage={beam_bits.bit_count() / example_count if example_count else 0.0:.4f}。\n\n"
        )
        f.write("| source | selected_k | source_hits | source_coverage |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in beam_selection:
            f.write(
                f"| {row['source']} | {row['selected_k']} | {row['source_hits']} | "
                f"{float(row['source_coverage']):.4f} |\n"
            )

        f.write("\n## Beam 没覆盖的样本分布\n\n")
        f.write("| field | value | miss_count |\n")
        f.write("|---|---|---:|\n")
        for row in missed_counts[:40]:
            f.write(f"| {row['field']} | {row['value']} | {row['miss_count']} |\n")


def main() -> None:
    args = parse_args()
    ks = [k for k in parse_int_list(args.ks) if 0 < k <= args.max_source_k]
    if not ks:
        raise ValueError("No K values left after applying --max-source-k.")

    config = GoalFlowConfig(
        project_root=Path(args.project_root),
        tid=args.tid,
        use_train_augmentation=not args.no_train_augmentation,
        rebuild_cache=args.rebuild_cache,
        retrieval_top_k=args.max_source_k,
        dev_limit=args.dev_limit,
    )
    out_dir = Path(args.out_dir) if args.out_dir else config.experiments_dir / "source_overlap_budget"
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = TrackCatalog(config.track_metadata_name)
    retriever = prepare_retriever(config, catalog)
    states = build_dev_states(config)
    examples = [
        ExampleMeta(
            index=index,
            session_id=state.session_id,
            user_id=state.user_id,
            turn_number=state.turn_number,
            gold_track_id=state.gold_track_id or "",
            intent=infer_intent(state),
            category=state.category,
            specificity=state.specificity,
            current_user_query=state.current_user_query,
            conversation_goal=state.listener_goal,
        )
        for index, state in enumerate(states)
    ]
    variants = [query_variants(state, catalog) for state in states]
    top_k_by_index = {index_name: args.max_source_k for index_name in retriever.indices}
    source_rows_raw = retriever.batch_search(
        query_variants_per_state=variants,
        top_k_by_index=top_k_by_index,
        query_weights=default_query_weights(),
        index_weights=default_index_weights(),
    )

    source_ranks: dict[str, dict[int, int]] = defaultdict(dict)
    all_sources: set[str] = set()
    for example, state, sources in tqdm(list(zip(examples, states, source_rows_raw)), desc="Collect source gold ranks"):
        gold = state.gold_track_id
        if not gold:
            continue
        for source_name, _index_name, _weight, results in sources:
            all_sources.add(source_name)
            rank = gold_rank(gold, results)
            if rank is not None:
                source_ranks[source_name][example.index] = rank

    sources = sorted(all_sources)
    hit_sets: dict[tuple[str, int], set[int]] = {}
    hit_bits: dict[tuple[str, int], int] = {}
    for source in sources:
        hit_sets[(source, 0)] = set()
        hit_bits[(source, 0)] = 0
        ranks = source_ranks.get(source, {})
        for k in ks:
            hits = {index for index, rank in ranks.items() if rank <= k}
            hit_sets[(source, k)] = hits
            hit_bits[(source, k)] = bitset_from_hits(hits)

    src_rows = source_hit_rows(sources, ks, hit_sets, len(examples))
    pair_rows = pair_overlap_rows(sources, ks, hit_sets, len(examples))
    stable_rows = stable_pair_rows(pair_rows)
    greedy_steps, greedy_choices, greedy_covered = greedy_budget(sources, ks, hit_sets, args.budget, len(examples))

    ordered_sources = sorted(sources, key=lambda source: len(hit_sets[(source, max(ks))]), reverse=True)
    beam_choices, beam_bits, beam_budget_used = beam_budget(
        ordered_sources,
        ks,
        hit_bits,
        args.budget,
        args.beam_size,
    )
    beam_covered = {index for index in range(len(examples)) if (beam_bits >> index) & 1}
    beam_selection = selection_rows(beam_choices, hit_sets, beam_covered, len(examples))
    missed = set(range(len(examples))) - beam_covered
    missed_counts = (
        count_field(examples, missed, "intent")
        + count_field(examples, missed, "category")
        + count_field(examples, missed, "specificity")
        + count_field(examples, missed, "turn_number")
    )

    write_csv(
        out_dir / "source_hit_by_k.csv",
        src_rows,
        [
            "source",
            "k",
            "hits",
            "coverage",
            "delta_hits_from_previous_k",
            "delta_k",
            "delta_hits_per_100_k",
        ],
    )
    write_csv(
        out_dir / "source_pair_overlap_by_k.csv",
        pair_rows,
        [
            "k",
            "source_a",
            "source_b",
            "hits_a",
            "hits_b",
            "intersection_hits",
            "union_hits",
            "union_coverage",
            "jaccard",
            "overlap_coef",
            "a_covered_by_b",
            "b_covered_by_a",
            "exclusive_a",
            "exclusive_b",
        ],
    )
    write_csv(
        out_dir / "source_pair_stable_overlap.csv",
        stable_rows,
        [
            "source_a",
            "source_b",
            "ks_seen",
            "avg_jaccard",
            "max_jaccard",
            "avg_overlap_coef",
            "max_overlap_coef",
            "high_jaccard_k_count",
            "high_overlap_k_count",
        ],
    )
    write_csv(
        out_dir / "greedy_budget_steps.csv",
        greedy_steps,
        [
            "step",
            "source",
            "from_k",
            "to_k",
            "delta_k",
            "budget_used",
            "new_hits",
            "covered_hits",
            "covered_rate",
            "new_hits_per_100_k",
            "source_hits_at_to_k",
            "source_hits_already_covered_before_step",
            "source_overlap_before_step",
        ],
    )
    write_csv(
        out_dir / "greedy_budget_selection.csv",
        selection_rows({source: k for source, k in greedy_choices.items() if k}, hit_sets, greedy_covered, len(examples)),
        ["source", "selected_k", "source_hits", "source_coverage", "hits_inside_final_union", "hits_outside_final_union"],
    )
    write_csv(
        out_dir / "beam_budget_selection.csv",
        beam_selection,
        ["source", "selected_k", "source_hits", "source_coverage", "hits_inside_final_union", "hits_outside_final_union"],
    )
    write_remaining_misses(out_dir / "beam_budget_misses.jsonl", examples, beam_covered)
    write_report(
        out_dir / "source_overlap_budget_report.md",
        len(examples),
        args.budget,
        src_rows,
        stable_rows,
        greedy_steps,
        beam_choices,
        beam_bits,
        beam_budget_used,
        beam_selection,
        missed_counts,
    )

    print(f"examples={len(examples)} sources={len(sources)}")
    print(f"out_dir={out_dir}")
    if greedy_steps:
        final = greedy_steps[-1]
        print(
            f"greedy: budget={final['budget_used']} hits={final['covered_hits']} "
            f"coverage={float(final['covered_rate']):.4f}"
        )
    print(
        f"beam: budget={beam_budget_used} hits={beam_bits.bit_count()} "
        f"coverage={beam_bits.bit_count() / len(examples) if examples else 0.0:.4f}"
    )


if __name__ == "__main__":
    main()
