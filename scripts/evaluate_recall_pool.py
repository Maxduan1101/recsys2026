from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from tqdm import tqdm

from goalflow.data import TrackCatalog
from goalflow.fusion import infer_intent, rrf_fuse
from goalflow.pipeline import (
    GoalFlowConfig,
    default_index_weights,
    default_query_weights,
    prepare_retriever,
    top_k_by_index,
)
from goalflow.state import ConversationState, build_state_for_dev_turn, query_variants


CHECKPOINTS = (1, 5, 10, 20, 50, 100, 200, 300, 500, 800, 1200, 2000)
NDCG_CHECKPOINTS = (1, 10, 20, 100)


@dataclass
class RankStats:
    examples: int = 0
    present: int = 0
    rank_sum: float = 0.0
    reciprocal_rank_sum: float = 0.0
    pool_size_sum: int = 0
    hits: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    ndcg: dict[int, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, rank: int | None, *, present: bool | None = None, pool_size: int = 0) -> None:
        self.examples += 1
        self.pool_size_sum += pool_size
        if rank is None:
            if present:
                self.present += 1
            return
        self.present += 1
        self.rank_sum += rank
        self.reciprocal_rank_sum += 1.0 / rank
        for k in CHECKPOINTS:
            if rank <= k:
                self.hits[k] += 1
        for k in NDCG_CHECKPOINTS:
            if rank <= k:
                self.ndcg[k] += 1.0 / math.log2(rank + 1)

    def to_row(self, group: str, source: str) -> dict[str, object]:
        examples = self.examples or 1
        row: dict[str, object] = {
            "group": group,
            "source": source,
            "examples": self.examples,
            "present": self.present,
            "coverage": self.present / examples,
            "mean_found_rank": self.rank_sum / self.present if self.present and self.rank_sum else None,
            "mrr": self.reciprocal_rank_sum / examples,
            "mean_pool_size": self.pool_size_sum / examples,
        }
        for k in CHECKPOINTS:
            row[f"hit@{k}"] = self.hits[k] / examples
        for k in NDCG_CHECKPOINTS:
            row[f"ndcg@{k}"] = self.ndcg[k] / examples
        return row


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval-stage recall only: candidate-pool coverage, per-source gold ranks, "
            "and RRF-k sensitivity before any LTR reranking."
        )
    )
    parser.add_argument("--tid", default="recall_pool_eval")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--retrieval-top-k", type=int, default=260)
    parser.add_argument("--rrf-k-list", default="20,26,60,100")
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-train-augmentation", action="store_true")
    return parser.parse_args()


def build_dev_states(config: GoalFlowConfig) -> list[ConversationState]:
    dataset = load_dataset(config.conversation_dataset_name, split="test")
    if config.dev_limit:
        dataset = dataset.select(range(min(config.dev_limit, len(dataset))))

    states = []
    for item in tqdm(dataset, desc="Build dev recall states"):
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id:
                states.append(state)
    return states


def source_rank(track_id: str, results: Iterable) -> int | None:
    for result in results:
        if result.track_id == track_id:
            return int(result.rank)
    return None


def group_names(state: ConversationState) -> list[str]:
    names = ["overall", f"turn={state.turn_number}", f"intent={infer_intent(state)}"]
    if state.category:
        names.append(f"category={state.category}")
    if state.specificity:
        names.append(f"specificity={state.specificity}")
    return names


def add_stats(
    stats: dict[tuple[str, str], RankStats],
    state: ConversationState,
    source: str,
    rank: int | None,
    *,
    present: bool | None = None,
    pool_size: int = 0,
) -> None:
    for group in group_names(state):
        stats[(group, source)].add(rank, present=present, pool_size=pool_size)


def sorted_fused_ranks(sources, rrf_k: int) -> dict[str, int]:
    fused = rrf_fuse(sources, rrf_k=rrf_k)
    return {
        candidate.track_id: rank
        for rank, candidate in enumerate(
            sorted(fused.values(), key=lambda item: item.score, reverse=True),
            start=1,
        )
    }


def write_outputs(
    config: GoalFlowConfig,
    rows: list[dict[str, object]],
    misses: list[dict[str, object]],
) -> tuple[Path, Path, Path]:
    out_dir = config.experiments_dir / "recall_pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "recall_pool_summary.json"
    csv_path = out_dir / "recall_pool_summary.csv"
    miss_path = out_dir / "candidate_pool_misses.jsonl"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "group",
        "source",
        "examples",
        "present",
        "coverage",
        "mean_found_rank",
        "mrr",
        "mean_pool_size",
        *[f"hit@{k}" for k in CHECKPOINTS],
        *[f"ndcg@{k}" for k in NDCG_CHECKPOINTS],
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(miss_path, "w", encoding="utf-8") as f:
        for row in misses:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return json_path, csv_path, miss_path


def main() -> None:
    args = parse_args()
    rrf_k_values = parse_int_list(args.rrf_k_list)
    config = GoalFlowConfig(
        project_root=Path(args.project_root),
        tid=args.tid,
        use_train_augmentation=not args.no_train_augmentation,
        rebuild_cache=args.rebuild_cache,
        retrieval_top_k=args.retrieval_top_k,
        dev_limit=args.dev_limit,
    )
    catalog = TrackCatalog(config.track_metadata_name)
    retriever = prepare_retriever(config, catalog)
    states = build_dev_states(config)
    variants = [query_variants(state, catalog) for state in states]
    source_rows = retriever.batch_search(
        query_variants_per_state=variants,
        top_k_by_index=top_k_by_index(config),
        query_weights=default_query_weights(),
        index_weights=default_index_weights(),
    )

    stats: dict[tuple[str, str], RankStats] = defaultdict(RankStats)
    misses: list[dict[str, object]] = []

    for state, sources in tqdm(list(zip(states, source_rows)), desc="Score recall pool"):
        gold = state.gold_track_id
        if not gold:
            continue

        candidate_set = {
            result.track_id
            for _source_name, _index_name, _weight, results in sources
            for result in results
        }
        add_stats(
            stats,
            state,
            "__candidate_union__",
            None,
            present=gold in candidate_set,
            pool_size=len(candidate_set),
        )
        if gold not in candidate_set:
            misses.append(
                {
                    "session_id": state.session_id,
                    "user_id": state.user_id,
                    "turn_number": state.turn_number,
                    "gold_track_id": gold,
                    "intent": infer_intent(state),
                    "category": state.category,
                    "specificity": state.specificity,
                    "current_user_query": state.current_user_query,
                    "conversation_goal": state.listener_goal,
                    "pool_size": len(candidate_set),
                }
            )

        best_source_rank: int | None = None
        best_rank_by_index: dict[str, int] = {}
        best_rank_by_query: dict[str, int] = {}
        seen_indices: set[str] = set()
        seen_queries: set[str] = set()

        for source_name, index_name, _weight, results in sources:
            query_name = source_name.split(":", 1)[1] if ":" in source_name else source_name
            rank = source_rank(gold, results)
            add_stats(stats, state, source_name, rank, pool_size=len(results))
            seen_indices.add(index_name)
            seen_queries.add(query_name)
            if rank is None:
                continue
            best_source_rank = rank if best_source_rank is None else min(best_source_rank, rank)
            best_rank_by_index[index_name] = min(rank, best_rank_by_index.get(index_name, rank))
            best_rank_by_query[query_name] = min(rank, best_rank_by_query.get(query_name, rank))

        add_stats(stats, state, "__best_single_source_rank__", best_source_rank)
        for index_name in seen_indices:
            add_stats(stats, state, f"index_any={index_name}", best_rank_by_index.get(index_name))
        for query_name in seen_queries:
            add_stats(stats, state, f"query_any={query_name}", best_rank_by_query.get(query_name))

        for rrf_k in rrf_k_values:
            rank_by_track = sorted_fused_ranks(sources, rrf_k=rrf_k)
            add_stats(
                stats,
                state,
                f"__rrf_k={rrf_k}__",
                rank_by_track.get(gold),
                pool_size=len(rank_by_track),
            )

    rows = [stat.to_row(group, source) for (group, source), stat in stats.items()]
    rows.sort(key=lambda row: (str(row["group"]), str(row["source"])))
    json_path, csv_path, miss_path = write_outputs(config, rows, misses)

    print(f"states={len(states)}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(f"misses={miss_path}")
    for row in rows:
        if row["group"] == "overall" and row["source"] in {
            "__candidate_union__",
            "__best_single_source_rank__",
            *{f"__rrf_k={value}__" for value in rrf_k_values},
        }:
            print(
                f"{row['source']}: coverage={row['coverage']:.4f} "
                f"hit@20={row['hit@20']:.4f} hit@100={row['hit@100']:.4f} "
                f"hit@300={row['hit@300']:.4f} hit@1200={row['hit@1200']:.4f} "
                f"mean_pool_size={row['mean_pool_size']:.1f}"
            )


if __name__ == "__main__":
    main()
