from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

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


CAPS = (300, 500, 1000)
NDCG_CHECKPOINTS = (20, 100)


@dataclass
class RankStats:
    examples: int = 0
    present: int = 0
    rank_sum: float = 0.0
    reciprocal_rank_sum: float = 0.0
    pool_size_sum: int = 0
    ndcg: dict[int, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, rank: int | None, pool_size: int) -> None:
        self.examples += 1
        self.pool_size_sum += pool_size
        if rank is None:
            return
        self.present += 1
        self.rank_sum += rank
        self.reciprocal_rank_sum += 1.0 / rank
        for k in NDCG_CHECKPOINTS:
            if rank <= k:
                self.ndcg[k] += 1.0 / math.log2(rank + 1)

    def to_row(self, strategy: str, cap: int) -> dict[str, object]:
        examples = self.examples or 1
        return {
            "strategy": strategy,
            "cap": cap,
            "examples": self.examples,
            "present": self.present,
            "coverage": self.present / examples,
            "mean_rank": self.rank_sum / self.present if self.present else None,
            "mrr": self.reciprocal_rank_sum / examples,
            "mean_pool_size": self.pool_size_sum / examples,
            "ndcg@20": self.ndcg[20] / examples,
            "ndcg@100": self.ndcg[100] / examples,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate capped candidate-pool coverage before LTR reranking.")
    parser.add_argument("--tid", default="capped_recall_pool_eval")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--retrieval-top-k", type=int, default=500)
    parser.add_argument("--caps", default="300,500,1000")
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-train-augmentation", action="store_true")
    return parser.parse_args()


def parse_caps(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_dev_states(config: GoalFlowConfig) -> list[ConversationState]:
    dataset = load_dataset(config.conversation_dataset_name, split="test")
    if config.dev_limit:
        dataset = dataset.select(range(min(config.dev_limit, len(dataset))))

    states = []
    for item in tqdm(dataset, desc="Build dev capped states"):
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id:
                states.append(state)
    return states


def rank_in_order(track_id: str, order: list[str]) -> int | None:
    try:
        return order.index(track_id) + 1
    except ValueError:
        return None


def source_priority(state: ConversationState, source_name: str, index_name: str, source_weight: float) -> float:
    intent = infer_intent(state)
    query_name = source_name.split(":", 1)[1] if ":" in source_name else source_name
    priority = source_weight

    if source_name == "legacy_metadata:legacy_history":
        priority += 3.0
    if query_name == "quoted_entities":
        priority += 2.2
    if query_name == "seed_current" and state.positive_seed_ids:
        priority += 1.2
    if query_name in {"current_goal", "legacy_history"}:
        priority += 0.5

    if intent == "specific_track":
        if index_name == "title_artist":
            priority += 2.4
        if index_name == "legacy_metadata":
            priority += 1.4
        if index_name == "album_artist":
            priority += 0.6
        if index_name == "tags":
            priority -= 0.7
    elif intent == "album":
        if index_name == "album_artist":
            priority += 2.4
        if index_name == "title_artist":
            priority += 1.1
    elif intent == "artist_exploration":
        if index_name in {"title_artist", "metadata_all", "enriched"}:
            priority += 1.4
        if query_name == "seed_current":
            priority += 0.8
    elif intent in {"mood_playlist", "lyrics_theme"}:
        if index_name in {"tags", "enriched", "metadata_all"}:
            priority += 1.6
        if query_name in {"goal", "current_goal"}:
            priority += 0.8
    elif intent == "cover_art":
        if index_name in {"metadata_all", "enriched"}:
            priority += 1.0

    return max(priority, 0.05)


def dedupe_append(order: list[str], seen: set[str], track_id: str, cap: int) -> None:
    if len(order) >= cap or track_id in seen:
        return
    seen.add(track_id)
    order.append(track_id)


def rrf_order(sources, cap: int) -> list[str]:
    fused = rrf_fuse(sources, rrf_k=26)
    return [
        candidate.track_id
        for candidate in sorted(fused.values(), key=lambda item: item.score, reverse=True)[:cap]
    ]


def priority_concat_order(state: ConversationState, sources, cap: int) -> list[str]:
    scored_sources = sorted(
        sources,
        key=lambda item: source_priority(state, item[0], item[1], item[2]),
        reverse=True,
    )
    order: list[str] = []
    seen: set[str] = set()
    for _source_name, _index_name, _weight, results in scored_sources:
        for result in results:
            dedupe_append(order, seen, result.track_id, cap)
            if len(order) >= cap:
                return order
    return order


def priority_round_robin_order(state: ConversationState, sources, cap: int) -> list[str]:
    scored_sources = sorted(
        [(source_priority(state, source_name, index_name, weight), results) for source_name, index_name, weight, results in sources],
        key=lambda item: item[0],
        reverse=True,
    )
    order: list[str] = []
    seen: set[str] = set()
    cursors = [0 for _priority, _results in scored_sources]
    while len(order) < cap:
        changed = False
        for source_idx, (_priority, results) in enumerate(scored_sources):
            cursor = cursors[source_idx]
            while cursor < len(results):
                track_id = results[cursor].track_id
                cursor += 1
                if track_id not in seen:
                    dedupe_append(order, seen, track_id, cap)
                    changed = True
                    break
            cursors[source_idx] = cursor
            if len(order) >= cap:
                break
        if not changed:
            break
    return order


def quota_fill_order(state: ConversationState, sources, cap: int) -> list[str]:
    scored_sources = [
        (source_priority(state, source_name, index_name, weight), results)
        for source_name, index_name, weight, results in sources
    ]
    total_priority = sum(priority for priority, _results in scored_sources) or 1.0
    scored_sources.sort(key=lambda item: item[0], reverse=True)

    order: list[str] = []
    seen: set[str] = set()
    for priority, results in scored_sources:
        quota = max(2, round(cap * priority / total_priority))
        taken = 0
        for result in results:
            before = len(order)
            dedupe_append(order, seen, result.track_id, cap)
            if len(order) > before:
                taken += 1
            if taken >= quota or len(order) >= cap:
                break
        if len(order) >= cap:
            return order

    for track_id in rrf_order(sources, cap):
        dedupe_append(order, seen, track_id, cap)
        if len(order) >= cap:
            break
    return order


def full_union_order(sources, cap: int) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for _source_name, _index_name, _weight, results in sources:
        for result in results:
            dedupe_append(order, seen, result.track_id, cap)
            if len(order) >= cap:
                return order
    return order


def write_outputs(config: GoalFlowConfig, rows: list[dict[str, object]]) -> Path:
    out_dir = config.experiments_dir / "capped_recall_pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "capped_recall_pool_summary.csv"
    json_path = out_dir / "capped_recall_pool_summary.json"
    fieldnames = [
        "strategy",
        "cap",
        "examples",
        "present",
        "coverage",
        "mean_rank",
        "mrr",
        "mean_pool_size",
        "ndcg@20",
        "ndcg@100",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return csv_path


def main() -> None:
    args = parse_args()
    caps = parse_caps(args.caps) or CAPS
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

    stats: dict[tuple[str, int], RankStats] = defaultdict(RankStats)
    strategies = {
        "rrf_k26": rrf_order,
        "source_priority_concat": priority_concat_order,
        "source_priority_round_robin": priority_round_robin_order,
        "source_quota_fill": quota_fill_order,
        "source_generation_order": full_union_order,
    }

    for state, sources in tqdm(list(zip(states, source_rows)), desc="Evaluate capped pools"):
        gold = state.gold_track_id
        if not gold:
            continue
        for cap in caps:
            for strategy_name, builder in strategies.items():
                if strategy_name == "rrf_k26" or strategy_name == "source_generation_order":
                    order = builder(sources, cap)
                else:
                    order = builder(state, sources, cap)
                stats[(strategy_name, cap)].add(rank_in_order(gold, order), pool_size=len(order))

    rows = [stat.to_row(strategy, cap) for (strategy, cap), stat in stats.items()]
    rows.sort(key=lambda row: (int(row["cap"]), str(row["strategy"])))
    csv_path = write_outputs(config, rows)
    print(f"states={len(states)}")
    print(f"csv={csv_path}")
    for row in rows:
        print(
            f"cap={row['cap']} {row['strategy']}: coverage={row['coverage']:.4f} "
            f"mean_rank={row['mean_rank']} ndcg@20={row['ndcg@20']:.4f}"
        )


if __name__ == "__main__":
    main()

