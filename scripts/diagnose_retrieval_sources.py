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


CHECKPOINTS = (1, 5, 10, 20, 50, 100, 260, 500, 1000)
NDCG_CHECKPOINTS = (1, 10, 20)


@dataclass
class RankStats:
    examples: int = 0
    present: int = 0
    rank_sum: float = 0.0
    reciprocal_rank_sum: float = 0.0
    hits: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    ndcg: dict[int, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, rank: int | None) -> None:
        self.examples += 1
        if rank is None:
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
        row: dict[str, object] = {
            "group": group,
            "source": source,
            "examples": self.examples,
            "present": self.present,
            "coverage": self.present / self.examples if self.examples else 0.0,
            "mean_found_rank": self.rank_sum / self.present if self.present else None,
            "mrr": self.reciprocal_rank_sum / self.examples if self.examples else 0.0,
        }
        for k in CHECKPOINTS:
            row[f"hit@{k}"] = self.hits[k] / self.examples if self.examples else 0.0
        for k in NDCG_CHECKPOINTS:
            row[f"ndcg@{k}"] = self.ndcg[k] / self.examples if self.examples else 0.0
        return row


@dataclass
class FusionDeltaStats:
    examples: int = 0
    gained: int = 0
    lost: int = 0
    demoted: int = 0
    promoted: int = 0
    same_hit: int = 0
    both_miss: int = 0
    legacy_hit20: int = 0
    fused_hit20: int = 0
    dcg_delta_sum: float = 0.0

    @staticmethod
    def dcg(rank: int | None, k: int = 20) -> float:
        if rank is None or rank > k:
            return 0.0
        return 1.0 / math.log2(rank + 1)

    def add(self, legacy_rank: int | None, fused_rank: int | None) -> None:
        self.examples += 1
        legacy_hit = legacy_rank is not None and legacy_rank <= 20
        fused_hit = fused_rank is not None and fused_rank <= 20
        self.legacy_hit20 += int(legacy_hit)
        self.fused_hit20 += int(fused_hit)
        self.dcg_delta_sum += self.dcg(fused_rank) - self.dcg(legacy_rank)
        if not legacy_hit and fused_hit:
            self.gained += 1
        elif legacy_hit and not fused_hit:
            self.lost += 1
        elif legacy_hit and fused_hit:
            if fused_rank > legacy_rank:
                self.demoted += 1
            elif fused_rank < legacy_rank:
                self.promoted += 1
            else:
                self.same_hit += 1
        else:
            self.both_miss += 1

    def to_row(self, group: str) -> dict[str, object]:
        denom = self.examples or 1
        return {
            "group": group,
            "examples": self.examples,
            "legacy_hit@20": self.legacy_hit20 / denom,
            "fused_hit@20": self.fused_hit20 / denom,
            "gained": self.gained,
            "lost": self.lost,
            "demoted": self.demoted,
            "promoted": self.promoted,
            "same_hit": self.same_hit,
            "both_miss": self.both_miss,
            "gained_rate": self.gained / denom,
            "lost_rate": self.lost / denom,
            "demoted_rate": self.demoted / denom,
            "promoted_rate": self.promoted / denom,
            "mean_dcg_delta@20": self.dcg_delta_sum / denom,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose per-source retrieval recall and gold rank for GoalFlow dev states."
    )
    parser.add_argument("--tid", default="goalflow_source_diagnostics")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--retrieval-top-k", type=int, default=260)
    parser.add_argument("--rerank-pool-size", type=int, default=1200)
    parser.add_argument("--legacy-head-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-train-augmentation", action="store_true")
    return parser.parse_args()


def build_dev_states(config: GoalFlowConfig) -> list[ConversationState]:
    dataset = load_dataset(config.conversation_dataset_name, split="test")
    if config.dev_limit:
        dataset = dataset.select(range(min(config.dev_limit, len(dataset))))

    states = []
    for item in tqdm(dataset, desc="Build dev diagnostic states"):
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


def fused_rank(track_id: str, sources, rrf_k: int) -> int | None:
    fused = rrf_fuse(sources, rrf_k=rrf_k)
    for rank, candidate in enumerate(
        sorted(fused.values(), key=lambda item: item.score, reverse=True),
        start=1,
    ):
        if candidate.track_id == track_id:
            return rank
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
) -> None:
    for group in group_names(state):
        stats[(group, source)].add(rank)


def add_delta_stats(
    stats: dict[str, FusionDeltaStats],
    state: ConversationState,
    legacy_rank: int | None,
    fused_rank_value: int | None,
) -> None:
    for group in group_names(state):
        stats[group].add(legacy_rank, fused_rank_value)


def sort_key(row: dict[str, object]) -> tuple[str, float, str]:
    return (
        str(row["group"]),
        -float(row["hit@20"]),
        str(row["source"]),
    )


def write_outputs(
    config: GoalFlowConfig,
    rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
) -> tuple[Path, Path, Path]:
    out_dir = config.experiments_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "retrieval_source_summary.json"
    csv_path = out_dir / "retrieval_source_summary.csv"
    delta_path = out_dir / "legacy_vs_fused_delta_summary.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(delta_path, "w", encoding="utf-8") as f:
        json.dump(delta_rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "group",
        "source",
        "examples",
        "present",
        "coverage",
        "mean_found_rank",
        "mrr",
        *[f"hit@{k}" for k in CHECKPOINTS],
        *[f"ndcg@{k}" for k in NDCG_CHECKPOINTS],
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path, delta_path


def main() -> None:
    args = parse_args()
    config = GoalFlowConfig(
        project_root=Path(args.project_root),
        tid=args.tid,
        use_train_augmentation=not args.no_train_augmentation,
        rebuild_cache=args.rebuild_cache,
        retrieval_top_k=args.retrieval_top_k,
        rerank_pool_size=args.rerank_pool_size,
        legacy_head_k=args.legacy_head_k,
        rrf_k=args.rrf_k,
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
    delta_stats: dict[str, FusionDeltaStats] = defaultdict(FusionDeltaStats)
    for state, sources in tqdm(list(zip(states, source_rows)), desc="Score diagnostic ranks"):
        gold = state.gold_track_id
        if not gold:
            continue
        fused_rank_value = fused_rank(gold, sources, rrf_k=config.rrf_k)
        add_stats(stats, state, "__rrf_fused__", fused_rank_value)
        best_source_rank: int | None = None
        legacy_rank: int | None = None
        best_rank_by_index: dict[str, int] = {}
        seen_indices: set[str] = set()
        for source_name, index_name, _weight, results in sources:
            rank = source_rank(gold, results)
            add_stats(stats, state, source_name, rank)
            if source_name == "legacy_metadata:legacy_history":
                legacy_rank = rank
            seen_indices.add(index_name)
            if rank is not None:
                best_source_rank = rank if best_source_rank is None else min(best_source_rank, rank)
                best_rank_by_index[index_name] = min(rank, best_rank_by_index.get(index_name, rank))
        add_stats(stats, state, "__best_single_source_rank__", best_source_rank)
        add_delta_stats(delta_stats, state, legacy_rank, fused_rank_value)
        for index_name in seen_indices:
            add_stats(stats, state, f"index_any={index_name}", best_rank_by_index.get(index_name))

    rows = [
        stat.to_row(group=group, source=source)
        for (group, source), stat in stats.items()
    ]
    rows.sort(key=sort_key)
    delta_rows = [stat.to_row(group) for group, stat in delta_stats.items()]
    delta_rows.sort(key=lambda row: (str(row["group"])))
    json_path, csv_path, delta_path = write_outputs(config, rows, delta_rows)
    print(f"states={len(states)}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(f"delta_json={delta_path}")
    for row in delta_rows:
        if row["group"] == "overall":
            print(
                "legacy_vs_fused_overall: "
                f"legacy_hit@20={row['legacy_hit@20']:.4f} "
                f"fused_hit@20={row['fused_hit@20']:.4f} "
                f"gained={row['gained']} lost={row['lost']} "
                f"demoted={row['demoted']} promoted={row['promoted']} "
                f"mean_dcg_delta@20={row['mean_dcg_delta@20']:.6f}"
            )
            break
    print("top_overall_by_hit20:")
    for row in [row for row in rows if row["group"] == "overall"][:15]:
        print(
            f"{row['source']} hit@20={row['hit@20']:.4f} "
            f"ndcg@20={row['ndcg@20']:.4f} "
            f"hit@100={row['hit@100']:.4f} coverage={row['coverage']:.4f} "
            f"mean_rank={row['mean_found_rank']}"
        )


if __name__ == "__main__":
    main()
