from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from goalflow.data import TrackCatalog
from goalflow.fusion import CandidateScore, rerank_candidates, rerank_candidates_gated
from goalflow.pipeline import default_index_weights, default_query_weights
from goalflow.response import generate_response
from goalflow.state import ConversationState
from goalflow.validation import validate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dev predictions from a source/K choice over exported source candidate matrix."
    )
    parser.add_argument(
        "--matrix-dir",
        default="goalflow_musiccrs/experiments/source_candidate_matrix_full_s20_k800/source_candidate_matrix",
    )
    parser.add_argument("--choice", required=True, help="Path to best_choice.tsv.")
    parser.add_argument("--tid", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rank-mode", choices=["rrf_only", "heuristic", "gated"], default="heuristic")
    parser.add_argument("--protect-head-k", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--response-style", default="judge_clean_mix")
    parser.add_argument("--copy-to-official-evaluator", action="store_true")
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


def read_choice(path: Path, source_name_to_index: dict[str, int]) -> dict[int, int]:
    choice: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            choice[source_name_to_index[row["source_name"]]] = int(row["selected_k"])
    return choice


def source_weight(source_name: str) -> float:
    if ":" not in source_name:
        return 1.0
    index_name, query_name = source_name.split(":", 1)
    return default_index_weights().get(index_name, 1.0) * default_query_weights().get(query_name, 1.0)


def state_from_example(row: dict[str, str]) -> ConversationState:
    conversation_goal = {
        "listener_goal": row.get("conversation_goal", ""),
        "category": row.get("category", ""),
        "specificity": row.get("specificity", ""),
    }
    return ConversationState(
        session_id=row["session_id"],
        user_id=row["user_id"],
        turn_number=int(row["turn_number"]),
        session_date="",
        current_user_query=row.get("current_user_query", ""),
        user_profile={},
        conversation_goal=conversation_goal,
        history_turns=[],
        progress_by_turn={},
        gold_track_id=row.get("gold_track_id"),
    )


def dcg_rank(gold_track_id: str | None, ranked: list[str], k: int) -> float:
    if not gold_track_id:
        return 0.0
    for rank, track_id in enumerate(ranked[:k], start=1):
        if track_id == gold_track_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def macro_ndcg_by_turn(predictions: list[dict], gold_by_key: dict[tuple[str, int], str]) -> dict[str, float]:
    by_turn: dict[int, list[dict[str, float]]] = defaultdict(list)
    for pred in predictions:
        key = (pred["session_id"], int(pred["turn_number"]))
        gold = gold_by_key.get(key)
        ranked = pred["predicted_track_ids"]
        by_turn[int(pred["turn_number"])].append(
            {
                "ndcg@1": dcg_rank(gold, ranked, 1),
                "ndcg@10": dcg_rank(gold, ranked, 10),
                "ndcg@20": dcg_rank(gold, ranked, 20),
            }
        )
    out = {}
    for metric in ["ndcg@1", "ndcg@10", "ndcg@20"]:
        turn_means = [
            sum(item[metric] for item in rows) / len(rows)
            for _turn, rows in sorted(by_turn.items())
            if rows
        ]
        out[metric] = sum(turn_means) / len(turn_means) if turn_means else 0.0
    return out


def source_legacy_order(
    candidates: np.memmap,
    counts: np.memmap,
    turn_index: int,
    source_name_to_index: dict[str, int],
    track_ids: list[str],
) -> list[str]:
    source_index = source_name_to_index.get("legacy_metadata:legacy_history")
    if source_index is None:
        return []
    count = int(counts[turn_index, source_index])
    return [track_ids[int(item)] for item in candidates[turn_index, source_index, :count] if int(item) >= 0]


def build_fused_candidates(
    candidates: np.memmap,
    counts: np.memmap,
    turn_index: int,
    choice: dict[int, int],
    source_names: list[str],
    track_ids: list[str],
    rrf_k: int,
) -> dict[str, CandidateScore]:
    fused: dict[str, CandidateScore] = {}
    for source_index, selected_k in choice.items():
        count = int(counts[turn_index, source_index])
        take = min(selected_k, count)
        if take <= 0:
            continue
        source_name = source_names[source_index]
        weight = source_weight(source_name)
        row = candidates[turn_index, source_index, :take]
        for rank, track_index in enumerate(row, start=1):
            track_index = int(track_index)
            if track_index < 0:
                continue
            track_id = track_ids[track_index]
            item = fused.setdefault(track_id, CandidateScore(track_id=track_id, score=0.0))
            item.score += weight / (rrf_k + rank)
            item.source_ranks[source_name] = rank
    return fused


def main() -> None:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    project_root = Path(args.project_root)
    meta = read_meta(matrix_dir)
    num_turns = int(meta["num_turns"])
    num_sources = int(meta["num_sources"])
    max_k = int(meta["max_k"])
    candidates = np.memmap(matrix_dir / str(meta["candidate_file"]), dtype=np.int32, mode="r", shape=(num_turns, num_sources, max_k))
    counts = np.memmap(matrix_dir / str(meta["counts_file"]), dtype=np.uint16, mode="r", shape=(num_turns, num_sources))

    sources = read_tsv(matrix_dir / "sources.tsv")
    examples = read_tsv(matrix_dir / "examples.tsv")
    tracks = read_tsv(matrix_dir / "track_ids.tsv")
    source_names = [row["source_name"] for row in sources]
    source_name_to_index = {row["source_name"]: int(row["source_index"]) for row in sources}
    track_ids = [row["track_id"] for row in tracks]
    choice = read_choice(Path(args.choice), source_name_to_index)

    catalog = TrackCatalog()
    predictions: list[dict] = []
    global_counts: Counter[str] = Counter()
    gold_by_key: dict[tuple[str, int], str] = {}
    pool_sizes: list[int] = []

    for turn_index, example in enumerate(examples):
        state = state_from_example(example)
        gold_by_key[(state.session_id, state.turn_number)] = example.get("gold_track_id", "")
        fused = build_fused_candidates(
            candidates,
            counts,
            turn_index,
            choice,
            source_names,
            track_ids,
            args.rrf_k,
        )
        pool_sizes.append(len(fused))
        if args.rank_mode == "rrf_only":
            track_ids_ranked = [item.track_id for item in sorted(fused.values(), key=lambda item: item.score, reverse=True)]
        elif args.rank_mode == "gated":
            legacy_order = source_legacy_order(candidates, counts, turn_index, source_name_to_index, track_ids)
            track_ids_ranked = rerank_candidates_gated(
                state,
                catalog,
                fused,
                legacy_order=legacy_order,
                top_k=args.top_k,
                global_counts=global_counts,
                protect_head_k=args.protect_head_k,
            )
        else:
            track_ids_ranked = rerank_candidates(
                state,
                catalog,
                fused,
                top_k=args.top_k,
                global_counts=global_counts,
            )
        track_ids_ranked = list(dict.fromkeys(track_ids_ranked))
        if len(track_ids_ranked) < args.top_k:
            for item in sorted(fused.values(), key=lambda item: item.score, reverse=True):
                if item.track_id not in track_ids_ranked:
                    track_ids_ranked.append(item.track_id)
                if len(track_ids_ranked) >= args.top_k:
                    break
        track_ids_ranked = track_ids_ranked[: args.top_k]
        global_counts.update(track_ids_ranked)
        predictions.append(
            {
                "session_id": state.session_id,
                "user_id": state.user_id,
                "turn_number": state.turn_number,
                "predicted_track_ids": track_ids_ranked,
                "predicted_response": generate_response(state, catalog, track_ids_ranked, style=args.response_style),
            }
        )

    validation = validate_predictions(predictions, catalog, expected_count=len(examples))
    if not validation["ok"]:
        raise ValueError(f"Invalid predictions: {validation}")

    out_dir = project_root / "experiments" / args.tid / "devset"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{args.tid}.json"
    pred_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")

    if args.copy_to_official_evaluator:
        official = project_root / "music-crs-evaluator" / "exp" / "inference" / "devset"
        official.mkdir(parents=True, exist_ok=True)
        (official / f"{args.tid}.json").write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")

    scores = macro_ndcg_by_turn(predictions, gold_by_key)
    summary = {
        "tid": args.tid,
        "choice": args.choice,
        "rank_mode": args.rank_mode,
        "rrf_k": args.rrf_k,
        "protect_head_k": args.protect_head_k,
        "pred_path": str(pred_path),
        "mean_pool_size": sum(pool_sizes) / len(pool_sizes),
        "median_pool_size": float(np.median(pool_sizes)),
        "p95_pool_size": float(np.percentile(pool_sizes, 95)),
        **scores,
    }
    summary_path = out_dir / "source_choice_dev_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
