from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from goalflow.data import CONVERSATION_DATASET, TRACK_METADATA, TrackCatalog, as_text
from goalflow.state import ConversationState, build_state_for_dev_turn, state_text


@dataclass(frozen=True)
class CaseRow:
    case_id: int
    session_id: str
    user_id: str
    turn_number: int
    gold_track_id: str
    text: str


def dcg_at_20(gold_track_id: str | None, ranked_track_ids: list[str]) -> float:
    if not gold_track_id:
        return 0.0
    for rank, track_id in enumerate(ranked_track_ids[:20], start=1):
        if track_id == gold_track_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def clipped_state_text(state: ConversationState, catalog: TrackCatalog, max_words: int = 180) -> str:
    text = state_text(state, catalog)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def build_cases(catalog: TrackCatalog, limit_sessions: int | None = None) -> list[CaseRow]:
    dataset = load_dataset(CONVERSATION_DATASET, split="train")
    if limit_sessions is not None:
        dataset = dataset.select(range(min(limit_sessions, len(dataset))))
    cases: list[CaseRow] = []
    for item in tqdm(dataset, desc="Build train cases"):
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id and catalog.has_track(state.gold_track_id):
                cases.append(
                    CaseRow(
                        case_id=len(cases),
                        session_id=state.session_id,
                        user_id=state.user_id,
                        turn_number=state.turn_number,
                        gold_track_id=state.gold_track_id,
                        text=clipped_state_text(state, catalog),
                    )
                )
    return cases


def build_dev_states(catalog: TrackCatalog, limit_groups: int | None = None) -> list[ConversationState]:
    dataset = load_dataset(CONVERSATION_DATASET, split="test")
    states = []
    for item in dataset:
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id:
                states.append(state)
                if limit_groups is not None and len(states) >= limit_groups:
                    return states
    return states


def doc_to_id(doc) -> int:
    if isinstance(doc, dict):
        return int(doc["id"])
    if hasattr(doc, "item"):
        value = doc.item()
        if isinstance(value, dict):
            return int(value["id"])
        return int(value)
    return int(doc)


def fill_by_case(base_ids: list[str], case_order: list[str], lock_head_k: int) -> list[str]:
    selected = []
    seen = set()
    for track_id in base_ids[:lock_head_k]:
        if track_id not in seen:
            selected.append(track_id)
            seen.add(track_id)
    for track_id in case_order:
        if len(selected) >= 20:
            break
        if track_id not in seen:
            selected.append(track_id)
            seen.add(track_id)
    for track_id in base_ids:
        if len(selected) >= 20:
            break
        if track_id not in seen:
            selected.append(track_id)
            seen.add(track_id)
    return selected[:20]


def summarize_variant(states: list[ConversationState], base: list[list[str]], variant: list[list[str]]) -> dict:
    base_scores = np.asarray([dcg_at_20(state.gold_track_id, ids) for state, ids in zip(states, base)])
    variant_scores = np.asarray([dcg_at_20(state.gold_track_id, ids) for state, ids in zip(states, variant)])
    deltas = variant_scores - base_scores
    return {
        "ndcg@20": float(variant_scores.mean()) if len(variant_scores) else 0.0,
        "base_ndcg@20": float(base_scores.mean()) if len(base_scores) else 0.0,
        "delta": float(deltas.mean()) if len(deltas) else 0.0,
        "better": int((deltas > 1e-12).sum()),
        "worse": int((deltas < -1e-12).sum()),
        "tie": int((np.abs(deltas) <= 1e-12).sum()),
        "changed_rows": int(sum(a != b for a, b in zip(base, variant))),
        "changed_top1": int(sum((a[:1] != b[:1]) for a, b in zip(base, variant))),
        "changed_top5": int(sum((a[:5] != b[:5]) for a, b in zip(base, variant))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe train-turn case neighbors as a protected candidate source.")
    parser.add_argument("--base-prediction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--track-metadata-name", default=TRACK_METADATA)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--limit-train-sessions", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    catalog = TrackCatalog(args.track_metadata_name)
    cases = build_cases(catalog, limit_sessions=args.limit_train_sessions)
    docs = [case.text.lower() for case in cases]
    model = bm25s.BM25()
    model.index(bm25s.tokenize(docs, show_progress=False), show_progress=False)

    states = build_dev_states(catalog, limit_groups=args.limit_groups)
    predictions = json.loads(Path(args.base_prediction).read_text(encoding="utf-8"))[: len(states)]
    base_ids = [row["predicted_track_ids"][:20] for row in predictions]
    queries = [clipped_state_text(state, catalog).lower() for state in states]
    results = model.retrieve(
        bm25s.tokenize(queries, show_progress=False),
        k=args.top_k,
        return_as="tuple",
        show_progress=True,
    )

    case_orders: list[list[str]] = []
    gold_hits = 0
    artist_hits = 0
    for state, row_docs in zip(states, results.documents):
        order = []
        seen = set()
        gold_artist = (
            catalog.normalized_field(state.gold_track_id, "artist_name")
            if state.gold_track_id and catalog.has_track(state.gold_track_id)
            else ""
        )
        has_gold = False
        has_artist = False
        for doc in row_docs:
            case = cases[doc_to_id(doc)]
            track_id = case.gold_track_id
            if track_id not in seen:
                order.append(track_id)
                seen.add(track_id)
            if state.gold_track_id and track_id == state.gold_track_id:
                has_gold = True
            if gold_artist and catalog.normalized_field(track_id, "artist_name") == gold_artist:
                has_artist = True
        gold_hits += int(has_gold)
        artist_hits += int(has_artist)
        case_orders.append(order)

    variants = {
        "case_only": [fill_by_case([], case_order, 0) for case_order in case_orders],
        "lock0": [fill_by_case(base, case_order, 0) for base, case_order in zip(base_ids, case_orders)],
        "lock5": [fill_by_case(base, case_order, 5) for base, case_order in zip(base_ids, case_orders)],
        "lock10": [fill_by_case(base, case_order, 10) for base, case_order in zip(base_ids, case_orders)],
        "lock15": [fill_by_case(base, case_order, 15) for base, case_order in zip(base_ids, case_orders)],
    }
    summary = {
        "top_k": args.top_k,
        "train_cases": len(cases),
        "groups": len(states),
        "base_prediction": args.base_prediction,
        "gold_in_neighbor_tracks_rate": gold_hits / len(states) if states else 0.0,
        "gold_artist_in_neighbor_tracks_rate": artist_hits / len(states) if states else 0.0,
        "variants": {
            name: summarize_variant(states, base_ids, ranked)
            for name, ranked in variants.items()
        },
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"summary={output}")


if __name__ == "__main__":
    main()
