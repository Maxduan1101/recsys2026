from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from goalflow.data import CONVERSATION_DATASET, TRACK_METADATA, TrackCatalog, as_text
from goalflow.state import ConversationState, build_state_for_dev_turn


def dcg_at_20(gold_track_id: str | None, ranked_track_ids: list[str]) -> float:
    if not gold_track_id:
        return 0.0
    for rank, track_id in enumerate(ranked_track_ids[:20], start=1):
        if track_id == gold_track_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def build_dev_states(dataset_name: str, limit_groups: int | None = None) -> list[ConversationState]:
    dataset = load_dataset(dataset_name, split="test")
    states = []
    for item in dataset:
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id:
                states.append(state)
                if limit_groups is not None and len(states) >= limit_groups:
                    return states
    return states


def clean_field(value: object, max_words: int = 28) -> str:
    text = re.sub(r"\s+", " ", as_text(value)).strip()
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def state_text(state: ConversationState, catalog: TrackCatalog) -> str:
    positives = [
        catalog.compact_summary(track_id)
        for track_id in state.positive_seed_ids[-2:]
        if catalog.has_track(track_id)
    ]
    negatives = [
        catalog.compact_summary(track_id)
        for track_id in state.negative_seed_ids[-2:]
        if catalog.has_track(track_id)
    ]
    parts = [
        f"Current user request: {clean_field(state.current_user_query, 34)}",
        f"Conversation goal: {clean_field(state.conversation_goal, 42)}",
        f"Category: {state.category}; specificity: {state.specificity}; turn: {state.turn_number}",
    ]
    profile = clean_field(state.user_profile, 36)
    if profile:
        parts.append(f"User profile: {profile}")
    if positives:
        parts.append("Previous helpful music: " + " | ".join(positives))
    if negatives:
        parts.append("Previous less helpful music: " + " | ".join(negatives))
    return "\n".join(parts)


def track_text(catalog: TrackCatalog, track_id: str) -> str:
    view = catalog.view(track_id)
    tags = clean_field(view.tag_list, 18)
    parts = [
        f"Title: {clean_field(view.track_name, 16)}",
        f"Artist: {clean_field(view.artist_name, 16)}",
        f"Album: {clean_field(view.album_name, 16)}",
    ]
    if tags:
        parts.append(f"Tags: {tags}")
    if view.release_date:
        parts.append(f"Release date: {clean_field(view.release_date, 8)}")
    return "\n".join(parts)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_pairs(
    *,
    model_name: str,
    query_texts: list[str],
    candidate_texts: list[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    scores = []
    for start in tqdm(range(0, len(query_texts), batch_size), desc="Score CE pairs"):
        end = min(start + batch_size, len(query_texts))
        encoded = tokenizer(
            query_texts[start:end],
            candidate_texts[start:end],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        scores.extend(logits.squeeze(-1).detach().cpu().float().tolist())
    return np.asarray(scores, dtype=np.float32)


def fill_by_ce(base_ids: list[str], ce_order: list[str], lock_head_k: int) -> list[str]:
    selected = []
    seen = set()
    for track_id in base_ids[:lock_head_k]:
        if track_id not in seen:
            selected.append(track_id)
            seen.add(track_id)
    for track_id in ce_order:
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
    parser = argparse.ArgumentParser(description="Zero-shot cross-encoder probe over existing LTR candidates.")
    parser.add_argument("--base-prediction", required=True)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conversation-dataset-name", default=CONVERSATION_DATASET)
    parser.add_argument("--track-metadata-name", default=TRACK_METADATA)
    parser.add_argument("--model-name", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    catalog = TrackCatalog(args.track_metadata_name)
    states = build_dev_states(args.conversation_dataset_name, limit_groups=args.limit_groups)
    predictions = json.loads(Path(args.base_prediction).read_text(encoding="utf-8"))
    predictions = predictions[: len(states)]
    base_ids = [row["predicted_track_ids"][:20] for row in predictions]

    df = pd.read_pickle(args.candidate_cache)
    if args.limit_groups is not None:
        df = df[df["group_id"] < args.limit_groups].copy()
    df = df.groupby("group_id", sort=False).head(args.top_n).copy()
    group_to_tracks = {
        int(group_id): list(group["track_id"])
        for group_id, group in df.groupby("group_id", sort=False)
    }

    query_texts = []
    candidate_texts = []
    pair_keys = []
    state_texts = [state_text(state, catalog) for state in states]
    track_text_cache: dict[str, str] = {}
    for group_id, state in enumerate(states):
        tracks = group_to_tracks.get(group_id, [])
        for track_id in tracks:
            if track_id not in track_text_cache:
                track_text_cache[track_id] = track_text(catalog, track_id)
            query_texts.append(state_texts[group_id])
            candidate_texts.append(track_text_cache[track_id])
            pair_keys.append((group_id, track_id))

    ce_scores = score_pairs(
        model_name=args.model_name,
        query_texts=query_texts,
        candidate_texts=candidate_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    scored_by_group: dict[int, list[tuple[str, float]]] = {}
    for (group_id, track_id), score in zip(pair_keys, ce_scores):
        scored_by_group.setdefault(group_id, []).append((track_id, float(score)))

    ce_orders = []
    for group_id in range(len(states)):
        ordered = sorted(scored_by_group.get(group_id, []), key=lambda item: item[1], reverse=True)
        ce_orders.append([track_id for track_id, _score in ordered])

    variants = {
        "ce_only": [fill_by_ce([], ce_order, 0) for ce_order in ce_orders],
        "lock0": [fill_by_ce(base, ce_order, 0) for base, ce_order in zip(base_ids, ce_orders)],
        "lock5": [fill_by_ce(base, ce_order, 5) for base, ce_order in zip(base_ids, ce_orders)],
        "lock10": [fill_by_ce(base, ce_order, 10) for base, ce_order in zip(base_ids, ce_orders)],
        "lock15": [fill_by_ce(base, ce_order, 15) for base, ce_order in zip(base_ids, ce_orders)],
    }
    summary = {
        "model_name": args.model_name,
        "device": device,
        "top_n": args.top_n,
        "groups": len(states),
        "pairs": len(pair_keys),
        "base_prediction": args.base_prediction,
        "candidate_cache": args.candidate_cache,
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
