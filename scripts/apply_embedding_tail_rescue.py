from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from goalflow.data import BLIND_A_DATASET, CONVERSATION_DATASET, TRACK_METADATA, TrackCatalog
from goalflow.embeddings import TrackEmbeddingStore, UserEmbeddingStore
from goalflow.fusion import infer_intent
from goalflow.response import generate_response
from goalflow.state import ConversationState, build_state_for_blind_item, build_state_for_dev_turn
from goalflow.validation import validate_predictions


def load_predictions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


def dev_state_map(dataset_name: str) -> dict[tuple[str, int], ConversationState]:
    dataset = load_dataset(dataset_name, split="test")
    states = {}
    for item in dataset:
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            states[(state.session_id, state.turn_number)] = state
    return states


def blind_state_map(dataset_name: str) -> dict[tuple[str, int], ConversationState]:
    dataset = load_dataset(dataset_name, split="test")
    states = {}
    for item in dataset:
        state = build_state_for_blind_item(item)
        states[(state.session_id, state.turn_number)] = state
    return states


def top_seed_cf(
    track_store: TrackEmbeddingStore,
    seed_id: str,
    top_k: int,
    cache: dict[str, list[tuple[str, float, int]]],
) -> list[tuple[str, float, int]]:
    if seed_id in cache:
        return cache[seed_id]
    if not track_store.has_track(seed_id):
        cache[seed_id] = []
        return []
    matrix = track_store.matrices["track_cf"]
    seed_index = track_store.track_index[seed_id]
    if not matrix.valid[seed_index]:
        cache[seed_id] = []
        return []
    scores = matrix.normalized @ matrix.normalized[seed_index]
    scores = scores.astype(np.float32)
    scores[~matrix.valid] = -np.inf
    scores[seed_index] = -np.inf
    cache[seed_id] = track_store.topk_from_scores(scores, top_k=top_k)
    return cache[seed_id]


def top_user_cf(
    user_store: UserEmbeddingStore,
    track_store: TrackEmbeddingStore,
    user_id: str,
    top_k: int,
    cache: dict[str, list[tuple[str, float, int]]],
) -> list[tuple[str, float, int]]:
    if user_id in cache:
        return cache[user_id]
    if not user_store.has_user(user_id):
        cache[user_id] = []
        return []
    scores = user_store.user_track_scores(user_id, track_store)
    cache[user_id] = track_store.topk_from_scores(scores, top_k=top_k)
    return cache[user_id]


def rescue_candidate(
    state: ConversationState,
    original: list[str],
    track_store: TrackEmbeddingStore,
    user_store: UserEmbeddingStore | None,
    seed_cache: dict[str, list[tuple[str, float, int]]],
    user_cache: dict[str, list[tuple[str, float, int]]],
    protected: set[str],
    top_k: int,
    min_score: float,
    use_user_cf: bool,
) -> str | None:
    intent = infer_intent(state)
    if intent in {"specific_track", "album", "cover_art", "lyrics_theme"}:
        return None

    score_by_track: Counter[str] = Counter()
    negative = set(state.negative_seed_ids[-4:])
    seen = set(original)
    if state.positive_seed_ids:
        seed_id = state.positive_seed_ids[-1]
        for track_id, _score, rank in top_seed_cf(track_store, seed_id, top_k=top_k, cache=seed_cache):
            if track_id in protected or track_id in negative or track_id in seen:
                continue
            score_by_track[track_id] += 1.0 / (20.0 + rank)

    if use_user_cf and user_store is not None and intent == "mood_playlist":
        for track_id, _score, rank in top_user_cf(
            user_store,
            track_store,
            state.user_id,
            top_k=top_k,
            cache=user_cache,
        ):
            if track_id in protected or track_id in negative or track_id in seen:
                continue
            score_by_track[track_id] += 0.45 / (30.0 + rank)

    if not score_by_track:
        return None
    track_id, score = score_by_track.most_common(1)[0]
    if score < min_score:
        return None
    return track_id


def apply_tail_rescue(
    predictions: list[dict],
    states: dict[tuple[str, int], ConversationState],
    catalog: TrackCatalog,
    track_store: TrackEmbeddingStore,
    user_store: UserEmbeddingStore | None,
    preserve_head_k: int,
    top_k: int,
    min_score: float,
    use_user_cf: bool,
    response_style: str,
) -> tuple[list[dict], dict[str, int]]:
    out = []
    stats = Counter()
    seed_cache: dict[str, list[tuple[str, float, int]]] = {}
    user_cache: dict[str, list[tuple[str, float, int]]] = {}

    for row in tqdm(predictions, desc="Apply embedding tail rescue"):
        key = (row["session_id"], int(row["turn_number"]))
        state = states[key]
        track_ids = list(row["predicted_track_ids"])
        preserve = max(0, min(preserve_head_k, len(track_ids)))
        protected = set(track_ids[:preserve])
        candidate = rescue_candidate(
            state=state,
            original=track_ids,
            track_store=track_store,
            user_store=user_store,
            seed_cache=seed_cache,
            user_cache=user_cache,
            protected=protected,
            top_k=top_k,
            min_score=min_score,
            use_user_cf=use_user_cf,
        )
        if candidate and catalog.has_track(candidate):
            tail = [track_id for track_id in track_ids[preserve:] if track_id != candidate]
            track_ids = track_ids[:preserve] + [candidate] + tail
            deduped = []
            seen = set()
            for track_id in track_ids:
                if track_id in seen:
                    continue
                deduped.append(track_id)
                seen.add(track_id)
            for track_id in row["predicted_track_ids"]:
                if len(deduped) >= 20:
                    break
                if track_id not in seen:
                    deduped.append(track_id)
                    seen.add(track_id)
            track_ids = deduped[:20]
            stats["rescued_rows"] += 1
        else:
            stats["unchanged_rows"] += 1

        new_row = dict(row)
        new_row["predicted_track_ids"] = track_ids
        new_row["predicted_response"] = generate_response(state, catalog, track_ids, style=response_style)
        out.append(new_row)
    stats["seed_cache"] = len(seed_cache)
    stats["user_cache"] = len(user_cache)
    return out, dict(stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Insert at most one CF-embedding candidate into the protected tail.")
    parser.add_argument("--mode", choices=["dev", "blind"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--tid", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--conversation-dataset-name", default=CONVERSATION_DATASET)
    parser.add_argument("--blind-dataset-name", default=BLIND_A_DATASET)
    parser.add_argument("--track-metadata-name", default=TRACK_METADATA)
    parser.add_argument("--preserve-head-k", type=int, default=19)
    parser.add_argument("--embedding-top-k", type=int, default=80)
    parser.add_argument("--min-score", type=float, default=0.02)
    parser.add_argument("--use-user-cf", action="store_true")
    parser.add_argument("--response-style", choices=["compact", "compact_broad", "concise", "setwise", "natural", "polished", "judge_v1", "judge_v2", "judge_v3", "judge_mix"], default="compact_broad")
    parser.add_argument("--copy-to-official-evaluator", action="store_true")
    parser.add_argument("--zip", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    catalog = TrackCatalog(args.track_metadata_name)
    predictions = load_predictions(Path(args.input))
    states = (
        dev_state_map(args.conversation_dataset_name)
        if args.mode == "dev"
        else blind_state_map(args.blind_dataset_name)
    )
    track_store = TrackEmbeddingStore(channels={"track_cf": "cf-bpr"})
    user_store = UserEmbeddingStore() if args.use_user_cf else None
    output, stats = apply_tail_rescue(
        predictions=predictions,
        states=states,
        catalog=catalog,
        track_store=track_store,
        user_store=user_store,
        preserve_head_k=args.preserve_head_k,
        top_k=args.embedding_top_k,
        min_score=args.min_score,
        use_user_cf=args.use_user_cf,
        response_style=args.response_style,
    )
    validation = validate_predictions(output, catalog, expected_count=len(predictions))
    if not validation["ok"]:
        raise ValueError(f"Invalid predictions: {validation}")

    if args.mode == "dev":
        out_path = project_root / "experiments" / args.tid / "devset" / f"{args.tid}.json"
        write_predictions(out_path, output)
        if args.copy_to_official_evaluator:
            official = project_root.parent / "music-crs-evaluator" / "exp" / "inference" / "devset"
            official.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out_path, official / f"{args.tid}.json")
    else:
        out_path = project_root / "experiments" / args.tid / "blindset_A" / "prediction.json"
        write_predictions(out_path, output)
        if args.zip:
            zip_path = out_path.parent / "submission.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(out_path, arcname="prediction.json")
            print(f"zip={zip_path}")

    stats_path = out_path.parent / "embedding_tail_rescue_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"output={out_path}")
    print(f"stats={stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
