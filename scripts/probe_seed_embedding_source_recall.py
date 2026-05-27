from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from goalflow.embeddings import TRACK_EMBEDDING_CHANNELS, TrackEmbeddingStore
from goalflow.pipeline import GoalFlowConfig
from run_rerank_v2 import load_dev_states
from build_nextgen_candidate_pool import channel_allowed, parse_seed_channels, text_flags, weighted_seed_vector


DEFAULT_GROUPS_CSV = "goalflow_musiccrs/experiments/rerank_v2_independent_features/missed_gold_deep_dive_groups.csv"
DEFAULT_OUT_DIR = "goalflow_musiccrs/experiments/seed_embedding_source_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe official seed-neighbor embedding sources on union-missed gold turns.")
    parser.add_argument("--project-root", default="goalflow_musiccrs")
    parser.add_argument("--groups-csv", default=DEFAULT_GROUPS_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--channels", default="track_cf,attributes,audio,lyrics,image")
    parser.add_argument("--ks", default="20,50,100,180,200,400")
    return parser.parse_args()


def read_missed_groups(path: Path) -> dict[int, dict[str, Any]]:
    groups: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["bucket"] != "missed_union":
                continue
            group_id = int(row["group_id"])
            all20 = int(row["all20_top800_hit"])
            groups[group_id] = {
                "miss_type": "recoverable_in_some_source_top800" if all20 else "absent_from_all_20_sources_top800",
                "category": row.get("category", ""),
                "specificity": row.get("specificity", ""),
                "gold_track_id": row.get("gold_track_id", ""),
                "positive_seed_count": int(row.get("num_positive_seeds") or 0),
                "negative_seed_count": int(row.get("num_negative_seeds") or 0),
            }
    return groups


def rank_gold_for_state(state, channel: str, store: TrackEmbeddingStore) -> int | None:
    if not state.gold_track_id or state.gold_track_id not in store.track_index:
        return None
    flags = text_flags(state)
    if not channel_allowed(channel, flags, state):
        return None
    pos, neg = weighted_seed_vector(state, store, channel)
    if pos is None:
        return None
    matrix = store.matrices[channel]
    gold_index = store.track_index[state.gold_track_id]
    if not matrix.valid[gold_index]:
        return None
    scores = matrix.normalized @ pos
    if neg is not None:
        scores = scores - 0.5 * (matrix.normalized @ neg)
    scores = scores.astype(np.float32)
    scores[~matrix.valid] = -np.inf
    for track_id in set(state.previous_music_track_ids) | set(state.negative_seed_ids):
        index = store.track_index.get(track_id)
        if index is not None:
            scores[index] = -np.inf
    gold_score = scores[gold_index]
    if not np.isfinite(gold_score):
        return None
    return int(np.sum(scores > gold_score) + 1)


def summarize_ranks(channel: str, rows: list[dict[str, Any]], ks: list[int]) -> list[dict[str, Any]]:
    out = []
    slices = {"all_missed": rows}
    for miss_type in ["recoverable_in_some_source_top800", "absent_from_all_20_sources_top800"]:
        slices[miss_type] = [row for row in rows if row["miss_type"] == miss_type]
    for name, subset in slices.items():
        ranks = [int(row["rank"]) for row in subset if row["rank"]]
        record = {
            "channel": channel,
            "slice": name,
            "missed_turns": len(subset),
            "active_with_rank": len(ranks),
            "rank_p50": float(np.percentile(ranks, 50)) if ranks else "",
            "rank_p90": float(np.percentile(ranks, 90)) if ranks else "",
        }
        for k in ks:
            record[f"hit_at_{k}"] = sum(1 for rank in ranks if rank <= k)
        out.append(record)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = [int(item) for item in args.ks.split(",") if item.strip()]
    channels = parse_seed_channels(args.channels)
    missed = read_missed_groups(Path(args.groups_csv))
    config = GoalFlowConfig(project_root=Path(args.project_root), tid="seed_embedding_source_probe")
    states = load_dev_states(config)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for channel in channels:
        store = TrackEmbeddingStore(channels={channel: TRACK_EMBEDDING_CHANNELS[channel]})
        channel_rows: list[dict[str, Any]] = []
        for group_id, meta in tqdm(missed.items(), desc=f"Probe {channel}"):
            state = states[group_id]
            rank = rank_gold_for_state(state, channel, store)
            row = {
                "group_id": group_id,
                "channel": channel,
                "rank": rank or "",
                **meta,
            }
            channel_rows.append(row)
            detail_rows.append(row)
        summary_rows.extend(summarize_ranks(channel, channel_rows, ks))

    detail_path = out_dir / "seed_embedding_source_probe_detail.csv"
    summary_path = out_dir / "seed_embedding_source_probe_summary.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (out_dir / "params.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
