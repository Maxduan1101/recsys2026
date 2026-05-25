from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from goalflow.data import CONVERSATION_DATASET, TrackCatalog, as_text
from goalflow.state import progress_map, role_at_turn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export readable samples and counts for goal_progress_assessments semantics."
    )
    parser.add_argument("--dataset-name", default=CONVERSATION_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-name", default="progress_label_audit")
    parser.add_argument("--sample-sessions", type=int, default=20)
    parser.add_argument("--max-text-chars", type=int, default=420)
    return parser.parse_args()


def truncate(text: str, max_chars: int) -> str:
    text = " ".join(as_text(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def track_summary(catalog: TrackCatalog, track_id: str, max_chars: int) -> str:
    if not catalog.has_track(track_id):
        return track_id
    return truncate(catalog.compact_summary(track_id), max_chars)


def build_turn_record(item: dict, turn_number: int, catalog: TrackCatalog, max_chars: int) -> dict:
    conversations = item.get("conversations", [])
    progress = progress_map(item.get("goal_progress_assessments", []))
    user_turn = role_at_turn(conversations, turn_number, "user") or {}
    music_turn = role_at_turn(conversations, turn_number, "music") or {}
    assistant_turn = role_at_turn(conversations, turn_number, "assistant") or {}
    next_user_turn = role_at_turn(conversations, turn_number + 1, "user") or {}
    track_id = as_text(music_turn.get("content"))
    return {
        "session_id": item.get("session_id"),
        "user_id": item.get("user_id"),
        "turn_number": turn_number,
        "label_at_turn": progress.get(turn_number),
        "current_user": truncate(user_turn.get("content"), max_chars),
        "music_track_id": track_id,
        "music_track": track_summary(catalog, track_id, max_chars),
        "music_thought": truncate(music_turn.get("thought"), max_chars),
        "assistant_response": truncate(assistant_turn.get("content"), max_chars),
        "next_user": truncate(next_user_turn.get("content"), max_chars),
    }


def markdown_report(summary: dict, records: list[dict]) -> str:
    lines = [
        "# Progress Label Audit",
        "",
        "This report is meant to verify whether `goal_progress_assessments[turn_number]` describes the same turn's music recommendation or the transition into the next turn.",
        "",
        "## Summary",
        "",
        f"- Dataset: `{summary['dataset_name']}`",
        f"- Split: `{summary['split']}`",
        f"- Sessions scanned: `{summary['sessions_scanned']}`",
        f"- Turn records scanned: `{summary['turn_records_scanned']}`",
        f"- Missing labels: `{summary['missing_labels']}`",
        "",
        "Label counts:",
        "",
    ]
    for label, count in summary["label_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(["", "Turn-label counts:", ""])
    for turn, counts in summary["label_counts_by_turn"].items():
        parts = ", ".join(f"{label}={count}" for label, count in counts.items())
        lines.append(f"- turn `{turn}`: {parts}")

    lines.extend(["", "## Samples", ""])
    for record in records:
        lines.extend(
            [
                f"### {record['session_id']} turn {record['turn_number']} label `{record['label_at_turn']}`",
                "",
                f"- Current user: {record['current_user']}",
                f"- Music: `{record['music_track_id']}` {record['music_track']}",
                f"- Music thought: {record['music_thought']}",
                f"- Assistant: {record['assistant_response']}",
                f"- Next user: {record['next_user']}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_name, split=args.split)
    catalog = TrackCatalog()
    label_counts: Counter[str] = Counter()
    label_counts_by_turn: dict[int, Counter[str]] = defaultdict(Counter)
    missing_labels = 0
    turn_records_scanned = 0
    sample_records = []

    for session_index, item in enumerate(tqdm(dataset, desc="Audit progress labels")):
        progress = progress_map(item.get("goal_progress_assessments", []))
        for turn_number in range(1, 9):
            label = progress.get(turn_number)
            turn_records_scanned += 1
            if label is None:
                missing_labels += 1
                label = "<missing>"
            label_counts[label] += 1
            label_counts_by_turn[turn_number][label] += 1
            if session_index < args.sample_sessions:
                sample_records.append(
                    build_turn_record(item, turn_number, catalog, max_chars=args.max_text_chars)
                )

    summary = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "sessions_scanned": len(dataset),
        "turn_records_scanned": turn_records_scanned,
        "missing_labels": missing_labels,
        "label_counts": dict(label_counts),
        "label_counts_by_turn": {
            str(turn): dict(counts) for turn, counts in sorted(label_counts_by_turn.items())
        },
    }
    out_dir = Path(args.project_root) / "research" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.split}_summary.json"
    records_path = out_dir / f"{args.split}_samples.json"
    markdown_path = out_dir / f"{args.split}_audit.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(sample_records, f, ensure_ascii=False, indent=2)
    markdown_path.write_text(markdown_report(summary, sample_records), encoding="utf-8")
    print(f"summary={summary_path}")
    print(f"samples={records_path}")
    print(f"markdown={markdown_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
