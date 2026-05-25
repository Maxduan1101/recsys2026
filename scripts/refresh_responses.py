from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from datasets import load_dataset

from goalflow.data import BLIND_A_DATASET, CONVERSATION_DATASET, TRACK_METADATA, TrackCatalog
from goalflow.response import generate_response
from goalflow.state import build_state_for_blind_item, build_state_for_dev_turn
from goalflow.validation import validate_predictions


def _load_predictions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_predictions(path: Path, predictions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)


def _dev_state_map(dataset_name: str) -> dict[tuple[str, int], object]:
    dataset = load_dataset(dataset_name, split="test")
    states = {}
    for item in dataset:
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            states[(state.session_id, state.turn_number)] = state
    return states


def _blind_state_map(dataset_name: str) -> dict[tuple[str, int], object]:
    dataset = load_dataset(dataset_name, split="test")
    states = {}
    for item in dataset:
        state = build_state_for_blind_item(item)
        states[(state.session_id, state.turn_number)] = state
    return states


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate responses for an existing prediction file.")
    parser.add_argument("--mode", choices=["dev", "blind"], required=True)
    parser.add_argument("--input", required=True, help="Existing prediction JSON.")
    parser.add_argument("--tid", required=True, help="New experiment id.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--conversation-dataset-name", default=CONVERSATION_DATASET)
    parser.add_argument("--blind-dataset-name", default=BLIND_A_DATASET)
    parser.add_argument("--track-metadata-name", default=TRACK_METADATA)
    parser.add_argument(
        "--response-style",
        choices=[
            "compact", "compact_broad", "concise", "setwise", "natural", "polished",
            "judge_v1", "judge_v2", "judge_v3", "judge_mix", "judge_brief",
            "judge_planned", "judge_compact_mix", "judge_clean_mix", "judge_balanced_mix",
        ],
        default="compact",
    )
    parser.add_argument("--copy-to-official-evaluator", action="store_true")
    parser.add_argument("--zip", action="store_true", help="Write blindset_A/submission.zip.")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    input_path = Path(args.input)
    catalog = TrackCatalog(args.track_metadata_name)
    predictions = _load_predictions(input_path)
    states = (
        _dev_state_map(args.conversation_dataset_name)
        if args.mode == "dev"
        else _blind_state_map(args.blind_dataset_name)
    )

    missing = []
    for row in predictions:
        key = (row["session_id"], int(row["turn_number"]))
        state = states.get(key)
        if state is None:
            missing.append(key)
            continue
        row["predicted_response"] = generate_response(
            state,
            catalog,
            row["predicted_track_ids"],
            style=args.response_style,
        )
    if missing:
        raise ValueError(f"Missing state rows for {len(missing)} predictions; first={missing[:3]}")

    validation = validate_predictions(predictions, catalog, expected_count=len(predictions))
    if not validation["ok"]:
        raise ValueError(f"Invalid refreshed predictions: {validation}")

    if args.mode == "dev":
        output = project_root / "experiments" / args.tid / "devset" / f"{args.tid}.json"
        _write_predictions(output, predictions)
        if args.copy_to_official_evaluator:
            official = project_root.parent / "music-crs-evaluator" / "exp" / "inference" / "devset"
            official.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output, official / f"{args.tid}.json")
    else:
        output = project_root / "experiments" / args.tid / "blindset_A" / "prediction.json"
        _write_predictions(output, predictions)
        if args.zip:
            zip_path = output.parent / "submission.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(output, arcname="prediction.json")
            print(f"zip={zip_path}")

    print(f"output={output}")


if __name__ == "__main__":
    main()
