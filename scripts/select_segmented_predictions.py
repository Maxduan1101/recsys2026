from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from goalflow.data import BLIND_A_DATASET, CONVERSATION_DATASET, TrackCatalog
from goalflow.pipeline import GoalFlowConfig
from goalflow.response import generate_response
from goalflow.state import build_state_for_blind_item, build_state_for_dev_turn
from goalflow.validation import validate_predictions

from datasets import load_dataset


def parse_named_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("inputs must use name=/path/to/prediction.json")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("input name cannot be empty")
    return name, Path(path)


def parse_choice(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("choices must use segment_key=input_name")
    key, name = value.split("=", 1)
    key = key.strip()
    name = name.strip()
    if not key or not name:
        raise argparse.ArgumentTypeError("choice key and input name cannot be empty")
    return key, name


def load_dev_states(config: GoalFlowConfig):
    dataset = load_dataset(config.conversation_dataset_name, split="test")
    states = []
    for item in dataset:
        for turn_number in range(1, 9):
            state = build_state_for_dev_turn(item, turn_number)
            if state.gold_track_id:
                states.append(state)
    return states


def load_blind_states(config: GoalFlowConfig):
    dataset = load_dataset(config.blind_dataset_name, split="test")
    return [build_state_for_blind_item(item) for item in dataset]


def segment_key(state, segment: str) -> str:
    if segment == "category":
        return state.category
    if segment == "specificity":
        return state.specificity
    if segment == "turn":
        return str(state.turn_number)
    if segment == "cat_spec":
        return f"{state.category}|{state.specificity}"
    if segment == "turn_spec":
        return f"{state.turn_number}|{state.specificity}"
    if segment == "cat_turn":
        return f"{state.category}|{state.turn_number}"
    raise ValueError(f"Unsupported segment={segment!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a prediction file by choosing among existing runs with a deterministic segment map."
    )
    parser.add_argument("--mode", choices=["dev", "blind"], required=True)
    parser.add_argument("--tid", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--conversation-dataset-name", default=CONVERSATION_DATASET)
    parser.add_argument("--blind-dataset-name", default=BLIND_A_DATASET)
    parser.add_argument(
        "--segment",
        choices=["category", "specificity", "turn", "cat_spec", "turn_spec", "cat_turn"],
        default="category",
    )
    parser.add_argument("--input", action="append", type=parse_named_input, required=True)
    parser.add_argument("--choice", action="append", type=parse_choice, default=[])
    parser.add_argument("--default", required=True)
    parser.add_argument(
        "--response-style",
        choices=[
            "compact",
            "compact_broad",
            "concise",
            "setwise",
            "natural",
            "polished",
            "judge_v1",
            "judge_v2",
            "judge_v3",
            "judge_mix",
            "judge_brief",
            "judge_compact_mix",
        ],
        default="judge_v2",
    )
    parser.add_argument("--zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GoalFlowConfig(
        project_root=Path(args.project_root),
        tid=args.tid,
        conversation_dataset_name=args.conversation_dataset_name,
        blind_dataset_name=args.blind_dataset_name,
        response_style=args.response_style,
    )
    catalog = TrackCatalog(config.track_metadata_name)
    states = load_dev_states(config) if args.mode == "dev" else load_blind_states(config)

    input_paths = dict(args.input)
    choice_map = dict(args.choice)
    if args.default not in input_paths:
        raise ValueError(f"default input {args.default!r} is not present in --input")
    for key, name in choice_map.items():
        if name not in input_paths:
            raise ValueError(f"choice {key!r} points to unknown input {name!r}")

    loaded = {}
    for name, path in input_paths.items():
        predictions = json.loads(path.read_text(encoding="utf-8"))
        if len(predictions) != len(states):
            raise ValueError(f"{name} has {len(predictions)} rows, expected {len(states)}")
        loaded[name] = predictions

    output_predictions = []
    usage_counts = {name: 0 for name in loaded}
    for index, state in enumerate(states):
        key = segment_key(state, args.segment)
        source_name = choice_map.get(key, args.default)
        usage_counts[source_name] += 1
        track_ids = loaded[source_name][index]["predicted_track_ids"]
        output_predictions.append(
            {
                "session_id": state.session_id,
                "user_id": state.user_id,
                "turn_number": state.turn_number,
                "predicted_track_ids": track_ids,
                "predicted_response": generate_response(state, catalog, track_ids, style=args.response_style),
            }
        )

    validation = validate_predictions(output_predictions, catalog, expected_count=len(states))
    if not validation["ok"]:
        raise ValueError(f"Invalid segmented predictions: {validation}")

    if args.mode == "blind":
        out_dir = config.experiments_dir / "blindset_A"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "prediction.json"
        out_path.write_text(json.dumps(output_predictions, ensure_ascii=False), encoding="utf-8")
        final_path = out_path
        if args.zip:
            final_path = out_dir / "submission.zip"
            with zipfile.ZipFile(final_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(out_path, arcname="prediction.json")
    else:
        out_dir = config.experiments_dir / "devset"
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path = out_dir / f"{config.tid}.json"
        final_path.write_text(json.dumps(output_predictions, ensure_ascii=False), encoding="utf-8")
        official = config.project_root.parent / "music-crs-evaluator" / "exp" / "inference" / "devset"
        official.mkdir(parents=True, exist_ok=True)
        (official / f"{config.tid}.json").write_text(final_path.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "tid": args.tid,
        "mode": args.mode,
        "segment": args.segment,
        "default": args.default,
        "choices": choice_map,
        "usage_counts": usage_counts,
        "output": str(final_path),
        "validation": validation,
    }
    summary_path = config.experiments_dir / f"{args.mode}_segmented_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
