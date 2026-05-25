from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from goalflow.data import TrackCatalog
from goalflow.validation import validate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RRF-ensemble multiple prediction files.")
    parser.add_argument("--mode", choices=["dev", "blind"], required=True)
    parser.add_argument("--tid", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--input", action="append", required=True, help="Prediction JSON path. Repeat per model.")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--response-source", type=int, default=0, help="Input index whose responses are copied.")
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def load_predictions(paths: list[Path]) -> list[list[dict]]:
    predictions = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected = len(predictions[0])
    for path, rows in zip(paths, predictions, strict=True):
        if len(rows) != expected:
            raise ValueError(f"{path} has {len(rows)} rows; expected {expected}")
    return predictions


def rrf_track_ids(rows: list[dict], *, rrf_k: int) -> list[str]:
    scores: dict[str, float] = {}
    first_seen: list[str] = []
    seen = set()
    for row in rows:
        for rank, track_id in enumerate(row["predicted_track_ids"], start=1):
            scores[track_id] = scores.get(track_id, 0.0) + 1.0 / (rrf_k + rank)
            if track_id not in seen:
                seen.add(track_id)
                first_seen.append(track_id)
    order = {track_id: index for index, track_id in enumerate(first_seen)}
    return sorted(scores, key=lambda track_id: (-scores[track_id], order[track_id]))[:20]


def output_path(project_root: Path, tid: str, mode: str) -> Path:
    split_dir = "blindset_A" if mode == "blind" else "devset"
    out_dir = project_root / "experiments" / tid / split_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / ("prediction.json" if mode == "blind" else f"{tid}.json")


def copy_to_official(project_root: Path, tid: str, prediction_path: Path) -> None:
    official_dir = project_root.parent / "music-crs-evaluator" / "exp" / "inference" / "devset"
    official_dir.mkdir(parents=True, exist_ok=True)
    official_path = official_dir / f"{tid}.json"
    official_path.write_text(prediction_path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    paths = [Path(path) for path in args.input]
    predictions = load_predictions(paths)
    if args.response_source < 0 or args.response_source >= len(predictions):
        raise ValueError("--response-source must point to one of the inputs")

    ensembled = []
    for row_index, source_row in enumerate(predictions[args.response_source]):
        rows = [model_rows[row_index] for model_rows in predictions]
        output_row = dict(source_row)
        output_row["predicted_track_ids"] = rrf_track_ids(rows, rrf_k=args.rrf_k)
        ensembled.append(output_row)

    validation = validate_predictions(ensembled, TrackCatalog(), expected_count=len(ensembled))
    if not validation["ok"]:
        raise ValueError(f"Invalid ensembled predictions: {validation}")

    prediction_path = output_path(project_root, args.tid, args.mode)
    prediction_path.write_text(json.dumps(ensembled, ensure_ascii=False), encoding="utf-8")
    result = {"output": str(prediction_path), "validation": validation}
    if args.mode == "blind" and not args.no_zip:
        zip_path = prediction_path.parent / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(prediction_path, arcname="prediction.json")
        result["zip"] = str(zip_path)
    if args.mode == "dev":
        copy_to_official(project_root, args.tid, prediction_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
