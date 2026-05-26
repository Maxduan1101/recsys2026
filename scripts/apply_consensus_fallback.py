from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from goalflow.data import TRACK_METADATA, TrackCatalog
from goalflow.validation import validate_predictions


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def topk_support(rows_by_component: list[list[dict]], row_index: int, track_id: str, top_k: int) -> int:
    return sum(
        1
        for rows in rows_by_component
        if track_id in rows[row_index]["predicted_track_ids"][:top_k]
    )


def apply_consensus_fallback(
    base_rows: list[dict],
    fallback_rows: list[dict],
    component_rows: list[list[dict]],
    *,
    support_top_k: int,
    base_max_support: int,
    fallback_min_support: int,
) -> tuple[list[dict], dict]:
    if len(base_rows) != len(fallback_rows):
        raise ValueError("Base and fallback predictions have different row counts")
    for rows in component_rows:
        if len(rows) != len(base_rows):
            raise ValueError("Component prediction row count does not match base")

    output = []
    changed_rows = 0
    top1_changed = 0
    for index, base_row in enumerate(base_rows):
        fallback_row = fallback_rows[index]
        base_top = base_row["predicted_track_ids"][0]
        fallback_top = fallback_row["predicted_track_ids"][0]
        use_fallback = (
            base_top != fallback_top
            and topk_support(component_rows, index, base_top, support_top_k) <= base_max_support
            and topk_support(component_rows, index, fallback_top, support_top_k) >= fallback_min_support
        )
        if use_fallback:
            row = dict(base_row)
            row["predicted_track_ids"] = list(fallback_row["predicted_track_ids"])
            changed_rows += 1
            top1_changed += 1
        else:
            row = base_row
        output.append(row)
    stats = {
        "rows": len(base_rows),
        "changed_rows": changed_rows,
        "top1_changed": top1_changed,
        "support_top_k": support_top_k,
        "base_max_support": base_max_support,
        "fallback_min_support": fallback_min_support,
    }
    return output, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a conservative model-consensus fallback rule.")
    parser.add_argument("--mode", choices=["dev", "blind"], required=True)
    parser.add_argument("--tid", required=True)
    parser.add_argument("--base", required=True, help="Primary prediction JSON.")
    parser.add_argument("--fallback", required=True, help="Fallback prediction JSON used when consensus says to switch.")
    parser.add_argument("--component", action="append", required=True, help="Component model prediction JSON; repeatable.")
    parser.add_argument("--support-top-k", type=int, default=1)
    parser.add_argument("--base-max-support", type=int, default=0)
    parser.add_argument("--fallback-min-support", type=int, default=2)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--track-metadata-name", default=TRACK_METADATA)
    parser.add_argument("--copy-to-official-evaluator", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def output_path(project_root: Path, tid: str, mode: str) -> Path:
    split_dir = "blindset_A" if mode == "blind" else "devset"
    out_dir = project_root / "experiments" / tid / split_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / ("prediction.json" if mode == "blind" else f"{tid}.json")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    base_rows = load_rows(Path(args.base))
    fallback_rows = load_rows(Path(args.fallback))
    component_rows = [load_rows(Path(path)) for path in args.component]
    output, stats = apply_consensus_fallback(
        base_rows,
        fallback_rows,
        component_rows,
        support_top_k=args.support_top_k,
        base_max_support=args.base_max_support,
        fallback_min_support=args.fallback_min_support,
    )
    validation = validate_predictions(
        output,
        TrackCatalog(args.track_metadata_name),
        expected_count=len(base_rows),
    )
    if not validation["ok"]:
        raise ValueError(f"Invalid consensus fallback output: {validation}")

    out_path = output_path(project_root, args.tid, args.mode)
    out_path.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    stats_path = out_path.parent / "consensus_fallback_stats.json"
    stats_path.write_text(json.dumps({**stats, "validation": validation}, indent=2), encoding="utf-8")

    result = {"output": str(out_path), "stats": stats, "validation": validation}
    if args.mode == "dev" and args.copy_to_official_evaluator:
        official = project_root.parent / "music-crs-evaluator" / "exp" / "inference" / "devset"
        official.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_path, official / f"{args.tid}.json")
    if args.mode == "blind" and not args.no_zip:
        zip_path = out_path.parent / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, arcname="prediction.json")
        result["zip"] = str(zip_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
