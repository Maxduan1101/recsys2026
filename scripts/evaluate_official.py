from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the official devset evaluator for a GoalFlow tid.")
    parser.add_argument("--tid", required=True)
    parser.add_argument(
        "--workspace-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Path containing music-crs-evaluator and .venv.",
    )
    args = parser.parse_args()

    root = Path(args.workspace_root)
    evaluator = root / "music-crs-evaluator"
    python = root / ".venv" / "bin" / "python"
    subprocess.run(
        [str(python), "evaluate_devset.py", "--eval_dataset", "devset", "--tid", args.tid],
        cwd=str(evaluator),
        check=True,
    )
    print(evaluator / "exp" / "scores" / "devset" / f"{args.tid}.json")


if __name__ == "__main__":
    main()
