from __future__ import annotations

import argparse
import json

from goalflow.data import TrackCatalog
from goalflow.validation import validate_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_json")
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()

    catalog = TrackCatalog()
    with open(args.prediction_json, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    result = validate_predictions(predictions, catalog, expected_count=args.expected_count)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
