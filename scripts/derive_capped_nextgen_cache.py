from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive a smaller nextgen pool/feature cache from a larger capped cache.")
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--meta-json", default="")
    return parser.parse_args()


def cap_group(group: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(group) <= cap:
        return group
    base = group.loc[group.get("is_nextgen_candidate", 0).fillna(0).astype(int) == 0]
    if len(base) >= cap:
        return base
    new = group.loc[group.get("is_nextgen_candidate", 0).fillna(0).astype(int) > 0].copy()
    sort_cols = [col for col in ["rrf_score", "best_source_rank"] if col in new.columns]
    if sort_cols:
        ascending = [False if col == "rrf_score" else True for col in sort_cols]
        new = new.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return pd.concat([base, new.head(cap - len(base))], ignore_index=False)


def main() -> None:
    args = parse_args()
    df = pd.read_pickle(args.input_pkl)
    pieces = []
    for _, group in tqdm(df.groupby("group_id", sort=False), desc=f"Cap groups to {args.cap}"):
        pieces.append(cap_group(group, args.cap))
    out = pd.concat(pieces, ignore_index=True)
    output = Path(args.output_pkl)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(output)
    if args.meta_json:
        meta = {"rows": len(out), "columns": list(out.columns), "source": args.input_pkl, "cap": args.cap}
        Path(args.meta_json).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {output} rows={len(out)}")


if __name__ == "__main__":
    main()
