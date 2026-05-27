from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a high-density/shadow candidate pool by source family, extra gold, "
            "rankability, and candidate-size cost."
        )
    )
    parser.add_argument("--pool-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-pool-pkl", default="", help="Optional baseline pool for unique-extra-over-base metrics.")
    parser.add_argument(
        "--score-pkl",
        default="",
        help="Optional scored rows containing group_id, track_id and a score column for rankable@K metrics.",
    )
    parser.add_argument("--score-column", default="final_score")
    parser.add_argument("--pool-name", default="pool")
    parser.add_argument("--base-name", default="base")
    return parser.parse_args()


def group_track_keys(df: pd.DataFrame) -> set[tuple[int, str]]:
    return set(zip(df["group_id"].astype(int), df["track_id"].astype(str)))


def gold_groups(df: pd.DataFrame) -> set[int]:
    if "label" not in df:
        return set()
    return set(df.loc[df["label"].fillna(0).astype(int) == 1, "group_id"].astype(int))


def size_summary(df: pd.DataFrame, name: str) -> dict[str, Any]:
    sizes = df.groupby("group_id").size()
    return {
        "pool": name,
        "groups": int(len(sizes)),
        "rows": int(len(df)),
        "group_size_min": int(sizes.min()) if len(sizes) else 0,
        "group_size_p50": float(sizes.quantile(0.50)) if len(sizes) else 0.0,
        "group_size_mean": float(sizes.mean()) if len(sizes) else 0.0,
        "group_size_p90": float(sizes.quantile(0.90)) if len(sizes) else 0.0,
        "group_size_p95": float(sizes.quantile(0.95)) if len(sizes) else 0.0,
        "group_size_p99": float(sizes.quantile(0.99)) if len(sizes) else 0.0,
        "group_size_max": int(sizes.max()) if len(sizes) else 0,
        "gold_in_pool": int(len(gold_groups(df))),
    }


def attach_final_rank(pool: pd.DataFrame, scored: pd.DataFrame, score_column: str) -> pd.DataFrame:
    if score_column not in scored:
        raise ValueError(f"score column {score_column!r} not found in score-pkl")
    score_cols = scored[["group_id", "track_id", score_column]].copy()
    score_cols["group_id"] = score_cols["group_id"].astype(int)
    score_cols["track_id"] = score_cols["track_id"].astype(str)
    score_cols["_final_rank"] = (
        score_cols.groupby("group_id")[score_column]
        .rank(method="first", ascending=False)
        .astype(np.float32)
    )
    out = pool.merge(score_cols[["group_id", "track_id", score_column, "_final_rank"]], on=["group_id", "track_id"], how="left")
    return out


def family_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("nextgen_family_", "source_present_")
    cols = [col for col in df.columns if col.startswith(prefixes)]
    # Keep explicit family/source flags only; bucket/rank/recip columns are analyzed through rank cols.
    return sorted(col for col in cols if not col.endswith("_names"))


def rank_col_for_family(df: pd.DataFrame, family_col: str) -> str | None:
    if family_col.startswith("nextgen_family_"):
        suffix = family_col.removeprefix("nextgen_family_")
        candidates = [
            f"source_rank_nextgen_{suffix}",
            f"rank_nextgen_{suffix}",
        ]
    elif family_col.startswith("source_present_"):
        suffix = family_col.removeprefix("source_present_")
        candidates = [
            f"source_rank_{suffix}",
            f"rank_{suffix}",
        ]
    else:
        candidates = []
    for col in candidates:
        if col in df:
            return col
    return None


def source_family_summary(
    pool: pd.DataFrame,
    base_keys: set[tuple[int, str]],
    base_gold: set[int],
) -> pd.DataFrame:
    rows = []
    pool_keys = group_track_keys(pool)
    new_key_mask = [key not in base_keys for key in zip(pool["group_id"].astype(int), pool["track_id"].astype(str))]
    pool = pool.copy()
    pool["_is_new_over_base"] = np.asarray(new_key_mask, dtype=bool)
    has_rank = "_final_rank" in pool
    for col in family_columns(pool):
        present = pool[col].fillna(0).astype(float) > 0
        subset = pool[present]
        if subset.empty:
            continue
        gold_subset = subset[subset.get("label", 0).fillna(0).astype(int) == 1]
        gold_hit = set(gold_subset["group_id"].astype(int))
        rank_col = rank_col_for_family(pool, col)
        source_kind = "nextgen_family" if col.startswith("nextgen_family_") else "source"
        row: dict[str, Any] = {
            "source_family": col,
            "source_kind": source_kind,
            "candidate_rows": int(len(subset)),
            "groups_present": int(subset["group_id"].nunique()),
            "added_unique_candidates_over_base": int(subset["_is_new_over_base"].sum()),
            "gold_hit": int(len(gold_hit)),
            "unique_extra_gold_over_base": int(len(gold_hit - base_gold)),
            "gold_hit_rankable_le20": None,
            "gold_hit_rankable_le50": None,
            "gold_hit_rankable_le100": None,
            "extra_gold_rankable_le50": None,
            "extra_gold_rankable_le100": None,
            "rankable_density_le50": None,
            "rankable_density_le100": None,
            "median_gold_source_rank": None,
            "p90_gold_source_rank": None,
        }
        if rank_col and len(gold_subset):
            ranks = pd.to_numeric(gold_subset[rank_col], errors="coerce").replace(9999, np.nan).dropna()
            if len(ranks):
                row["median_gold_source_rank"] = float(ranks.quantile(0.50))
                row["p90_gold_source_rank"] = float(ranks.quantile(0.90))
        if has_rank and len(gold_subset):
            extra = gold_subset[gold_subset["group_id"].astype(int).isin(gold_hit - base_gold)]
            for k in [20, 50, 100]:
                row[f"gold_hit_rankable_le{k}"] = int((gold_subset["_final_rank"].fillna(np.inf) <= k).sum())
            row["extra_gold_rankable_le50"] = int((extra["_final_rank"].fillna(np.inf) <= 50).sum())
            row["extra_gold_rankable_le100"] = int((extra["_final_rank"].fillna(np.inf) <= 100).sum())
            denom = max(int(subset["_is_new_over_base"].sum()), 1)
            row["rankable_density_le50"] = float(row["extra_gold_rankable_le50"] / denom)
            row["rankable_density_le100"] = float(row["extra_gold_rankable_le100"] / denom)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["unique_extra_gold_over_base", "gold_hit", "candidate_rows"],
        ascending=[False, False, True],
    )


def extra_gold_diagnostics(pool: pd.DataFrame, base_gold: set[int]) -> pd.DataFrame:
    if "label" not in pool:
        return pd.DataFrame()
    gold = pool[pool["label"].fillna(0).astype(int) == 1].copy()
    if gold.empty:
        return pd.DataFrame()
    gold["bucket"] = np.where(gold["group_id"].astype(int).isin(base_gold), "base_hit", "extra_gold")
    keep_cols = [
        "bucket",
        "group_id",
        "session_id",
        "turn_number",
        "track_id",
        "source_count",
        "best_source_rank",
        "rrf_score",
        "_final_rank",
    ]
    family_cols = [col for col in family_columns(gold) if col.startswith("nextgen_family_")]
    keep_cols.extend(family_cols)
    keep_cols = [col for col in keep_cols if col in gold]
    return gold[keep_cols].sort_values(["bucket", "group_id"])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = pd.read_pickle(args.pool_pkl).copy()
    pool["group_id"] = pool["group_id"].astype(int)
    pool["track_id"] = pool["track_id"].astype(str)
    if args.score_pkl:
        scored = pd.read_pickle(args.score_pkl)
        pool = attach_final_rank(pool, scored, args.score_column)

    base = pd.DataFrame()
    base_keys: set[tuple[int, str]] = set()
    base_gold: set[int] = set()
    if args.base_pool_pkl:
        base = pd.read_pickle(args.base_pool_pkl).copy()
        base["group_id"] = base["group_id"].astype(int)
        base["track_id"] = base["track_id"].astype(str)
        base_keys = group_track_keys(base)
        base_gold = gold_groups(base)

    summaries = [size_summary(pool, args.pool_name)]
    if len(base):
        summaries.insert(0, size_summary(base, args.base_name))
        pool_gold = gold_groups(pool)
        summaries[-1]["unique_extra_gold_over_base"] = int(len(pool_gold - base_gold))
        summaries[-1]["lost_gold_vs_base"] = int(len(base_gold - pool_gold))
    pd.DataFrame(summaries).to_csv(out_dir / "pool_shadow_summary.csv", index=False)

    sf = source_family_summary(pool, base_keys, base_gold)
    sf.to_csv(out_dir / "source_family_shadow_summary.csv", index=False)
    extra = extra_gold_diagnostics(pool, base_gold)
    extra.to_csv(out_dir / "extra_gold_rankability_diagnostics.csv", index=False)

    readme = [
        "# Shadow / Admission Pool Analysis",
        "",
        "This report focuses on high-density candidate-pool quality:",
        "- `gold_in_pool`: raw recall upper bound",
        "- `unique_extra_gold_over_base`: new gold found beyond the baseline pool",
        "- `extra_gold_rankable_le50/le100`: extra gold that the current score can lift near the head",
        "- `rankable_density`: rankable extra gold per added candidate row",
        "",
        f"Pool: `{args.pool_name}` from `{args.pool_pkl}`",
    ]
    if args.base_pool_pkl:
        readme.append(f"Base: `{args.base_name}` from `{args.base_pool_pkl}`")
    if args.score_pkl:
        readme.append(f"Rankability score: `{args.score_column}` from `{args.score_pkl}`")
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    manifest = {
        "pool_pkl": args.pool_pkl,
        "base_pool_pkl": args.base_pool_pkl,
        "score_pkl": args.score_pkl,
        "score_column": args.score_column,
        "outputs": [
            "pool_shadow_summary.csv",
            "source_family_shadow_summary.csv",
            "extra_gold_rankability_diagnostics.csv",
            "README.md",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote shadow/admission analysis to {out_dir}")


if __name__ == "__main__":
    main()
