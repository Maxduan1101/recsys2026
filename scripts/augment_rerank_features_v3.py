from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CHANNELS = ["track_cf", "attributes", "audio", "lyrics", "image", "metadata"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add derived semantic-source interaction features to an existing rerank_v2 feature cache. "
            "This avoids rebuilding expensive text/embedding features."
        )
    )
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--meta-json", default="")
    return parser.parse_args()


def col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df:
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0).astype(np.float32)
    return pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)


def flag(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df:
        return pd.to_numeric(df[name], errors="coerce").fillna(0).astype(np.float32)
    return pd.Series(np.zeros(len(df), dtype=np.float32), index=df.index)


def add_max(df: pd.DataFrame, out_name: str, names: list[str]) -> None:
    values = [col(df, name).to_numpy(dtype=np.float32, copy=False) for name in names]
    if not values:
        df[out_name] = np.float32(0.0)
        return
    df[out_name] = np.maximum.reduce(values).astype(np.float32)


def add_mean(df: pd.DataFrame, out_name: str, names: list[str]) -> None:
    values = [col(df, name).to_numpy(dtype=np.float32, copy=False) for name in names if name in df]
    if not values:
        df[out_name] = np.float32(0.0)
        return
    df[out_name] = np.mean(values, axis=0).astype(np.float32)


def weighted_sum(df: pd.DataFrame, weights: dict[str, float], suffix: str) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.float32)
    for channel, weight in weights.items():
        out += np.float32(weight) * col(df, f"emb_{channel}_{suffix}").to_numpy(dtype=np.float32, copy=False)
    return out


def augment_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add v3 semantic interaction features to an in-memory rerank feature frame."""
    df = df.copy()
    before_cols = set(df.columns)

    pos_mean_cols = [f"emb_{channel}_pos_mean" for channel in CHANNELS]
    pos_max_cols = [f"emb_{channel}_pos_max" for channel in CHANNELS]
    weak_mean_cols = [f"emb_{channel}_weak_history_mean" for channel in CHANNELS]
    weak_max_cols = [f"emb_{channel}_weak_history_max" for channel in CHANNELS]
    minus_cols = [f"emb_{channel}_pos_minus_neg" for channel in CHANNELS]

    add_max(df, "sem_pos_mean_max_all_channels", pos_mean_cols)
    add_max(df, "sem_pos_max_max_all_channels", pos_max_cols)
    add_max(df, "sem_weak_mean_max_all_channels", weak_mean_cols)
    add_max(df, "sem_weak_max_max_all_channels", weak_max_cols)
    add_max(df, "sem_pos_minus_neg_max_all_channels", minus_cols)
    add_mean(df, "sem_pos_mean_avg_all_channels", pos_mean_cols)
    add_mean(df, "sem_weak_mean_avg_all_channels", weak_mean_cols)
    add_mean(df, "sem_pos_minus_neg_avg_all_channels", minus_cols)

    blend_weights = {
        "track_cf": 0.30,
        "attributes": 0.25,
        "audio": 0.20,
        "lyrics": 0.10,
        "image": 0.05,
        "metadata": 0.10,
    }
    df["sem_pos_mean_blend"] = weighted_sum(df, blend_weights, "pos_mean")
    df["sem_pos_max_blend"] = weighted_sum(df, blend_weights, "pos_max")
    df["sem_weak_history_mean_blend"] = weighted_sum(df, blend_weights, "weak_history_mean")
    df["sem_weak_history_max_blend"] = weighted_sum(df, blend_weights, "weak_history_max")
    df["sem_pos_minus_neg_blend"] = weighted_sum(df, blend_weights, "pos_minus_neg")

    semantic_thresholds = {
        "track_cf": 0.18,
        "attributes": 0.88,
        "audio": 0.75,
        "lyrics": 0.82,
        "image": 0.78,
        "metadata": 0.82,
    }
    support = np.zeros(len(df), dtype=np.float32)
    weak_support = np.zeros(len(df), dtype=np.float32)
    for channel, threshold in semantic_thresholds.items():
        support += (col(df, f"emb_{channel}_pos_mean").to_numpy(dtype=np.float32, copy=False) >= threshold).astype(np.float32)
        weak_support += (
            col(df, f"emb_{channel}_weak_history_mean").to_numpy(dtype=np.float32, copy=False) >= threshold
        ).astype(np.float32)
    df["sem_pos_support_count"] = support
    df["sem_weak_support_count"] = weak_support

    low_current_lex = (col(df, "current_metadata_jaccard").to_numpy(dtype=np.float32, copy=False) < 0.03).astype(np.float32)
    low_goal_lex = (col(df, "goal_metadata_jaccard").to_numpy(dtype=np.float32, copy=False) < 0.03).astype(np.float32)
    df["low_current_lex_x_sem_pos_blend"] = low_current_lex * df["sem_pos_mean_blend"].to_numpy(dtype=np.float32, copy=False)
    df["low_goal_lex_x_sem_pos_blend"] = low_goal_lex * df["sem_pos_mean_blend"].to_numpy(dtype=np.float32, copy=False)
    df["low_current_lex_x_sem_weak_blend"] = low_current_lex * df["sem_weak_history_mean_blend"].to_numpy(
        dtype=np.float32, copy=False
    )

    nextgen = flag(df, "is_nextgen_candidate")
    df["nextgen_x_sem_pos_blend"] = nextgen * df["sem_pos_mean_blend"].to_numpy(dtype=np.float32, copy=False)
    df["nextgen_x_sem_weak_blend"] = nextgen * df["sem_weak_history_mean_blend"].to_numpy(dtype=np.float32, copy=False)
    df["nextgen_x_sem_support_count"] = nextgen * df["sem_pos_support_count"].to_numpy(dtype=np.float32, copy=False)

    family_map = {
        "audio_seed": "audio",
        "attributes_seed": "attributes",
        "cf_seed": "track_cf",
        "lyrics_seed": "lyrics",
        "image_seed": "image",
        "metadata_seed": "metadata",
    }
    family_score = np.zeros(len(df), dtype=np.float32)
    family_rank_quality = np.zeros(len(df), dtype=np.float32)
    family_support = np.zeros(len(df), dtype=np.float32)
    for family, channel in family_map.items():
        fam = flag(df, f"nextgen_family_{family}").to_numpy(dtype=np.float32, copy=False)
        pos_mean = col(df, f"emb_{channel}_pos_mean").to_numpy(dtype=np.float32, copy=False)
        weak_mean = col(df, f"emb_{channel}_weak_history_mean").to_numpy(dtype=np.float32, copy=False)
        recip = col(df, f"source_recip_rank_nextgen_{family}").to_numpy(dtype=np.float32, copy=False)
        df[f"family_{family}_x_{channel}_pos_mean"] = fam * pos_mean
        df[f"family_{family}_x_{channel}_weak_mean"] = fam * weak_mean
        df[f"family_{family}_x_{channel}_pos_minus_neg"] = fam * col(df, f"emb_{channel}_pos_minus_neg").to_numpy(
            dtype=np.float32, copy=False
        )
        df[f"family_{family}_x_rank_quality"] = fam * recip
        family_score += fam * np.maximum(pos_mean, weak_mean)
        family_rank_quality += fam * recip
        family_support += fam
    df["nextgen_family_semantic_score"] = family_score
    df["nextgen_family_rank_quality_sum"] = family_rank_quality
    df["nextgen_embedding_family_count"] = family_support

    if "nextgen_source_count" in df:
        src_count = col(df, "nextgen_source_count").to_numpy(dtype=np.float32, copy=False)
        df["nextgen_source_count_x_sem_pos_blend"] = src_count * df["sem_pos_mean_blend"].to_numpy(
            dtype=np.float32, copy=False
        )
        df["nextgen_source_count_x_sem_weak_blend"] = src_count * df["sem_weak_history_mean_blend"].to_numpy(
            dtype=np.float32, copy=False
        )

    for name in sorted(set(df.columns) - before_cols):
        if pd.api.types.is_float_dtype(df[name]):
            df[name] = df[name].astype(np.float32)

    added = sorted(set(df.columns) - before_cols)
    return df, added


def main() -> None:
    args = parse_args()
    df = pd.read_pickle(args.input_pkl)
    df, added = augment_features(df)

    output = Path(args.output_pkl)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output)
    if args.meta_json:
        Path(args.meta_json).write_text(
            json.dumps(
                {
                    "rows": len(df),
                    "source": args.input_pkl,
                    "added_columns": added,
                    "num_added_columns": len(added),
                    "columns": list(df.columns),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"wrote {output} rows={len(df)} added_columns={len(added)}")
    for name in added:
        print(name)


if __name__ == "__main__":
    main()
