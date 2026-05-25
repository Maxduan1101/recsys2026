from __future__ import annotations

import argparse
from pathlib import Path

from goalflow.pipeline import GoalFlowConfig, run_blind, run_dev, write_run_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run GoalFlow-MusicCRS experiments.")
    parser.add_argument("--mode", choices=["dev", "blind"], required=True)
    parser.add_argument("--tid", default="goalflow_bm25_aug_v1")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--blind-dataset-name", default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    parser.add_argument("--retrieval-top-k", type=int, default=260)
    parser.add_argument("--rerank-pool-size", type=int, default=1200)
    parser.add_argument("--legacy-head-k", type=int, default=20)
    parser.add_argument("--fusion-mode", choices=["standard", "gated"], default="standard")
    parser.add_argument("--tail-diversity-start", type=int, default=20)
    parser.add_argument("--global-repeat-penalty", type=float, default=0.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--response-style",
        choices=[
            "compact", "compact_broad", "concise", "setwise", "natural", "polished",
            "judge_v1", "judge_v2", "judge_v3", "judge_mix", "judge_brief",
            "judge_planned", "judge_compact_mix", "judge_clean_mix", "judge_balanced_mix",
        ],
        default="compact",
    )
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-train-augmentation", action="store_true")
    parser.add_argument("--no-copy-to-official-evaluator", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = GoalFlowConfig(
        project_root=Path(args.project_root),
        tid=args.tid,
        blind_dataset_name=args.blind_dataset_name,
        use_train_augmentation=not args.no_train_augmentation,
        rebuild_cache=args.rebuild_cache,
        retrieval_top_k=args.retrieval_top_k,
        rerank_pool_size=args.rerank_pool_size,
        legacy_head_k=args.legacy_head_k,
        fusion_mode=args.fusion_mode,
        tail_diversity_start=args.tail_diversity_start,
        global_repeat_penalty=args.global_repeat_penalty,
        rrf_k=args.rrf_k,
        response_style=args.response_style,
        dev_limit=args.dev_limit,
    )
    if args.mode == "dev":
        output = run_dev(config, copy_to_official_evaluator=not args.no_copy_to_official_evaluator)
    else:
        output = run_blind(config, zip_submission=not args.no_zip)
    summary = write_run_summary(config, output, args.mode)
    print(f"output={output}")
    print(f"summary={summary}")


if __name__ == "__main__":
    main()
