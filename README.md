# GoalFlow-MusicCRS

GoalFlow-MusicCRS is an isolated experiment project for RecSys Challenge 2026 Music-CRS. It does not modify the official baseline repository.

For a beginner-friendly system design and work log in Chinese, read:

```text
docs/SYSTEM_REPORT.md
```

The first runnable system implements:

- goal-aware conversation state construction;
- train-context track document augmentation;
- multiple BM25 item indices;
- weighted reciprocal-rank fusion;
- a copied legacy BM25-history source kept as a high-weight anchor;
- optional `legacy_head_k` protection so early ranks can stay aligned with the strongest known baseline while later ranks are diversified;
- lightweight entity, era, tag, profile, and seed-feedback boosts;
- conservative top-20 diversity post-processing;
- optional source-gated fusion and tail-only global-repeat diversification;
- LightGBM LambdaRank reranking for a learned source-fusion path;
- cached LTR candidate frames for faster hyperparameter sweeps;
- RRF ensembling over compatible prediction files;
- configurable response styles, with `judge_v2` as the current LTR submission default and `compact_broad` as the high-lexical backup;
- devset prediction, official evaluator compatibility, Blind A `submission.zip` generation.

## Run

```bash
cd /Users/bytedance/generated_problems/recsys2026_music_crs
source .venv/bin/activate
python -m pip install -e goalflow_musiccrs
```

Quick smoke test:

```bash
python goalflow_musiccrs/scripts/run_goalflow.py \
  --mode dev \
  --tid goalflow_smoke \
  --dev-limit 5 \
  --retrieval-top-k 80
```

Full dev run:

```bash
python goalflow_musiccrs/scripts/run_goalflow.py \
  --mode dev \
  --tid goalflow_safe_bm25_response \
  --retrieval-top-k 180 \
  --rerank-pool-size 900 \
  --legacy-head-k 20

python goalflow_musiccrs/scripts/evaluate_official.py \
  --tid goalflow_safe_bm25_response
```

Blind A package:

```bash
python goalflow_musiccrs/scripts/run_goalflow.py \
  --mode blind \
  --tid goalflow_safe_bm25_response \
  --retrieval-top-k 180 \
  --rerank-pool-size 900 \
  --legacy-head-k 20
```

The upload file will be:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_safe_bm25_response/blindset_A/submission.zip
```

The current shifted-feedback safe package is:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_safe_bm25_response_shifted_feedback/blindset_A/submission.zip
```

Tail-diversity Blind A candidate:

```bash
python goalflow_musiccrs/scripts/run_goalflow.py \
  --mode blind \
  --tid goalflow_taildiv_head15 \
  --retrieval-top-k 180 \
  --rerank-pool-size 900 \
  --legacy-head-k 15 \
  --tail-diversity-start 15 \
  --global-repeat-penalty 0.06
```

The upload file is:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_taildiv_head15/blindset_A/submission.zip
```

Current recommended Blind A package:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ltr120_lambda2_head0_judge_v2_clean/blindset_A/submission.zip
```

This package uses full-dev LightGBM LambdaRank training for Blind A, with `reg_lambda=2`, then generates short judge-focused metadata-grounded responses. OOF dev validation, where each dev turn is scored by a model that did not train on it, reaches official `nDCG@20=0.18302`, catalog diversity `0.52801`, and lexical diversity `0.14874`. A high-lexical response backup with the same LTR ranking is:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ltr120_lambda2_head0_compact_broad_clean/blindset_A/submission.zip
```

OOF-max micro-gain ensemble package:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v2_clean/blindset_A/submission.zip
```

This package RRF-ensembles the 120/140/200-tree `reg_lambda=2` LTR rankings with `rrf_k=60`. It has the best local OOF `nDCG@20=0.18325`, but the gain over the single 120-tree LTR is only `+0.00023`, so keep the single-model package as the more conservative main choice unless submission budget allows testing the ensemble. Its high-lexical backup is:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_compact_broad_clean/blindset_A/submission.zip
```

The previous LTR package is still available as a fallback:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ltr120_head0_judge_v2_clean/blindset_A/submission.zip
```

The older 260-tree LTR package is also available as a fallback:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ltr_head0_polished_v3/blindset_A/submission.zip
```

Previous conservative Blind A package:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_head20_compact_broad/blindset_A/submission.zip
```

This package preserves the previously strong BM25-head ranking and only changes the response text. A safer diversity backup is:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_taildiv_head19_compact_broad/blindset_A/submission.zip
```

A stronger diversity backup is:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_taildiv_head18_compact_broad/blindset_A/submission.zip
```

Source-level retrieval diagnostics:

```bash
python goalflow_musiccrs/scripts/diagnose_retrieval_sources.py \
  --tid source_diag_full
```

The diagnostic report is written to:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/source_diag_full/diagnostics/retrieval_source_summary.csv
```

LTR feature export smoke:

```bash
python goalflow_musiccrs/scripts/export_ltr_dataset.py \
  --tid ltr_export_smoke \
  --dev-limit 3 \
  --max-candidates-per-group 30 \
  --include-missed-gold
```

Progress-label audit:

```bash
python goalflow_musiccrs/scripts/audit_progress_labels.py \
  --split train \
  --sample-sessions 20
```

Official embedding schema check:

```bash
python goalflow_musiccrs/scripts/inspect_embeddings.py
```

Prediction diversity summary without gold labels:

```bash
python goalflow_musiccrs/scripts/summarize_predictions.py \
  goalflow_musiccrs/experiments/goalflow_head20_compact_broad/blindset_A/prediction.json
```

Blind-A-shaped dev panel validation:

```bash
python goalflow_musiccrs/scripts/evaluate_blind_like.py \
  --prediction head20=music-crs-evaluator/exp/inference/devset/goalflow_bm25_aug_v3_head20.json \
  --prediction head18=goalflow_musiccrs/experiments/goalflow_taildiv_head18_compactresp_v2/devset/goalflow_taildiv_head18_compactresp_v2.json \
  --baseline-label head20
```

Protected seed-CF tail rescue experiment:

```bash
python goalflow_musiccrs/scripts/apply_embedding_tail_rescue.py \
  --mode dev \
  --input goalflow_musiccrs/experiments/goalflow_head20_style_compact_broad/devset/goalflow_head20_style_compact_broad.json \
  --tid goalflow_head20_cf_tail19 \
  --preserve-head-k 19 \
  --copy-to-official-evaluator
```

Important: use `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings`, not the older `TalkPlayData-2-Track-Embeddings` from the baseline tips. The Challenge embedding `all_tracks` split has full overlap with the Challenge track catalog.

LTR validation and Blind A package:

```bash
# One-time local dependency for LTR experiments. On macOS LightGBM may also need:
# brew install libomp
python -m pip install -e "goalflow_musiccrs[ltr]"

python goalflow_musiccrs/scripts/run_ltr_rerank.py \
  --mode oof-dev \
  --tid goalflow_ltr120_lambda2_head0_oof_judge_v2 \
  --max-candidates-per-group 300 \
  --n-estimators 120 \
  --subsample 1.0 \
  --colsample-bytree 0.9 \
  --reg-lambda 2 \
  --preserve-head-k 0 \
  --response-style judge_v2

python goalflow_musiccrs/scripts/run_ltr_rerank.py \
  --mode blind \
  --tid goalflow_ltr120_lambda2_head0_judge_v2_clean \
  --max-candidates-per-group 300 \
  --n-estimators 120 \
  --subsample 1.0 \
  --colsample-bytree 0.9 \
  --reg-lambda 2 \
  --preserve-head-k 0 \
  --response-style judge_v2

python goalflow_musiccrs/scripts/refresh_responses.py \
  --mode blind \
  --input goalflow_musiccrs/experiments/goalflow_ltr120_lambda2_head0_judge_v2_clean/blindset_A/prediction.json \
  --tid goalflow_ltr120_lambda2_head0_compact_broad_clean \
  --response-style compact_broad \
  --zip

python goalflow_musiccrs/scripts/ensemble_predictions.py \
  --mode blind \
  --tid goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v2_clean \
  --input goalflow_musiccrs/experiments/goalflow_ltr120_lambda2_head0_judge_v2_clean/blindset_A/prediction.json \
  --input goalflow_musiccrs/experiments/goalflow_ltr140_lambda2_head0_judge_v2_clean/blindset_A/prediction.json \
  --input goalflow_musiccrs/experiments/goalflow_ltr200_lambda2_head0_judge_v2_clean/blindset_A/prediction.json \
  --rrf-k 60
```

Older polished-response fallback:

```bash
python goalflow_musiccrs/scripts/refresh_responses.py \
  --mode blind \
  --input goalflow_musiccrs/experiments/goalflow_ltr_head0_compact_broad/blindset_A/prediction.json \
  --tid goalflow_ltr_head0_polished_v3 \
  --response-style polished \
  --zip
```

## Current Scope

This is Phase 1/2 infrastructure. It deliberately avoids direct dependence on gated LLaMA, GPU-only FlashAttention, FAISS, or cross-encoder models. LightGBM is optional under the `ltr` extra and is now the strongest local ranking path. Remaining research items are tracked in `research/DEEP_RESEARCH_BACKLOG.md`.

The current primary submission setting uses unprotected LightGBM LambdaRank with `preserve_head_k=0`, `n_estimators=120`, `colsample_bytree=0.9`, and `reg_lambda=2`. The older `legacy_head_k=20` package is now only a conservative fallback: it preserves the strongest BM25-style head ranking, but its local dev validation is far below the tuned LTR path.

Public Blind A feedback from one conservative submission was `nDCG@20=0.1935`, `catalog_diversity=0.0257`, `lexical_diversity=0.0125`, and `llm_judge_score=1.0`. Blind A currently has only 80 rows, so the maximum possible catalog diversity is `1600 / 47071 = 0.0340`; catalog is not the main bottleneck. After OOF validation and LTR tuning, the immediate conservative submission candidate is `goalflow_ltr120_lambda2_head0_judge_v2_clean`, with `goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v2_clean` as a tiny-OOF-gain ensemble candidate, `goalflow_ltr120_lambda2_head0_compact_broad_clean` as a higher-Distinct-2 text backup, and `goalflow_head20_compact_broad` as the conservative legacy-rank backup.

The newest diagnostics show the added sources are useful for recall but not yet calibrated for rank fusion: the best single source per dev state reaches hit@20 `0.4715` / nDCG@20 `0.2600`, while the current RRF fusion reaches hit@20 `0.2595` / nDCG@20 `0.1015`. Legacy-vs-fused deltas show `446` gained top-20 hits but `212` lost hits and `642` demotions, so the next implementation step is source gating or a learning-to-rank model rather than simply adding more BM25 sources.
