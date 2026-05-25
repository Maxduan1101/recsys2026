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
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_clean_mix_clean/blindset_A/submission.zip
```

This package RRF-ensembles the 120/140/200-tree `reg_lambda=2` LTR rankings plus the 120-tree `colsample_bytree=1.0` L2 variant with `rrf_k=60`, then generates a cleaner high-lexical mix of `judge_v2`, `judge_brief`, and cleaned `compact` responses. OOF dev ranking rises to official `nDCG@20=0.18348` versus `0.18302` for the single 120-tree L2 model and `0.18325` for the previous three-model ensemble. Blind-A-shaped 500-panel validation also favors it: mean delta `+0.00298` nDCG@20 versus the single model. Blind A local Distinct-2 is `0.61068`.

Its high-lexical backups with the same four-model ranking are:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_compact_clean/blindset_A/submission.zip
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_compact_broad_clean/blindset_A/submission.zip
```

Conservative single-model clean-mix package:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ltr120_lambda2_head0_judge_clean_mix_clean/blindset_A/submission.zip
```

This package keeps the same full-dev LightGBM LambdaRank ranking as the conservative L2 package. OOF dev ranking is official `nDCG@20=0.18302`; lexical diversity improves from `0.14874` for fixed `judge_v2` and `0.15926` for `judge_mix` to `0.19930`, while Blind A local Distinct-2 rises from `0.48531` to `0.60692`.

Lower-risk mixed response package:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ltr120_lambda2_head0_judge_mix_clean/blindset_A/submission.zip
```

This keeps the same ranking but avoids the more mechanical `compact_broad` templates. It is a text-safety fallback if the LLM judge appears to punish compact metadata-heavy wording.

Conservative fixed-style response package:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ltr120_lambda2_head0_judge_v2_clean/blindset_A/submission.zip
```

This keeps the exact same ranking but uses only the shorter `judge_v2` response template. A clean high-lexical response backup with the same LTR ranking is:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ltr120_lambda2_head0_compact_clean/blindset_A/submission.zip
```

The highest-Distinct-2 mechanical backup with the same LTR ranking is:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ltr120_lambda2_head0_compact_broad_clean/blindset_A/submission.zip
```

Previous three-model micro-gain ensemble package:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_clean_mix_clean/blindset_A/submission.zip
```

This package RRF-ensembles the 120/140/200-tree `reg_lambda=2` LTR rankings with `rrf_k=60` and uses the same clean mixed response style. The ranking gives a tiny OOF gain (`nDCG@20=0.18325` versus `0.18302`) and official dev lexical diversity is `0.19901`, so keep it as a fallback behind the four-model ensemble. Its high-lexical backup is:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_compact_broad_clean/blindset_A/submission.zip
```

High-risk segmented LTR package:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_segcat_ltr120_140_200_ens_judge_v2_clean/blindset_A/submission.zip
```

This package chooses among the 120/140/200-tree L2 LTR rankings and their RRF ensemble by `conversation_goal.category`. It reaches official dev `nDCG@20=0.18407` when the segment map is selected on all OOF dev diagnostics, but nested segment validation drops to about `0.18235`, so treat it as a high-risk experiment rather than the first retry.

Fuller-prose LLM-judge text backups:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ltr120_lambda2_head0_judge_v3_clean/blindset_A/submission.zip
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v3_clean/blindset_A/submission.zip
```

These keep the same rankings as the `judge_v2` packages but use longer, more natural explanations. Local lexical diversity is lower than `judge_v2`, so they are backups for testing whether Gemini-style judging rewards fuller explanation more than Distinct-2.

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
  --tid goalflow_ltr120_lambda2_head0_judge_clean_mix_clean \
  --response-style judge_clean_mix \
  --zip

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

python goalflow_musiccrs/scripts/run_ltr_rerank.py \
  --mode blind \
  --tid goalflow_ltr120_lambda2_col1_head0_judge_v2_clean \
  --max-candidates-per-group 300 \
  --n-estimators 120 \
  --subsample 1.0 \
  --colsample-bytree 1.0 \
  --reg-lambda 2 \
  --preserve-head-k 0 \
  --response-style judge_v2

python goalflow_musiccrs/scripts/ensemble_predictions.py \
  --mode blind \
  --tid goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_v2_clean \
  --input goalflow_musiccrs/experiments/goalflow_ltr120_lambda2_head0_judge_v2_clean/blindset_A/prediction.json \
  --input goalflow_musiccrs/experiments/goalflow_ltr140_lambda2_head0_judge_v2_clean/blindset_A/prediction.json \
  --input goalflow_musiccrs/experiments/goalflow_ltr200_lambda2_head0_judge_v2_clean/blindset_A/prediction.json \
  --input goalflow_musiccrs/experiments/goalflow_ltr120_lambda2_col1_head0_judge_v2_clean/blindset_A/prediction.json \
  --rrf-k 60

python goalflow_musiccrs/scripts/refresh_responses.py \
  --mode blind \
  --input goalflow_musiccrs/experiments/goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_v2_clean/blindset_A/prediction.json \
  --tid goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_clean_mix_clean \
  --response-style judge_clean_mix \
  --zip

python goalflow_musiccrs/scripts/select_segmented_predictions.py \
  --mode blind \
  --tid goalflow_segcat_ltr120_140_200_ens_judge_v2_clean \
  --segment category \
  --default ltr120 \
  --input ltr120=goalflow_musiccrs/experiments/goalflow_ltr120_lambda2_head0_judge_v2_clean/blindset_A/prediction.json \
  --input ltr140=goalflow_musiccrs/experiments/goalflow_ltr140_lambda2_head0_judge_v2_clean/blindset_A/prediction.json \
  --input ltr200=goalflow_musiccrs/experiments/goalflow_ltr200_lambda2_head0_judge_v2_clean/blindset_A/prediction.json \
  --input ens=goalflow_musiccrs/experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v2_clean/blindset_A/prediction.json \
  --choice A=ens --choice C=ltr140 --choice G=ltr200 \
  --choice H=ltr140 --choice I=ltr140 --choice J=ltr140 --choice K=ltr200 \
  --response-style judge_v2 \
  --zip
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

The current recommended setting is an unprotected LightGBM LambdaRank RRF ensemble with `preserve_head_k=0`: 120/140/200-tree `reg_lambda=2` models plus a 120-tree `colsample_bytree=1.0` L2 variant. The single 120-tree L2 model remains the conservative fallback. Category-segmented selection has the highest non-nested dev score but fails stricter nested segment validation, so it is high-risk. The older `legacy_head_k=20` package is now only a conservative fallback: it preserves the strongest BM25-style head ranking, but its local dev validation is far below the tuned LTR path.

Public Blind A feedback from one conservative submission was `nDCG@20=0.1935`, `catalog_diversity=0.0257`, `lexical_diversity=0.0125`, and `llm_judge_score=1.0`. Blind A currently has only 80 rows, so the maximum possible catalog diversity is `1600 / 47071 = 0.0340`; catalog is not the main bottleneck. After OOF validation and LTR tuning, the immediate candidate is `goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_clean_mix_clean`, the conservative single-model fallback is `goalflow_ltr120_lambda2_head0_judge_clean_mix_clean`, the lower-risk mixed text fallback is `goalflow_ltr120_lambda2_head0_judge_mix_clean`, the fixed-style conservative text fallback is `goalflow_ltr120_lambda2_head0_judge_v2_clean`, `goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_compact_clean` and `goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_compact_broad_clean` are the current high-lexical ensemble backups, `goalflow_ltr120_lambda2_head0_compact_clean` is the single-model clean high-lexical backup, `goalflow_segcat_ltr120_140_200_ens_judge_v2_clean` is a high-risk segment-selection experiment, and `goalflow_head20_compact_broad` remains the conservative legacy-rank backup.

The newest diagnostics show the added sources are useful for recall but not yet calibrated for rank fusion: the best single source per dev state reaches hit@20 `0.4715` / nDCG@20 `0.2600`, while the current RRF fusion reaches hit@20 `0.2595` / nDCG@20 `0.1015`. Legacy-vs-fused deltas show `446` gained top-20 hits but `212` lost hits and `642` demotions, so the next implementation step is source gating or a learning-to-rank model rather than simply adding more BM25 sources.
