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
- compact, metadata-grounded response templates with user-query, profile, and feedback cues;
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
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_head20_compactresp_v2/blindset_A/submission.zip
```

This package preserves the previously strong BM25-head ranking and only changes the response text. A safer diversity backup is:

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs/experiments/goalflow_taildiv_head18_compactresp_v2/blindset_A/submission.zip
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
  goalflow_musiccrs/experiments/goalflow_head20_compactresp_v2/blindset_A/prediction.json
```

Important: use `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings`, not the older `TalkPlayData-2-Track-Embeddings` from the baseline tips. The Challenge embedding `all_tracks` split has full overlap with the Challenge track catalog.

## Current Scope

This is Phase 1/2 infrastructure. It deliberately avoids direct dependence on gated LLaMA, GPU-only FlashAttention, LightGBM, FAISS, or cross-encoder models. Those are tracked in `research/DEEP_RESEARCH_BACKLOG.md`.

The current safe submission setting uses `legacy_head_k=20`: recommendation IDs exactly preserve the strongest known BM25 dev ranking while GoalFlow upgrades response generation. Experimental settings with lower `legacy_head_k` are useful for research, but currently reduce nDCG.

Public Blind A feedback from one conservative submission was `nDCG@20=0.1935`, `catalog_diversity=0.0257`, `lexical_diversity=0.0125`, and `llm_judge_score=1.0`. Blind A currently has only 80 rows, so the maximum possible catalog diversity is `1600 / 47071 = 0.0340`; catalog is not the main bottleneck. The immediate best submission is therefore `goalflow_head20_compactresp_v2`: keep the ranking fixed and test the stronger text generator. The `goalflow_taildiv_head18_compactresp_v2` package is a backup that changes only the final two ranks.

The newest diagnostics show the added sources are useful for recall but not yet calibrated for rank fusion: the best single source per dev state reaches hit@20 `0.4715` / nDCG@20 `0.2600`, while the current RRF fusion reaches hit@20 `0.2595` / nDCG@20 `0.1015`. Legacy-vs-fused deltas show `446` gained top-20 hits but `212` lost hits and `642` demotions, so the next implementation step is source gating or a learning-to-rank model rather than simply adding more BM25 sources.
