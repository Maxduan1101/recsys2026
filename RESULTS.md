# GoalFlow Results

Development-set scores from the official evaluator.

| Run | nDCG@1 | nDCG@10 | nDCG@20 | Catalog Diversity | Lexical Diversity | Notes |
|---|---:|---:|---:|---:|---:|---|
| `bm25_static_devset` | 0.009625 | 0.067601 | 0.085870 | 0.388966 | 0.000125 | Strong copied BM25-history baseline. |
| `goalflow_bm25_aug_v1` | 0.011250 | 0.040984 | 0.067874 | 0.478872 | 0.082518 | Multi-source RRF without legacy head protection. Better diversity, worse ranking. |
| `goalflow_bm25_aug_v2` | 0.012125 | 0.047612 | 0.074497 | 0.481932 | 0.082975 | Added high-weight legacy source, still diluted head ranks. |
| `goalflow_bm25_aug_v3_head10` | 0.009625 | 0.067601 | 0.083776 | 0.451021 | 0.082975 | Preserves legacy top 10, lets GoalFlow change tail. Slight ranking loss. |
| `goalflow_bm25_aug_v3_head20` | 0.009625 | 0.067601 | 0.085870 | 0.388966 | 0.082975 | Current safe baseline: legacy ranking + GoalFlow response. |
| `goalflow_gated_head5` | 0.009875 | 0.055209 | 0.078712 | 0.471161 | 0.073882 | Source-gated fusion. Higher diversity, but still loses too much ranking. |
| `goalflow_taildiv_head10` | 0.009875 | 0.067536 | 0.072093 | 0.832253 | 0.101913 | Aggressive tail diversification after rank 10. Useful stress test, too much nDCG loss. |
| `goalflow_taildiv_head15` | 0.009875 | 0.067536 | 0.081796 | 0.767628 | 0.101913 | Current diversity candidate: preserves first 15, diversifies last 5. |
| `goalflow_head20_compactresp_v2` | 0.009625 | 0.067601 | 0.085870 | 0.388966 | 0.175800 | Recommended public submission candidate: unchanged safe ranking, compact grounded response. |
| `goalflow_taildiv_head18_compactresp_v2` | 0.009875 | 0.067536 | 0.085271 | 0.614327 | 0.175789 | Safer diversity backup: preserves first 18, diversifies final 2. |
| `goalflow_taildiv_head15_compactresp_v2` | 0.009875 | 0.067536 | 0.081796 | 0.767628 | 0.175789 | More aggressive backup; not recommended before head20/head18 because nDCG loss is larger. |

Immediate interpretation:

- Goal/document/augmentation sources are not yet calibrated enough to improve ranking.
- They do improve response diversity and candidate diversity.
- Real Blind A feedback suggests the conservative ranking anchor is strong, while response quality/diversity is the biggest immediate score bottleneck.
- Blind A has only 80 rows, so catalog diversity is capped at `1600 / 47071 = 0.0340`; improving catalog by a few hundred unique tail tracks has a small composite effect compared with preserving nDCG and improving response text.
- Next useful work is response quality validation, safer tail diversity if extra submissions remain, source-level calibration, and then LightGBM or dense retrieval.

## Public Blind A Feedback

User-submitted public Blind A score before the latest tail-diversity work:

| Metric | Score |
|---|---:|
| nDCG@20 | 0.1935 |
| Catalog Diversity | 0.0257 |
| Lexical Diversity | 0.0125 |
| LLM Judge Score | 1.0000 |
| Composite Score | 0.1006 |

Interpretation:

- The conservative BM25/legacy-head ranking is materially useful on the public blind split.
- The public bottleneck is highly templated/weak response text. Catalog diversity looked small numerically, but Blind A's 80-row ceiling is only `0.0340`, and the previous package already used about 76% of its available unique slots.
- The composite score is consistent with `0.5 * nDCG + 0.1 * catalog_diversity + 0.1 * lexical_diversity + 0.3 * ((llm_judge_score - 1) / 4)`. This is an inference, but it exactly explains the reported `0.1006`.
- Pro sanity check saved at `research/pro_answers/round4/tab1_submission_package_decision.txt`: submit `head20_compactresp_v2` first; use `head18` as a later backup.

## Blind A Local Diversity Summaries

These are gold-free checks from `scripts/summarize_predictions.py`; they do not estimate nDCG.

| Package | Unique Tracks | Unique Slot Ratio | Catalog Diversity | Catalog Ceiling | Distinct-2 | Recommendation |
|---|---:|---:|---:|---:|---:|---|
| `goalflow_head20_compactresp_v2` | 1216 | 0.7600 | 0.025833 | 0.033991 | 0.665424 | Submit first: preserves known ranking. |
| `goalflow_taildiv_head18_compactresp_v2` | 1268 | 0.7925 | 0.026938 | 0.033991 | 0.665424 | Backup if one more submission is available. |
| `goalflow_taildiv_head15_compactresp_v2` | 1348 | 0.8425 | 0.028638 | 0.033991 | 0.665424 | Useful experiment, but nDCG risk is likely too high for first retry. |

## Retrieval Source Diagnostics

Run: `source_diag_full`

Artifacts:

- `experiments/source_diag_full/diagnostics/retrieval_source_summary.csv`
- `experiments/source_diag_full/diagnostics/retrieval_source_summary.json`
- `experiments/source_diag_full/diagnostics/legacy_vs_fused_delta_summary.json`

Overall dev source recall/rank summary:

| Source | hit@20 | nDCG@20 | hit@100 | Coverage | Mean Found Rank | Notes |
|---|---:|---:|---:|---:|---:|---|
| `__best_single_source_rank__` | 0.471500 | 0.260000 | 0.628400 | 0.706000 | 33.06 | Oracle over available sources; shows recall potential. |
| `index_any=enriched` | 0.368500 | 0.163200 | 0.529800 | 0.614200 | 40.43 | Best single index family by hit@20. |
| `index_any=metadata_all` | 0.363500 | 0.157700 | 0.521400 | 0.602000 | 39.29 | Similar to enriched; useful but needs calibration. |
| `metadata_all:seed_current` | 0.293800 | 0.109000 | 0.461200 | 0.541000 | 43.72 | Strong individual source. |
| `enriched:seed_current` | 0.285000 | 0.106500 | 0.455500 | 0.544000 | 47.09 | Augmentation helps recall but not enough as-is. |
| `__rrf_fused__` | 0.259500 | 0.101500 | 0.416800 | 0.706000 | 401.31 | Fusion keeps coverage but badly dilutes head ranks. |
| `legacy_metadata:legacy_history` | 0.230250 | 0.085700 | 0.345000 | 0.380625 | 34.35 | Strong ranking anchor with lower coverage. |

Intent-level pattern:

- `specific_track`: best source hit@20 `0.4902`, RRF hit@20 `0.2722`.
- `album`: best source hit@20 `0.5256`, RRF hit@20 `0.3209`.
- `artist_exploration`: best source hit@20 `0.4509`, RRF hit@20 `0.2366`.
- `cover_art`: best source hit@20 `0.3581`, RRF hit@20 `0.1730`.
- `lyrics_theme`: best source hit@20 `0.4124`, RRF hit@20 `0.1856`.
- `mood_playlist`: best source hit@20 `0.3658`, RRF hit@20 `0.1775`.

Interpretation:

- Added sources are finding gold tracks much more often than the legacy source alone.
- Current RRF fusion is the bottleneck because it preserves source coverage but pushes many gold tracks down.
- Legacy-vs-fused delta: RRF gained `446` top-20 hits and lost `212`, but demoted `642` legacy hits and promoted `723`; mean DCG delta@20 is `+0.0158` against the single legacy source, still far below the oracle source-selection ceiling.
- The next ranking experiment should be source gating, source-weight tuning, or LambdaRank features over per-source ranks rather than more uncalibrated retrieval channels.

## LTR Export

Script: `scripts/export_ltr_dataset.py`

Smoke run: `ltr_export_smoke`

- States: `24`
- Rows: `735`
- Groups with positive after `--include-missed-gold`: `24`
- Groups whose gold was outside the top-30 candidate pool before forced inclusion: `15`

This is intended as the handoff format for LightGBM/CatBoost: one JSONL row per `(session-turn, candidate_track)` with label, source ranks, RRF score, rule boost, intent/category/specificity, track priors, and seed/profile-derived features.

## Progress Label Audit

Script: `scripts/audit_progress_labels.py`

Smoke run: `progress_label_audit_smoke`

- Train sessions scanned: `15199`
- Turn records scanned: `121592`
- Missing labels: `15199`
- Turn 1 labels: all missing
- Turns 2-8 labels: `MOVES_TOWARD_GOAL` or `DOES_NOT_MOVE_TOWARD_GOAL`

Interpretation:

- `goal_progress_assessments[t]` is very likely feedback about the music recommendation at turn `t-1`, because the first recommendation has no same-turn label and the next user utterance carries the reaction.
- State construction now assigns historical music turn `m` the label from `progress[m + 1]`.
- This should be ablated against the previous same-turn assumption before changing any safe submission ranking.

## Current Blind A Packages

- Original safe package: `experiments/goalflow_safe_bm25_response/blindset_A/submission.zip`
- Shifted-feedback safe package: `experiments/goalflow_safe_bm25_response_shifted_feedback/blindset_A/submission.zip`
- Tail-diversity candidate package: `experiments/goalflow_taildiv_head15/blindset_A/submission.zip`
- Recommended current package: `experiments/goalflow_head20_compactresp_v2/blindset_A/submission.zip`
- Safer diversity backup: `experiments/goalflow_taildiv_head18_compactresp_v2/blindset_A/submission.zip`
- Aggressive diversity backup: `experiments/goalflow_taildiv_head15_compactresp_v2/blindset_A/submission.zip`

The `head20_compactresp_v2` package keeps `legacy_head_k=20` and therefore preserves the safe ranking anchor while replacing the weak response text with compact metadata-grounded explanations.

The `head18` tail-diversity package uses `legacy_head_k=18`, `tail_diversity_start=18`, and `global_repeat_penalty=0.06`. It changes only the final two positions and is the safer diversity experiment. The `head15` package is more aggressive and should be held back unless the first response-focused retry already succeeds.

## Embedding Schema

Script: `scripts/inspect_embeddings.py`

Artifact: `experiments/embedding_schema/summary.json`

Correct Challenge datasets:

- Track embeddings: `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings`
- User embeddings: `talkpl-ai/TalkPlayData-Challenge-User-Embeddings`

Do not use the older `TalkPlayData-2-Track-Embeddings` from the baseline tips for this challenge catalog: it has `0` overlap with the Challenge track UUIDs. The Challenge track embedding `all_tracks` split has `47071 / 47071` overlap with the current catalog.

Available track channels:

- `audio-laion_clap`: 512
- `image-siglip2`: 768
- `cf-bpr`: 128
- `attributes-qwen3_embedding_0.6b`: 1024
- `lyrics-qwen3_embedding_0.6b`: 1024
- `metadata-qwen3_embedding_0.6b`: 1024

## Pro Research Answers

Saved answers:

- `research/pro_answers/tab2_rrf_source-weight_regression_fix.txt`
- `research/pro_answers/tab3_goal_progress_assessment_validation.txt`
- `research/pro_answers/tab4_music_crs_implementation_plan.txt`
- `research/pro_answers/tab5_track_document_augmentation.txt`
- `research/pro_answers/tab6_recsys_2026_lambdarank_design.txt`
- `research/pro_answers/round2/tab1_source_gating_design.txt`
- `research/pro_answers/round2/tab2_embedding-based_extension_design.txt`
- `research/pro_answers/round2/tab3_recsys_challenge_2026_plan.txt`
- `research/pro_answers/round2/tab4_recsys_2026_response_generation.txt`
- `research/pro_answers/round2/tab5_recsys_2026_report_outline.txt`
- `research/pro_answers/round3/tab1_blind_postprocessing_strategy.txt`
- `research/pro_answers/round3/tab2_next_step_modeling_roi.txt`
- `research/pro_answers/round3/tab3_catalog_diversity_optimization.txt`
- `research/pro_answers/round3/tab4_dev_blind_evaluation_analysis.txt`
- `research/pro_answers/round3/tab5_metadata_grounded_response_design.txt`
- `research/pro_answers/round4/tab1_submission_package_decision.txt`

Operational takeaways:

- RRF regression should be attacked with source gating or LTR, not more uncalibrated sources.
- Progress labels should be shifted: label `t` is feedback for recommendation `t-1`.
- Embedding work should start with seed metadata/attributes/cf and user-cf channels using Challenge embeddings, with per-channel masks and normalization.
- Track document augmentation dev reporting must stay train-only; train+dev augmentation is only a separately marked final blind retrain after freezing choices.
- LambdaRank groups are `session_id × turn_number`, with binary exact-track labels and hard negatives from per-source top candidates.
- Public blind feedback shifts the immediate priority toward richer metadata-grounded response realization. Tail diversification is secondary on 80-row Blind A because catalog diversity has a low ceiling.
