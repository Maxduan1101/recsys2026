# Deep Research Backlog

This file tracks questions and optimization directions that deserve a dedicated research pass.

## High Priority

0. **RRF source-weight regression**
   - First GoalFlow v1 underperformed the single-source BM25 baseline (`nDCG@20=0.0679` vs `0.0859`).
   - GoalFlow v2 improved to `nDCG@20=0.0745` after adding the legacy source, but still diluted early ranks.
   - `head10` recovered most performance (`nDCG@20=0.0838`), while `head20` matched legacy ranking and improved lexical diversity.
   - Source diagnostics now confirm the hypothesis: `__best_single_source_rank__` reaches dev hit@20 `0.4715`, while current RRF reaches hit@20 `0.2595`.
   - Legacy-vs-fused delta diagnostic: RRF gains `446` top-20 hits and loses `212`, but demotes `642` legacy hits.
   - Pro answer saved at `research/pro_answers/tab2_rrf_source-weight_regression_fix.txt`.
   - Source-gated experiment `goalflow_gated_head5` reached dev nDCG@20 `0.0787` and catalog diversity `0.4712`, so the current hand-written gates are useful as instrumentation but not yet a safe replacement for legacy-head protection.
   - Additional Pro answer saved at `research/pro_answers/round2/tab1_source_gating_design.txt`.
   - Research/implementation need: source gating, intent-specific source weights, and/or learning-to-rank features over per-source ranks.

0.5. **Public blind diversity bottleneck**
   - Public Blind A submission reported by user: nDCG@20 `0.1935`, catalog diversity `0.0257`, lexical diversity `0.0125`, LLM judge `1.0`, composite `0.1006`.
   - This indicates the ranking anchor is valuable on blind data, but templated responses are the major immediate bottleneck.
   - New interpretation: Blind A has only 80 rows, so catalog diversity is capped at `1600 / 47071 = 0.0340`; the previous `0.0257` already represents about 76% unique slots.
   - Composite formula appears to be `0.5 * nDCG + 0.1 * catalog + 0.1 * lexical + 0.3 * ((judge - 1) / 4)`, inferred from the reported `0.1006`.
   - Implemented `goalflow_taildiv_head15`: protect first 15 ranks, diversify only positions 16-20 with global repeat penalties and artist/album soft caps.
   - Dev result: nDCG@20 `0.0818`, catalog diversity `0.7676`, lexical diversity `0.1019`.
   - Blind candidate package: `experiments/goalflow_taildiv_head15/blindset_A/submission.zip`.
   - Pro answers saved at `research/pro_answers/round3/tab1_blind_postprocessing_strategy.txt` and `research/pro_answers/round3/tab5_metadata_grounded_response_design.txt`.
   - Implemented compact response v2. Dev lexical diversity rises to `0.1758`; Blind A local Distinct-2 rises to `0.6654`.
   - Recommended first retry: `experiments/goalflow_head20_compactresp_v2/blindset_A/submission.zip`.
   - Safer diversity backup: `experiments/goalflow_taildiv_head18_compactresp_v2/blindset_A/submission.zip`; dev nDCG@20 `0.08527`, Blind A unique tracks `1268`.
   - Pro answer saved at `research/pro_answers/round4/tab1_submission_package_decision.txt`: submit head20 compact response first, not head15.
   - Implemented `compact_broad` response style. It keeps dev lexical diversity at `0.1757` while filtering bad/private tag artifacts that compact v2 allowed.
   - Current first-choice package: `experiments/goalflow_head20_compact_broad/blindset_A/submission.zip`.
   - Current middle backup: `experiments/goalflow_taildiv_head19_compact_broad/blindset_A/submission.zip`; dev nDCG@20 `0.08576`, Blind A unique tracks `1244`.
   - Current diversity backup: `experiments/goalflow_taildiv_head18_compact_broad/blindset_A/submission.zip`; Blind-A-shaped dev panels favor head18, while full dev favors head19.

1. **Progress-label semantics**
   - Is `goal_progress_assessment[turn_number]` judging the recommendation in the same turn, or the transition into the next turn?
   - Audited with `scripts/audit_progress_labels.py`: train turn 1 has no labels, while turns 2-8 have labels.
   - Sample conversations indicate label `t` is the user's reaction to the music at `t-1`.
   - Implementation updated so a historical music turn `m` uses `progress[m + 1]`.
   - Pro answer saved at `research/pro_answers/tab3_goal_progress_assessment_validation.txt`.
   - Next: run an ablation comparing old same-turn seed labeling vs shifted labeling on dev.

2. **Use of `conversation_goal` in blind submissions**
   - The field is present in released Blind A samples. Confirm whether official rules consider it a legitimate input or a synthetic-generation artifact that should be ablated.
   - Current implementation uses it but keeps the dependency explicit for ablations.

3. **Train-context item document augmentation leakage boundary**
   - For dev evaluation, using train-only augmentation is clean.
   - For Blind A/B, should dev labels be folded into augmented documents after local validation, or should we keep train-only for methodological safety?
   - Pro answer saved at `research/pro_answers/tab5_track_document_augmentation.txt`.
   - Current protocol: report dev scores with train-only augmentation; optionally build a separate final train+dev blind index only after all choices are frozen and clearly mark it as public-label retrain, not a dev result.

4. **Official embedding datasets schema and modality alignment**
   - Inspected with `scripts/inspect_embeddings.py`.
   - Correct datasets are `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings` and `talkpl-ai/TalkPlayData-Challenge-User-Embeddings`.
   - Warning: older `TalkPlayData-2-Track-Embeddings` has zero overlap with Challenge track UUIDs.
   - Implemented initial `goalflow/embeddings.py` store with per-channel raw/L2-normalized matrices and masks.
   - Pro answer saved at `research/pro_answers/tab4_music_crs_implementation_plan.txt`.
   - Additional Pro answer saved at `research/pro_answers/round2/tab2_embedding-based_extension_design.txt`.
   - Recommended first implementation: official embedding store with per-channel masks + L2 normalization, then seed_metadata, seed_attributes, seed_cf, and user_cf channels before lyrics/audio/image direct-query work.

5. **Candidate recall diagnostics**
   - Implemented in `scripts/diagnose_retrieval_sources.py`.
   - Full dev artifacts are in `experiments/source_diag_full/diagnostics/`.
   - Next: add source ablations that produce full predictions and evaluator nDCG, not only gold-rank diagnostics.
   - This decides whether to invest in dense retrieval or reranking first.

## Ranking Research

6. **LightGBM/CatBoost LambdaRank design**
   - Compare binary classification, pairwise ranker, and LambdaRank grouped by session-turn.
   - Hard negatives should include BM25, dense, same-artist wrong-track, same-album wrong-track, and popularity negatives.
   - First feature export is implemented in `scripts/export_ltr_dataset.py`; it writes JSONL rows grouped by dev session-turn with source-rank, RRF, rule boost, profile, seed, and track prior features.
   - Pro answer saved at `research/pro_answers/round2/tab3_recsys_challenge_2026_plan.txt`.
   - Round 3 Pro answer saved at `research/pro_answers/round3/tab2_next_step_modeling_roi.txt`: first fix response/diversity and use LTR after the safe public retry.
   - Round 5 Pro answer saved at `research/pro_answers/round5/tab3_ranker_implementation_guidance.txt`: LTR should initially protect legacy top ranks and act as a tail reranker/tie-breaker, not a wholesale replacement.
   - Next: decide feature matrix encoding, candidate sampling, and whether missed gold rows should be included for training or only diagnostics.

7. **Cross-encoder reranking**
   - Compare `bge-reranker-v2-m3`, DeBERTa, and Qwen reranker-style LoRA for top-100 reranking.
   - Determine whether cross-encoder is worth the latency for Blind B.

8. **Entity extraction strategy**
   - The current system uses lexical heuristics. Research whether a small LLM parser materially improves exact title/artist/album recognition.

9. **Feedback seed modeling**
   - Current positive/negative seed boosts are metadata/tag based. Research multi-modal seed similarity and how to handle user phrases like "closer but more energetic."

10. **RRF and boost calibration**
   - Tune index/query weights and boost magnitudes with proper dev splits and intent-specific analysis.

## Multi-Modal Retrieval Research

11. **Official `metadata-qwen3`, `lyrics-qwen3`, and `attributes-qwen3` embeddings**
   - Decide whether to use dot product, cosine, whitening, or learned fusion.

12. **Audio CLAP retrieval**
   - Determine whether text-query-to-audio embedding is possible with a compatible CLAP text encoder, or whether audio embedding should be used only for seed similarity.

13. **Image/SigLIP cover-art retrieval**
   - Test cover-art queries and whether public SigLIP text encoder shares the same embedding space as released `image-siglip2`.

14. **CF-BPR user-track scoring**
   - Evaluate warm/cold user coverage and whether user profile + CF improves broad mood queries.

## Response And Diversity

15. **LLM-as-a-Judge response optimization**
   - Establish prompts and self-critique criteria for personalization and explanation quality without hallucinating metadata.
   - Implemented metadata-grounded response variants keyed by session/turn. This raises dev lexical diversity from `0.0830` to `0.1019`.
   - Implemented compact response v2 with current-query snippets, profile cues, feedback cues, album/year/tag grounding, and tag profanity filtering. This raises dev lexical diversity to `0.1758` and Blind A local Distinct-2 to `0.6654` on the 80-row split.
   - Implemented `compact_broad`, `compact`, `concise`, `setwise`, and `natural` response styles. `compact_broad` is the current submission default; `natural` and `setwise` were more judge-friendly in tone but lowered Distinct-2 to roughly `0.10-0.12`.
   - Pro answers saved at `research/pro_answers/round2/tab4_recsys_2026_response_generation.txt` and `research/pro_answers/round3/tab5_metadata_grounded_response_design.txt`.
   - Additional Pro answer saved at `research/pro_answers/round5/tab1_response_generator_design.txt`.

16. **Distinct-2 vs. naturalness tradeoff**
   - Template diversity can inflate lexical diversity but may hurt Gemini quality. Need controlled eval.

17. **Global catalog-diversity post-processing**
   - Current top-5 is protected and positions 6-20 are lightly diversified. Research stronger xQuAD/MMR settings without damaging nDCG.
   - `taildiv_head10` reached dev catalog diversity `0.8323` but nDCG@20 only `0.0721`.
   - `taildiv_head15` reached dev catalog diversity `0.7676` with nDCG@20 `0.0818`; this is the current best diversity/ranking tradeoff candidate.
   - `taildiv_head18_compactresp_v2` reached dev nDCG@20 `0.08527` and catalog diversity `0.6143`; it is a safer backup than head15.
   - `taildiv_head19_compact_clean` reached dev nDCG@20 `0.08576` and catalog diversity `0.5084`; it is the lower-risk middle backup.
   - Blind-like 500-panel validation: head18 mean delta `+0.00070` nDCG@20 vs head20, head19 `+0.00031`, head17 `-0.00174`.
   - On Blind A, catalog diversity gains are small in composite terms because the 80-row ceiling is only `0.0340`; do not sacrifice head ranking for catalog unless public feedback specifically shows ranking is unchanged.
   - Pro answer saved at `research/pro_answers/round3/tab3_catalog_diversity_optimization.txt`.
   - Additional Pro answer saved at `research/pro_answers/round5/tab2_low_risk_ranking_improvements.txt`.

## Evaluation Research

21. **Dev/blind mismatch**
   - Local dev nDCG around `0.08` aligns with the official baseline expectation, while public Blind A feedback showed `0.1935`.
   - Possible causes include Blind A containing only selected/prefix turns, distribution shift, or different public split difficulty.
   - Round 3 Pro answer saved at `research/pro_answers/round3/tab4_dev_blind_evaluation_analysis.txt`.
   - Implemented `scripts/evaluate_blind_like.py`, which samples dev panels matching Blind A `(turn_number, category, specificity)` strata and reports mean/p10/median nDCG deltas.
   - Round 5 Pro answer saved at `research/pro_answers/round5/tab5_offline_validation_design.txt`.

## Engineering

18. **FAISS/NumPy vector backend**
   - Implement shared vector retrieval once embedding schema is confirmed.

19. **Experiment tracking**
   - Add reproducible run manifests, source recall metrics, and automated score tables.

20. **Submission validation against Codabench**
   - Confirm whether submission zip requires only `prediction.json` or a nested path for all phases.
