Round 10 Pro/Web Status
=======================

Saved answers:

- `tab1_cross_encoder_audition.txt`: cross-encoder/top-50 reranking is worth only one protected offline audition, with a submission bar around `+0.004` OOF nDCG@20 and no top-5/top-10 damage.
- `tab2_direct_query_embedding_path.txt`: the safest direct dense path is current-query text encoded with `Qwen/Qwen3-Embedding-0.6B` against official `attributes-qwen3`, `metadata-qwen3`, and gated `lyrics-qwen3` track vectors. It must pass dimension, normalization, synthetic title/artist, BM25 agreement, and protected-head gates before label tuning.
- `tab3_direct_embedding_guardrails.txt`: dense retrieval should be a guarded recall source or weak feature, not a new head ranker.
- `tab4_consensus_fallback_calibration.txt`: keep the consensus fallback as a rare future Blind-B repair rule, with extra RRF-rank/margin gates and trigger-case diagnostics. It is not evidence for Blind A because it changes `0 / 80` rows.
- `tab5_final_iteration_decision_tree.txt`: keep weighted four-model RRF + `judge_clean_mix` as the primary package; treat `lexplus` as a response-only challenger; stop large new ranker/embedding/index changes unless public feedback forces that direction.

Follow-up questions sent after local probing:

- `tab1_ce_negative_stop_decision.txt`: after the negative local MiniLM probe, Pro recommends stopping CE reranking as a submission path.
- Tab 2 now asks whether case-based train-turn distillation can still help after train-context augmentation and LTR/RRF.
