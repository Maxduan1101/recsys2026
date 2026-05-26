Round 10 Pro/Web Status
=======================

Saved answers:

- `tab1_cross_encoder_audition.txt`: cross-encoder/top-50 reranking is worth only one protected offline audition, with a submission bar around `+0.004` OOF nDCG@20 and no top-5/top-10 damage.
- `tab2_direct_query_embedding_path.txt`: the safest direct dense path is current-query text encoded with `Qwen/Qwen3-Embedding-0.6B` against official `attributes-qwen3`, `metadata-qwen3`, and gated `lyrics-qwen3` track vectors. It must pass dimension, normalization, synthetic title/artist, BM25 agreement, and protected-head gates before label tuning.
- `tab3_direct_embedding_guardrails.txt`: dense retrieval should be a guarded recall source or weak feature, not a new head ranker.
- `tab4_consensus_fallback_calibration.txt`: keep the consensus fallback as a rare future Blind-B repair rule, with extra RRF-rank/margin gates and trigger-case diagnostics. It is not evidence for Blind A because it changes `0 / 80` rows.
- `tab5_final_iteration_decision_tree.txt`: keep weighted four-model RRF + `judge_clean_mix` as the primary package; treat `lexplus` as a response-only challenger; stop large new ranker/embedding/index changes unless public feedback forces that direction.
- `tab6_case_based_distillation.txt`: case-neighbor signal is worth at most one minimal feature-only experiment, and should not become a new retrieval backbone.

Follow-up questions sent after local probing:

- `tab1_ce_negative_stop_decision.txt`: after the negative local MiniLM probe, Pro recommends stopping CE reranking as a submission path.

Local follow-up probes:

- `scripts/probe_cross_encoder_zero_shot.py`: MiniLM CE over top-50 on 160 dev turns is negative even with top-15 protection.
- `scripts/probe_case_neighbors.py`: BM25 train-turn case neighbors over 1000 train sessions are negative as a direct protected candidate source; this stops case-source promotion before LTR integration.
- `judge_clean_mix_lexplus_softened`: response-only backup created after round 10 guidance. It keeps the same ranking, raises dev lexical to `0.20531`, raises Blind A Distinct-2 to `0.63566`, and removes the previous lexplus long response row.
