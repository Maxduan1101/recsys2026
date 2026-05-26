Round 11 Final Triage
=====================

Saved answers:

- `tab1_lexplus_softened_decision.txt`: `lexplus_softened` should supersede plain `lexplus` as the response-only backup, but the primary submission should still be weighted four-model RRF + `judge_clean_mix` because the softened variant may be more compact-template-heavy.
- `tab2_case_branch_stop_decision.txt`: after the negative case-neighbor probe, do not code feature-only case distillation now. Reopen only if a revised case retriever reaches much higher exact-track/artist coverage and near-neutral protected insertion.
- `tab5_final_ranking_stop_decision.txt`: freeze all changes that alter `predicted_track_ids`. Direct Qwen3 current-query dense retrieval is no-go at final stage. Only response-only variants and packaging/validity checks remain allowed.

Final local decision:

1. Primary package remains `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_clean/blindset_A/submission.zip`.
2. Strongest response-only backup is now `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_lexplus_softened/blindset_A/submission.zip`.
3. Plain `lexplus` is obsolete because `lexplus_softened` has better dev lexical, better Blind A Distinct-2, and no long response rows.
4. Ranking-side experimentation is frozen unless public feedback gives a clear reason to reopen it.
