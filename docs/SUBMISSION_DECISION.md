# Blind A Submission Decision

Recommended first retry after LTR tuning and response cleanup:

```text
experiments/goalflow_ltr120_lambda2_head0_judge_compact_mix_clean/blindset_A/submission.zip
```

Why this one:

- Five-fold out-of-fold dev validation reaches official `nDCG@20=0.18302`, versus `0.18210` for the unregularized 120-tree LTR, `0.18095` for the previous 260-tree LTR, and `0.08587` for the conservative legacy head20 dev baseline.
- Blind-A-shaped 500-panel validation gives mean nDCG@20 `0.16737`, with mean delta `+0.08088` over head20.
- Local Blind A catalog diversity is `0.03178`, close to the 80-row ceiling `0.03399`.
- Local Blind A Distinct-2 is `0.61160`, versus `0.52209` for `judge_mix` and `0.48531` for the fixed `judge_v2` style, while the ranking is identical.
- Official dev lexical diversity improves to `0.19493`, versus `0.15926` for `judge_mix` and `0.14874` for fixed `judge_v2`.
- The response cleanup removes private/noisy tag artifacts and title-cases profile fields before writing them into the explanation.
- It directly attacks the previous public weak points: `lexical_diversity=0.0125` and `llm_judge_score=1.0`, while also improving local ranking validation.

Lower-risk mixed text fallback:

```text
experiments/goalflow_ltr120_lambda2_head0_judge_mix_clean/blindset_A/submission.zip
```

Same ranking as the first package. It uses only judge-focused/natural/concise/setwise templates, so its Distinct-2 is lower (`0.52209`) but it is less mechanical than the compact mixed version.

Conservative fixed-style text fallback:

```text
experiments/goalflow_ltr120_lambda2_head0_judge_v2_clean/blindset_A/submission.zip
```

Same ranking as the first package, but it uses only the fixed `judge_v2` response template.

High-risk category-segmented LTR package:

```text
experiments/goalflow_segcat_ltr120_140_200_ens_judge_v2_clean/blindset_A/submission.zip
```

This chooses among the 120/140/200-tree L2 LTR models and their RRF ensemble by `conversation_goal.category`. It reaches official dev `nDCG@20=0.18407` if the segment map is selected on all OOF diagnostics, but stricter nested segment validation drops to about `0.18235`; use only as a high-risk experiment. Its high-lexical version is:

```text
experiments/goalflow_segcat_ltr120_140_200_ens_compact_broad_clean/blindset_A/submission.zip
```

High-lexical LTR backup:

```text
experiments/goalflow_ltr120_lambda2_head0_compact_broad_clean/blindset_A/submission.zip
```

Use this if the first LTR judge-v2 package unexpectedly loses on lexical diversity or if maximizing Distinct-2 becomes the only goal. It has the same best L2-regularized LTR ranking and local Blind A catalog diversity, but Distinct-2 is higher at `0.69667` and the response is more mechanical.

OOF-max ensemble backup:

```text
experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_compact_mix_clean/blindset_A/submission.zip
```

This RRF-ensembles the 120/140/200-tree L2 LTR packages with `rrf_k=60` and uses `judge_compact_mix` responses. It gives a small local OOF gain over the single 120-tree model (`nDCG@20=0.18325` versus `0.18302`) and official dev lexical diversity is `0.19455`, but Blind A unique-track coverage is slightly lower (`1494` vs `1496`). Treat it as a micro-gain experiment, not a risk-free replacement.

Fixed-style ensemble fallback:

```text
experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v2_clean/blindset_A/submission.zip
```

Lower-risk mixed ensemble fallback:

```text
experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_mix_clean/blindset_A/submission.zip
```

OOF-max high-lexical ensemble backup:

```text
experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_compact_broad_clean/blindset_A/submission.zip
```

Same ensemble ranking, with Distinct-2 `0.70020`.

Fuller-prose LLM-judge backups:

```text
experiments/goalflow_ltr120_lambda2_head0_judge_v3_clean/blindset_A/submission.zip
experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v3_clean/blindset_A/submission.zip
```

These use the same rankings as the `judge_v2` packages but longer, more natural explanations. They have lower local lexical diversity (`0.43434` / `0.43706` on Blind A), so use them only to test whether the LLM judge rewards fuller explanation quality more than Distinct-2.

Previous LTR backup:

```text
experiments/goalflow_ltr120_head0_judge_v2_clean/blindset_A/submission.zip
```

This was the previous 120-tree first-choice package. It is useful as a fallback, but its OOF ranking is slightly weaker (`nDCG@20=0.18210`) than the accepted L2 setting.

Older 260-tree LTR backup:

```text
experiments/goalflow_ltr_head0_polished_v3/blindset_A/submission.zip
```

This was the earlier first-choice package. It is still useful as a fallback, but its OOF ranking is weaker (`nDCG@20=0.18095`).

Rejected near-misses:

- `n_estimators=260`: stable but lower OOF than 120 trees.
- `num_leaves=63`: won the first held-out fold, but five-fold OOF dropped to `nDCG@20=0.18124`, so it did not generalize.
- `max_candidates_per_group=500`: added more candidates but lowered the main held-out score compared with max 300.
- `max_candidates_per_group=200`: reduced training noise but lost too many positives; held-out nDCG@20 fell to `0.18156` and valid positive groups dropped from `787` to `737`.
- Extra lexical/entity/year aggregate features: held-out nDCG@20 fell to `0.17941`, so the added features were noise.
- `min_child_samples=80`: won one held-out split, but five-fold OOF dropped to `nDCG@20=0.18114`.
- Row bagging/subsample values below `1.0`: once `subsample_freq=1` made row subsampling active, all tested values underperformed no-bagging.
- L2 `reg_lambda=0.1`: won the first held-out split, but five-fold OOF dropped to `nDCG@20=0.18154`; `reg_lambda=2` is the accepted L2 setting.
- L1 regularization: every tested `reg_alpha` value underperformed `reg_alpha=0` on the held-out split.
- More trees as a single model: 140/160/200 trees looked good on fold 0, but five-fold OOF was worse than 120 trees.
- `colsample_bytree=1.0` and `learning_rate=0.06`: both had small fold 0 gains and worse five-fold OOF.
- `lambdarank_truncation_level=100`: won fold 0 very slightly, but five-fold OOF dropped to `nDCG@20=0.18244`; default 30 remains best.
- Directly mixing labeled train-split sessions into the LTR training pool hurt held-out dev: 50 sampled sessions scored `0.18274` and 500 sampled sessions scored `0.17101` on the same fold where the 120-tree L2 dev-only model scored about `0.18489`. The optional code path remains for research, but it is not a submission setting.

Key correction:

Blind A has 80 rows and top-20 predictions, so the catalog-diversity ceiling is:

```text
1600 / 47071 = 0.033991
```

The previous public `catalog_diversity=0.0257` is about 76% of the unique-slot ceiling, not a catastrophic repeat problem.

Inferred composite formula:

```text
0.5 * nDCG@20
+ 0.1 * catalog_diversity
+ 0.1 * lexical_diversity
+ 0.3 * ((llm_judge_score - 1) / 4)
```

This exactly matches the reported composite `0.1006` from:

```text
nDCG@20=0.1935
catalog=0.0257
lexical=0.0125
llm_judge=1.0
```

Conservative legacy-rank backup:

```text
experiments/goalflow_head20_compact_broad/blindset_A/submission.zip
```

This keeps the same ranking strategy that already scored public nDCG@20 `0.1935` and changes only response text. Use it if LTR underperforms on public Blind A despite strong OOF validation.

Legacy middle backup:

```text
experiments/goalflow_taildiv_head19_compact_broad/blindset_A/submission.zip
```

Use it only after the response-focused package is tested. It changes only the final rank, improves Blind A unique tracks from `1216` to `1244`, and keeps full-dev nDCG@20 extremely close to safe (`0.08576` vs `0.08587`).

Stronger diversity backup:

```text
experiments/goalflow_taildiv_head18_compact_broad/blindset_A/submission.zip
```

This changes positions 19-20, improves Blind A unique tracks from `1216` to `1268`, and is favored by Blind-A-shaped dev panels. Use it when one more submission is available and the first response-focused retry confirms that ranking remains strong.

Hold back:

```text
experiments/goalflow_taildiv_head15_compactresp_v2/blindset_A/submission.zip
```

It has better diversity, but dev nDCG@20 drops to `0.08180`; the catalog gain is unlikely to pay for that risk on the public composite.
