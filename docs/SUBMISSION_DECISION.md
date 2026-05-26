# Blind A Submission Decision

Recommended first retry after LTR tuning, response cleanup, and ensemble probing:

```text
experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_clean/blindset_A/submission.zip
```

Why this one:

- Weighted four-model RRF OOF dev validation reaches official `nDCG@20=0.18392`, versus `0.18348` for the equal-weight four-model ensemble, `0.18325` for the previous 120/140/200-tree ensemble, `0.18302` for the single 120-tree L2 model, and `0.08587` for the conservative legacy head20 dev baseline.
- It ensembles the 120/140/200-tree `reg_lambda=2` LTR rankings plus the 120-tree `colsample_bytree=1.0` L2 variant with `rrf_k=26` and weights `[1.0, 0.5, 1.3, 1.0]`.
- Blind-A-shaped validation also favors it over the equal-weight four-model package: the 2000-panel run has mean delta `+0.00162` nDCG@20 and median delta `+0.00062`.
- Local Blind A catalog diversity is `0.03176`, close to the 80-row ceiling `0.03399`.
- Local Blind A Distinct-2 is `0.60604`, versus `0.48531` for the fixed `judge_v2` style.
- Official dev lexical diversity is `0.19958`, versus `0.14874` for fixed `judge_v2`.
- The response cleanup removes private/noisy tag artifacts, title-cases profile fields, and collapses duplicate/list-valued track titles, artist names, and album names before writing them into the explanation.
- It directly attacks the previous public weak points: `lexical_diversity=0.0125` and `llm_judge_score=1.0`, while also improving local ranking validation.

Clean high-lexical response backup:

```text
experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_lexplus_tagclean/blindset_A/submission.zip
```

This keeps the exact same weighted RRF ranking and uses only cleaned `judge_v2`, `judge_brief`, and compact metadata-grounded templates. It raises official dev lexical diversity to `0.20460` and local Blind A Distinct-2 to `0.63192`, with the same local Blind A catalog diversity `0.03174`. Keep it as a response-only backup rather than the default first submission because it is more compact-template-heavy than `judge_clean_mix`, so Gemini may or may not prefer it.

Response-only judge-quality backup:

```text
experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_plus/blindset_A/submission.zip
```

This keeps the exact same weighted RRF ranking but mixes a small amount of fuller/natural and compact-broad wording into `judge_clean_mix`. Official dev lexical is `0.19928` and local Blind A Distinct-2 is `0.60729`. Keep it behind the primary clean package because spot checks found occasional less-clean broad tags, even though no quick banned-term/opening-duplication flags fired.

Conservative single-model clean-mix fallback:

```text
experiments/goalflow_ltr120_lambda2_head0_judge_clean_mix_clean/blindset_A/submission.zip
```

Use this if the submission budget favors the simplest validated LTR path. Five-fold out-of-fold dev validation reaches official `nDCG@20=0.18302`, with local Blind A catalog diversity `0.03178` and Distinct-2 `0.60453`.

Future-split consensus fallback:

```text
experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_single_fallback_top1_judge_clean_mix_script/blindset_A/submission.zip
```

This currently changes `0 / 80` Blind A rows, so it is identical to the primary package for Blind A. The script is still useful for Blind B or future splits: on dev it changes 52 rows, improves official OOF nDCG@20 to `0.183986`, and has positive Blind-A-shaped 5000-panel mean delta `+0.000278` with p10 `0.0`.

Equal-weight four-model fallback:

```text
experiments/goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_clean_mix_clean/blindset_A/submission.zip
```

Use this if the weighted RRF feels too tuned for local OOF. It has official dev `nDCG@20=0.18348`, local Blind A Distinct-2 `0.60519`, and changes fewer ranking positions relative to the previous first-choice package.

Lower-risk mixed text fallback:

```text
experiments/goalflow_ltr120_lambda2_head0_judge_mix_clean/blindset_A/submission.zip
```

Same single-model ranking as the conservative clean-mix fallback. It uses only judge-focused/natural/concise/setwise templates, so its Distinct-2 is lower (`0.52209`) but it is less compact-template-heavy than the clean mixed version.

Natural explanation ensemble fallback:

```text
experiments/goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_balanced_mix/blindset_A/submission.zip
```

Same four-model ensemble ranking as the primary package, but with a more natural response mix informed by the Pro response-judge pass. It has official dev lexical `0.17624` and local Blind A Distinct-2 `0.55719` after album-display cleanup, so use it only if the LLM judge seems to reward prose quality more than Distinct-2.

Conservative fixed-style text fallback:

```text
experiments/goalflow_ltr120_lambda2_head0_judge_v2_clean/blindset_A/submission.zip
```

Same single-model ranking as the conservative clean-mix fallback, but it uses only the fixed `judge_v2` response template.

High-risk category-segmented LTR package:

```text
experiments/goalflow_segcat_ltr120_140_200_ens_judge_v2_clean/blindset_A/submission.zip
```

This chooses among the 120/140/200-tree L2 LTR models and their RRF ensemble by `conversation_goal.category`. It reaches official dev `nDCG@20=0.18407` if the segment map is selected on all OOF diagnostics, but stricter nested segment validation drops to about `0.18235`; use only as a high-risk experiment. Its high-lexical version is:

Blind-A-shaped 500-panel validation also keeps it below the four-model ensemble: mean delta `-0.00215` nDCG@20 versus `goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_clean_mix_clean`.

```text
experiments/goalflow_segcat_ltr120_140_200_ens_compact_broad_clean/blindset_A/submission.zip
```

High-lexical LTR backup:

```text
experiments/goalflow_ltr120_lambda2_head0_compact_clean/blindset_A/submission.zip
```

Use this if the clean mixed package appears too cautious on lexical diversity and the LLM judge tolerates concise metadata-grounded templates. It has official dev lexical diversity `0.20982` and local Blind A Distinct-2 `0.67533`.

Highest-lexical LTR backup:

```text
experiments/goalflow_ltr120_lambda2_head0_compact_broad_clean/blindset_A/submission.zip
```

Use this if maximizing Distinct-2 becomes the only goal. It has the same best L2-regularized LTR ranking and local Blind A catalog diversity, but Distinct-2 is higher at `0.69667` and the response is more mechanical.

Previous three-model ensemble backup:

```text
experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_clean_mix_clean/blindset_A/submission.zip
```

This RRF-ensembles the 120/140/200-tree L2 LTR packages with `rrf_k=60` and uses `judge_clean_mix` responses. It gives a small local OOF gain over the single 120-tree model (`nDCG@20=0.18325` versus `0.18302`) and official dev lexical diversity is `0.19901`, but it is now behind the four-model ensemble.

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
experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_compact_clean/blindset_A/submission.zip
```

Same weighted four-model ensemble ranking, with official dev lexical `0.21109` and local Blind A Distinct-2 `0.67379` after album-display cleanup.

OOF-max highest-lexical ensemble backup:

```text
experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_compact_broad_clean/blindset_A/submission.zip
```

Same weighted four-model ensemble ranking, with official dev lexical `0.22279` and local Blind A Distinct-2 `0.69548` after album-display cleanup. Use only if the leaderboard strongly rewards Distinct-2 and the LLM judge tolerates compact metadata-heavy wording.

Previous three-model high-lexical ensemble backup:

```text
experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_compact_clean/blindset_A/submission.zip
```

Same previous three-model ensemble ranking, with official dev lexical `0.20943` and local Blind A Distinct-2 `0.67838`.

Previous three-model highest-lexical ensemble backup:

```text
experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_compact_broad_clean/blindset_A/submission.zip
```

Same ensemble ranking, with Distinct-2 `0.70020`.

Fuller-prose LLM-judge backups:

```text
experiments/goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_planned/blindset_A/submission.zip
experiments/goalflow_ltr120_lambda2_head0_judge_v3_clean/blindset_A/submission.zip
experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v3_clean/blindset_A/submission.zip
```

These use unchanged rankings but longer, more natural explanations. `judge_planned` has the lowest lexical diversity (`dev 0.11423`, Blind A `0.39854`) but reads closest to a direct explanation; use these only to test whether the LLM judge rewards fuller explanation quality more than Distinct-2.

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
- Position-band weighted RRF did not beat the accepted global weighted RRF; the best variants tied the same `0.183923885` score.
- Anchored weighted RRF found only a microscopic gain (`0.183956`, `+0.000032`) and is below the acceptance bar for a new ranking package.
- Optional embedding LTR features are implemented but rejected for promotion. Fold-0 probes scored `0.18341` for `track_cf + user_cf`, `0.18301` for metadata seed cosine, and `0.17892` for attributes seed cosine, all below the accepted baseline path.
- Ultra-conservative batch-level repeat repair is rejected. Even the safest variant, which freezes ranks 1-19 and repairs only rank 20, lowers OOF nDCG@20 to `0.183497` while buying catalog diversity that has little Blind A headroom left.
- `judge_clean_mix_safeplus` is rejected for promotion: removing broad/noisy tags was safe, but the style mix lowered official dev lexical diversity to `0.19188` and Blind A Distinct-2 to `0.60487`, both below the primary `judge_clean_mix` package.

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
