# GoalFlow Results

Development-set scores from the official evaluator.

| Run | nDCG@1 | nDCG@10 | nDCG@20 | Catalog Diversity | Lexical Diversity | Notes |
|---|---:|---:|---:|---:|---:|---|
| `bm25_static_devset` | 0.009625 | 0.067601 | 0.085870 | 0.388966 | 0.000125 | Strong copied BM25-history baseline. |
| `goalflow_bm25_aug_v1` | 0.011250 | 0.040984 | 0.067874 | 0.478872 | 0.082518 | Multi-source RRF without legacy head protection. Better diversity, worse ranking. |
| `goalflow_bm25_aug_v2` | 0.012125 | 0.047612 | 0.074497 | 0.481932 | 0.082975 | Added high-weight legacy source, still diluted head ranks. |
| `goalflow_bm25_aug_v3_head10` | 0.009625 | 0.067601 | 0.083776 | 0.451021 | 0.082975 | Preserves legacy top 10, lets GoalFlow change tail. Slight ranking loss. |
| `goalflow_bm25_aug_v3_head20` | 0.009625 | 0.067601 | 0.085870 | 0.388966 | 0.082975 | Safe legacy baseline: legacy ranking + GoalFlow response. |
| `goalflow_gated_head5` | 0.009875 | 0.055209 | 0.078712 | 0.471161 | 0.073882 | Source-gated fusion. Higher diversity, but still loses too much ranking. |
| `goalflow_taildiv_head10` | 0.009875 | 0.067536 | 0.072093 | 0.832253 | 0.101913 | Aggressive tail diversification after rank 10. Useful stress test, too much nDCG loss. |
| `goalflow_taildiv_head15` | 0.009875 | 0.067536 | 0.081796 | 0.767628 | 0.101913 | Historical diversity candidate: preserves first 15, diversifies last 5. |
| `goalflow_head20_compactresp_v2` | 0.009625 | 0.067601 | 0.085870 | 0.388966 | 0.175800 | Historical response-first candidate: unchanged safe ranking, compact grounded response. |
| `goalflow_taildiv_head18_compactresp_v2` | 0.009875 | 0.067536 | 0.085271 | 0.614327 | 0.175789 | Safer diversity backup: preserves first 18, diversifies final 2. |
| `goalflow_taildiv_head15_compactresp_v2` | 0.009875 | 0.067536 | 0.081796 | 0.767628 | 0.175789 | More aggressive backup; not recommended before head20/head18 because nDCG loss is larger. |
| `goalflow_head20_style_compact_broad` | 0.009625 | 0.067601 | 0.085870 | 0.388966 | 0.175710 | Historical best response-only style: compact v2 diversity with broader cleaned tags. |
| `goalflow_taildiv_head19_compact_clean` | 0.009875 | 0.067536 | 0.085758 | 0.508402 | 0.169888 | Conservative tail-diversity variant; only rank 20 can be diversified. |
| `goalflow_taildiv_head19_compact_broad` | 0.009875 | 0.067536 | 0.085758 | 0.508402 | 0.175699 | Legacy middle backup with compact broad responses. |
| `goalflow_taildiv_head18_compact_broad` | 0.009875 | 0.067536 | 0.085271 | 0.614327 | 0.175699 | Legacy stronger diversity backup with compact broad responses. |
| `goalflow_taildiv_head17_compact_clean` | 0.009875 | 0.067536 | 0.084303 | 0.688598 | 0.169888 | Too much nDCG loss; held back. |
| `goalflow_head20_cf_tail19` | 0.009625 | 0.067601 | 0.085927 | 0.392216 | 0.175710 | Experimental seed-CF tail rescue; tiny full-dev gain, blind-like neutral. |
| `goalflow_taildiv_head19_cf_tail19` | 0.009875 | 0.067536 | 0.085701 | 0.504982 | 0.175699 | CF rescue stacked on head19 was worse than head19 alone. |
| `goalflow_taildiv_head18_cf_tail18` | 0.009875 | 0.067536 | 0.085299 | 0.611077 | 0.175699 | CF rescue stacked on head18 gives tiny nDCG gain but lower diversity. |
| `goalflow_ltr_head0_oof_compact_broad` | 0.067500 | 0.159372 | 0.180947 | 0.520958 | 0.220660 | Five-fold out-of-fold LTR validation; each dev row scored by a model that did not train on it. |
| `goalflow_ltr_head0_oof_polished_v3` | 0.067500 | 0.159372 | 0.180947 | 0.520958 | 0.136947 | Same OOF LTR ranking, more natural response text for LLM judge. |
| `goalflow_ltr120_head0_oof_polished_v1` | 0.069250 | 0.161358 | 0.182098 | 0.526524 | 0.137308 | Better LTR tree count: 120 estimators, lr 0.04, 31 leaves. |
| `goalflow_ltr120_head0_oof_judge_v2_clean` | 0.069250 | 0.161358 | 0.182098 | 0.526524 | 0.148765 | Same improved LTR ranking, shorter judge-focused responses after tag/profile cleanup. |
| `goalflow_ltr120_lr004_leaves63_head0_oof_judge_v2` | 0.068875 | 0.160685 | 0.181239 | 0.531516 | 0.148933 | Rejected: single-fold looked better, but five-fold OOF was worse than 31 leaves. |
| `goalflow_ltr120_lambda01_head0_oof_judge_v2` | 0.068250 | 0.161462 | 0.181537 | 0.527097 | 0.148856 | Rejected: light L2 won fold 0 but lost five-fold OOF. |
| `goalflow_ltr120_lambda2_head0_oof_judge_v2` | 0.070875 | 0.162514 | 0.183021 | 0.528011 | 0.148741 | Current best single-model OOF ranking: 120 estimators, 31 leaves, `reg_lambda=2`. |
| `goalflow_ltr140_lambda2_head0_oof_judge_v2` | 0.070000 | 0.162808 | 0.182530 | 0.525440 | 0.148565 | Rejected: 140 trees won fold 0 but lost five-fold OOF. |
| `goalflow_ltr160_lambda2_head0_oof_judge_v2` | 0.069625 | 0.163259 | 0.182413 | 0.523380 | 0.148486 | Rejected: 160 trees did not generalize. |
| `goalflow_ltr200_lambda2_head0_oof_judge_v2` | 0.070875 | 0.163578 | 0.182725 | 0.522232 | 0.148377 | Rejected: 200 trees had the best fold 0 score but worse OOF. |
| `goalflow_ltr120_lambda2_col1_head0_oof_judge_v2` | 0.070000 | 0.160575 | 0.182011 | 0.528776 | 0.148899 | Rejected: all-column sampling hurt OOF despite a small fold 0 win. |
| `goalflow_ltr120_lr006_lambda2_head0_oof_judge_v2` | 0.069250 | 0.161655 | 0.181458 | 0.523040 | 0.148666 | Rejected: higher learning rate overfit fold 0. |
| `goalflow_ens_oof_ltr120_140_200_lambda2_rrf60` | 0.071000 | 0.163316 | 0.183253 | 0.525695 | 0.148741 | Micro-gain ensemble: only `+0.00023` over the single 120-tree model. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_rrf60` | 0.071000 | 0.162823 | 0.183482 | 0.525547 | 0.148741 | New best validated ensemble: adds the all-column 120-tree L2 variant to the RRF pool. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_rrf20` | 0.072000 | 0.163043 | 0.183727 | 0.525844 | 0.148741 | Previous weighted-RRF best: `rrf_k=20`, weights `[1.0, 0.5, 1.0, 1.0]`. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26` | 0.072000 | 0.163392 | 0.183924 | 0.525547 | 0.148741 | Current weighted-RRF best: `rrf_k=26`, weights `[1.0, 0.5, 1.3, 1.0]`. |
| `goalflow_ltr120_lambda2_head0_oof_judge_v3` | 0.070875 | 0.162514 | 0.183021 | 0.528011 | 0.125937 | Same ranking as `judge_v2`; more complete prose but lower lexical diversity. |
| `goalflow_ltr120_lambda2_head0_oof_judge_mix` | 0.070875 | 0.162514 | 0.183021 | 0.528011 | 0.159260 | Same ranking as `judge_v2`; mixed response templates improve official lexical diversity. |
| `goalflow_ltr120_lambda2_head0_oof_judge_brief_probe` | 0.070875 | 0.162514 | 0.183021 | 0.528011 | 0.174312 | Same ranking as `judge_v2`; shorter grounded response probe. |
| `goalflow_ltr120_lambda2_head0_oof_judge_compact_mix_probe` | 0.070875 | 0.162514 | 0.183021 | 0.528011 | 0.194935 | Compact/broad mixed response probe. |
| `goalflow_ltr120_lambda2_head0_oof_judge_clean_mix` | 0.070875 | 0.162514 | 0.183021 | 0.528011 | 0.200611 | Current single-model text choice: cleaner high-lexical mix of judge/brief/compact responses after title display cleanup. |
| `goalflow_ltr120_lambda2_head0_oof_compact_probe` | 0.070875 | 0.162514 | 0.183021 | 0.528011 | 0.209822 | Clean compact response: higher lexical, more template-like. |
| `goalflow_ltr120_lambda2_head0_oof_compact_broad_probe` | 0.070875 | 0.162514 | 0.183021 | 0.528011 | 0.220792 | Same ranking; highest lexical among grounded templates but more mechanical. |
| `goalflow_ens_oof_ltr120_140_200_lambda2_rrf60_judge_mix` | 0.071000 | 0.163316 | 0.183253 | 0.525695 | 0.159014 | Same ensemble ranking; mixed response templates improve lexical diversity. |
| `goalflow_ens_oof_ltr120_140_200_lambda2_rrf60_judge_compact_mix` | 0.071000 | 0.163316 | 0.183253 | 0.525695 | 0.194546 | Same ensemble ranking with the compact mixed response style. |
| `goalflow_ens_oof_ltr120_140_200_lambda2_rrf60_judge_clean_mix` | 0.071000 | 0.163316 | 0.183253 | 0.525695 | 0.199012 | Same ensemble ranking with the clean mixed response style. |
| `goalflow_ens_oof_ltr120_140_200_lambda2_rrf60_compact_clean` | 0.071000 | 0.163316 | 0.183253 | 0.525695 | 0.209428 | Ensemble ranking with clean compact response. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_rrf60_judge_clean_mix` | 0.071000 | 0.162823 | 0.183482 | 0.525547 | 0.199655 | Four-model ensemble ranking with clean mixed response style after title/artist/album display cleanup. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_rrf60_judge_balanced_mix` | 0.071000 | 0.162823 | 0.183482 | 0.525547 | 0.176244 | Same four-model ranking; more natural explanation mix, lower lexical diversity after album display cleanup. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_rrf60_judge_planned` | 0.071000 | 0.162823 | 0.183482 | 0.525547 | 0.114233 | Same four-model ranking; strongest prose-style explanation backup, much lower lexical diversity. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_rrf60_compact_clean` | 0.071000 | 0.162823 | 0.183482 | 0.525547 | 0.211125 | Four-model ensemble ranking with clean compact response after title/artist/album display cleanup. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_rrf60_compact_broad_clean` | 0.071000 | 0.162823 | 0.183482 | 0.525547 | 0.222801 | Four-model ensemble ranking with highest-lexical compact-broad response after title/artist/album display cleanup. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_rrf20_judge_clean_mix` | 0.072000 | 0.163043 | 0.183727 | 0.525844 | 0.199574 | Previous weighted four-model RRF with clean mixed response. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_rrf20_compact_clean` | 0.072000 | 0.163043 | 0.183727 | 0.525844 | 0.211007 | Same weighted ranking with clean compact high-lexical response. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_rrf20_compact_broad_clean` | 0.072000 | 0.163043 | 0.183727 | 0.525844 | 0.222685 | Same weighted ranking with highest-lexical compact-broad response. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix` | 0.072000 | 0.163392 | 0.183924 | 0.525547 | 0.199581 | Weighted four-model RRF with clean mixed response; current best OOF candidate. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_single_fallback_top1_script` | 0.072125 | 0.163487 | 0.183986 | 0.525547 | 0.199581 | Experimental consensus fallback: 52 dev rows switch to the single 120-tree model when weighted top1 has no component-top1 support. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_tagclean` | 0.072000 | 0.163392 | 0.183924 | 0.525547 | 0.199573 | Same primary response style with stricter noisy-tag filtering; Blind A artifact is identical to the primary package. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_safeplus` | 0.072000 | 0.163392 | 0.183924 | 0.525547 | 0.191884 | Rejected clean-plus attempt: no broad tags, but lexical diversity fell below the primary. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_lexplus_tagclean` | 0.072000 | 0.163392 | 0.183924 | 0.525547 | 0.204602 | Clean high-lexical response backup: same ranking, more compact metadata-grounded wording, no broad-tag source. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_lexplus_softened` | 0.072000 | 0.163392 | 0.183924 | 0.525547 | 0.205313 | Softened high-lexical response backup: same ranking as the primary, but long lexplus responses fall back to shorter grounded wording. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_respblend_r25` | 0.072000 | 0.163392 | 0.183924 | 0.525547 | 0.200886 | Rejected response-only blend: 25% softened responses, below `lexplus_softened`. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_respblend_r50` | 0.072000 | 0.163392 | 0.183924 | 0.525547 | 0.202473 | Rejected response-only blend: 50% softened responses, below `lexplus_softened`. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_respblend_r75` | 0.072000 | 0.163392 | 0.183924 | 0.525547 | 0.203737 | Rejected response-only blend: 75% softened responses, below `lexplus_softened`. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_respblend_opening` | 0.072000 | 0.163392 | 0.183924 | 0.525547 | 0.192786 | Rejected response-only blend: replacing repeated primary openings lowered lexical below the primary. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_respblend_hybrid` | 0.072000 | 0.163392 | 0.183924 | 0.525547 | 0.204196 | Rejected response-only blend: closest blend result, still below `lexplus_softened`. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_compact_clean` | 0.072000 | 0.163392 | 0.183924 | 0.525547 | 0.211093 | Same weighted ranking with clean compact high-lexical response. |
| `goalflow_ens_oof_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_compact_broad_clean` | 0.072000 | 0.163392 | 0.183924 | 0.525547 | 0.222793 | Same weighted ranking with highest-lexical compact-broad response. |
| `goalflow_segcat_ltr120_140_200_ens_judge_v2` | 0.072125 | 0.164281 | 0.184069 | 0.526481 | 0.148494 | High-risk segment-selection experiment: best non-nested OOF score, but nested segment validation regresses. |

Immediate interpretation:

- Goal/document/augmentation sources are not yet calibrated enough to improve ranking.
- They do improve response diversity and candidate diversity.
- Real Blind A feedback suggests the conservative ranking anchor is strong, while response quality/diversity is the biggest immediate score bottleneck.
- Blind A has only 80 rows, so catalog diversity is capped at `1600 / 47071 = 0.0340`; improving catalog by a few hundred unique tail tracks has a small composite effect compared with preserving nDCG and improving response text.
- LightGBM LambdaRank is now the strongest local ranking path. The current best single model is 120 estimators / 31 leaves / learning rate 0.04 / L2 `reg_lambda=2`; same-family RRF ensembling gives a tiny gain.
- Category-segmented model selection reaches the best non-nested OOF score. It chooses `ens` for category A, `ltr140` for C/H/I/J, `ltr200` for G/K, and `ltr120` otherwise. A stricter nested segment-selection check drops to about `0.18235`, so it is high-risk and not the recommended first retry.
- The remaining high-impact work is response judge calibration, better candidate recall without head-rank dilution, and richer dense/embedding features.

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
- Pro sanity check saved at `research/pro_answers/round4/tab1_submission_package_decision.txt`: submit the safest head20 response-first package before diversity-heavy variants.
- Latest response iteration upgrades the response style to `compact_broad`, which keeps compact v2 lexical diversity while filtering noisy/private tag artifacts.

## Blind A Local Diversity Summaries

These are gold-free checks from `scripts/summarize_predictions.py`; they do not estimate nDCG.

| Package | Unique Tracks | Unique Slot Ratio | Catalog Diversity | Catalog Ceiling | Distinct-2 | Recommendation |
|---|---:|---:|---:|---:|---:|---|
| `goalflow_head20_compactresp_v2` | 1216 | 0.7600 | 0.025833 | 0.033991 | 0.665424 | Submit first: preserves known ranking. |
| `goalflow_taildiv_head18_compactresp_v2` | 1268 | 0.7925 | 0.026938 | 0.033991 | 0.665424 | Backup if one more submission is available. |
| `goalflow_taildiv_head15_compactresp_v2` | 1348 | 0.8425 | 0.028638 | 0.033991 | 0.665424 | Useful experiment, but nDCG risk is likely too high for first retry. |
| `goalflow_head20_compact_broad` | 1216 | 0.7600 | 0.025833 | 0.033991 | 0.664176 | Previous response-only first choice: same safe ranking, cleaner broad-tag response. |
| `goalflow_taildiv_head19_compact_broad` | 1244 | 0.7775 | 0.026428 | 0.033991 | 0.664176 | Middle backup: small tail diversity gain, lower full-dev nDCG risk than head18. |
| `goalflow_taildiv_head18_compact_broad` | 1268 | 0.7925 | 0.026938 | 0.033991 | 0.664176 | Stronger diversity backup; blind-like panels favor it, full-dev nDCG is lower. |
| `goalflow_head20_cf_tail19` | 1217 | 0.7606 | 0.025855 | 0.033991 | 0.664176 | Experimental only: seed-CF changes just 1 Blind A row. |
| `goalflow_ltr_head0_compact_broad` | 1497 | 0.9356 | 0.031803 | 0.033991 | 0.699961 | Same LTR ranking as current candidate; higher Distinct-2, more mechanical response. |
| `goalflow_ltr_head0_polished_v3` | 1497 | 0.9356 | 0.031803 | 0.033991 | 0.451235 | Previous LTR first choice: LTR ranking plus more judge-friendly natural response. |
| `goalflow_ltr120_head0_judge_v2_clean` | 1500 | 0.9375 | 0.031867 | 0.033991 | 0.488329 | Previous 120-tree first choice before L2 regularization. |
| `goalflow_ltr120_head0_compact_broad_clean` | 1500 | 0.9375 | 0.031867 | 0.033991 | 0.702228 | Previous high-lexical backup before L2 regularization. |
| `goalflow_ltr120_lambda2_head0_judge_v2_clean` | 1496 | 0.9350 | 0.031782 | 0.033991 | 0.485312 | Conservative single-model package: best fixed LTR ranking plus concise judge-focused response. |
| `goalflow_ltr120_lambda2_head0_judge_mix_clean` | 1496 | 0.9350 | 0.031782 | 0.033991 | 0.522089 | Lower-risk mixed text backup: same ranking as judge-v2, more varied judge-focused responses. |
| `goalflow_ltr120_lambda2_head0_judge_clean_mix_clean` | 1496 | 0.9350 | 0.031782 | 0.033991 | 0.604534 | Conservative single-model text fallback: same ranking, cleaner high-lexical mixing after title display cleanup. |
| `goalflow_ltr120_lambda2_head0_judge_compact_mix_clean` | 1496 | 0.9350 | 0.031782 | 0.033991 | 0.611604 | Compact/broad mixed backup: slightly higher Blind A Distinct-2 but lower official dev lexical. |
| `goalflow_ltr120_lambda2_head0_compact_clean` | 1496 | 0.9350 | 0.031782 | 0.033991 | 0.675327 | Same L2 ranking; clean high-lexical template backup. |
| `goalflow_ltr120_lambda2_head0_compact_broad_clean` | 1496 | 0.9350 | 0.031782 | 0.033991 | 0.696674 | Same L2 ranking; high-lexical backup if Distinct-2 matters more than naturalness. |
| `goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v2_clean` | 1494 | 0.9338 | 0.031739 | 0.033991 | 0.485312 | OOF-max ensemble backup: slightly better dev OOF, slightly lower Blind A coverage. |
| `goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_mix_clean` | 1494 | 0.9338 | 0.031739 | 0.033991 | 0.524761 | Same ensemble ranking; mixed judge-focused responses. |
| `goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_clean_mix_clean` | 1494 | 0.9338 | 0.031739 | 0.033991 | 0.610336 | Same ensemble ranking; clean mixed responses. |
| `goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_compact_mix_clean` | 1494 | 0.9338 | 0.031739 | 0.033991 | 0.615890 | Same ensemble ranking; compact/broad mixed responses. |
| `goalflow_ens_ltr120_140_200_lambda2_rrf60_compact_clean` | 1494 | 0.9338 | 0.031739 | 0.033991 | 0.678379 | Same ensemble ranking; clean high-lexical template backup. |
| `goalflow_ens_ltr120_140_200_lambda2_rrf60_compact_broad_clean` | 1494 | 0.9338 | 0.031739 | 0.033991 | 0.700198 | Same ensemble ranking; high-lexical backup. |
| `goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_clean_mix_clean` | 1495 | 0.9344 | 0.031761 | 0.033991 | 0.605192 | Equal-weight four-model ensemble fallback: strong OOF nDCG, middle Blind A coverage, cleaned duplicate/list-valued display titles, artists, and albums. |
| `goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_balanced_mix` | 1495 | 0.9344 | 0.031761 | 0.033991 | 0.557186 | Same ranking; natural explanation backup if LLM judge rewards prose more than Distinct-2. |
| `goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_planned` | 1495 | 0.9344 | 0.031761 | 0.033991 | 0.398543 | Same ranking; highest-prose, low-lexical LLM-judge experiment. |
| `goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_compact_clean` | 1495 | 0.9344 | 0.031761 | 0.033991 | 0.673512 | Four-model ensemble with clean high-lexical template backup after title/artist/album display cleanup. |
| `goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_compact_broad_clean` | 1495 | 0.9344 | 0.031761 | 0.033991 | 0.695210 | Four-model ensemble with highest-lexical compact-broad template backup after title/artist/album display cleanup. |
| `goalflow_ens_ltr120_140_200_col1_lambda2_w140half_rrf20_judge_clean_mix_clean` | 1495 | 0.9344 | 0.031761 | 0.033991 | 0.606673 | Weighted four-model RRF primary candidate; 63/80 Blind A rows differ from equal-weight but only 2 top-1 changes. |
| `goalflow_ens_ltr120_140_200_col1_lambda2_w140half_rrf20_compact_clean` | 1495 | 0.9344 | 0.031761 | 0.033991 | 0.673990 | Same weighted ranking; clean compact high-lexical backup. |
| `goalflow_ens_ltr120_140_200_col1_lambda2_w140half_rrf20_compact_broad_clean` | 1495 | 0.9344 | 0.031761 | 0.033991 | 0.695922 | Same weighted ranking; highest-lexical backup. |
| `goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_clean` | 1494 | 0.9338 | 0.031739 | 0.033991 | 0.606040 | Current weighted four-model RRF primary candidate; 66/80 rows differ from equal-weight but only 1 top-1 change. |
| `goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_lexplus_tagclean` | 1494 | 0.9338 | 0.031739 | 0.033991 | 0.631916 | Clean high-lexical response backup; same ranking as the primary package, more compact-template-heavy. |
| `goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_lexplus_softened` | 1494 | 0.9338 | 0.031739 | 0.033991 | 0.635660 | Softened high-lexical response backup; same ranking as the primary package, no long response rows in the local audit. |
| `goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_compact_clean` | 1494 | 0.9338 | 0.031739 | 0.033991 | 0.673790 | Same weighted ranking; clean compact high-lexical backup. |
| `goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_compact_broad_clean` | 1494 | 0.9338 | 0.031739 | 0.033991 | 0.695481 | Same weighted ranking; highest-lexical backup. |
| `goalflow_ltr120_lambda2_head0_judge_v3_clean` | 1496 | 0.9350 | 0.031782 | 0.033991 | 0.434335 | Same 120-tree L2 ranking; fuller prose for LLM-judge testing, lower Distinct-2 than v2. |
| `goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v3_clean` | 1494 | 0.9338 | 0.031739 | 0.033991 | 0.437063 | Same ensemble ranking; fuller prose backup. |
| `goalflow_segcat_ltr120_140_200_ens_judge_v2_clean` | 1496 | 0.9350 | 0.031782 | 0.033991 | 0.487628 | High-risk category-segmented LTR selection with judge-v2 responses. |
| `goalflow_segcat_ltr120_140_200_ens_compact_broad_clean` | 1496 | 0.9350 | 0.031782 | 0.033991 | 0.698188 | Same category-segmented ranking; high-lexical backup. |

## Blind-Like Dev Panels

Script: `scripts/evaluate_blind_like.py`

This samples 80-row dev panels matching Blind A's `(turn_number, category, specificity)` distribution, then compares prediction files against a baseline. It is not a substitute for Codabench, but it catches changes that look good on full dev while being risky for the Blind A-shaped slice.

Latest 500-panel result, baseline `head20`:

| Candidate | Mean nDCG@20 | Median nDCG@20 | P10 nDCG@20 | Mean Delta vs Head20 | Median Delta | Notes |
|---|---:|---:|---:|---:|---:|---|
| `head20` | 0.086485 | 0.085973 | 0.065782 | 0.000000 | 0.000000 | Safe ranking anchor. |
| `head18` | 0.087183 | 0.086981 | 0.066606 | +0.000698 | +0.000437 | Best blind-like tail-diversity signal. |
| `head19` | 0.086794 | 0.086925 | 0.065330 | +0.000309 | +0.000182 | More conservative middle option. |
| `head17` | 0.084748 | 0.085102 | 0.062943 | -0.001737 | -0.001721 | Rejected. |
| `ltr_oof` | 0.167368 | 0.166469 | 0.132476 | +0.080883 | +0.080768 | OOF LTR head0, evaluated on Blind-A-shaped panels. |

CF-tail blind-like follow-up:

- `goalflow_head20_cf_tail19` improved full-dev nDCG@20 by `+0.000057`, but the 500-panel Blind-A-shaped sample was exactly neutral versus head20.
- On Blind A, it changed only `1 / 80` row, raising unique tracks from `1216` to `1217`.
- Stacking CF-tail on `head19` reduced dev nDCG; stacking it on `head18` raised nDCG slightly but lowered catalog diversity. Do not prioritize it over the three main packages.
- Applying the same rank-20 seed-CF rescue to the current four-model ensemble changed `397 / 8000` dev rows and only `1 / 80` Blind A row. Official full-dev nDCG@20 stayed exactly `0.183482` while catalog diversity rose to `0.527798`, but 500 Blind-A-shaped panels had mean delta `-0.00036`; keep this as a rejected shadow experiment.
- Applying a batch-level repeat repair to the current weighted four-model ensemble is also rejected. The safest `head_k=19` repair improves dev catalog diversity by `+0.01476` but lowers OOF nDCG@20 from `0.183924` to `0.183497`; more aggressive `head_k=15-18` variants lose more nDCG for a small extra catalog gain.
- A narrower RRF grid around the current weighted package found `rrf_k=28`, weights `[1.0, 0.45, 1.45, 1.05]`, but official OOF only reached `0.183966` and Blind-A-shaped 1000-panel mean delta was `-0.000096`; reject it as dev-fold noise.
- `scripts/apply_consensus_fallback.py` implements a conservative top-1 support fallback. The first rule (`support_top_k=1`, `base_max_support=0`, `fallback_min_support=2`) raises official OOF to `0.183986` and Blind-A-shaped 5000-panel mean delta to `+0.000278` with p10 `0.0`, but it changes `0 / 80` Blind A rows. Keep it as a Blind-B/future-split tool rather than a replacement for the current Blind A package.
- `scripts/probe_cross_encoder_zero_shot.py` implements a protected zero-shot cross-encoder probe. On the 160-turn dev cache with `cross-encoder/ms-marco-MiniLM-L-6-v2` over top-50 candidates, the current weighted RRF base scored `0.29136` nDCG@20 on that slice, while CE-only/lock0 scored `0.14385`, lock5 `0.24440`, lock10 `0.27000`, and lock15 `0.28224`. Even top-15 protection is `-0.00912` nDCG@20 with only `3` better rows versus `11` worse rows. Round 10 Pro follow-up recommends stopping CE reranking as a submission path, so no full-dev CE run is planned.
- `scripts/probe_case_neighbors.py` implements a cheap train-turn case-neighbor source probe. On a train-1000-session / dev-160-turn sample with BM25 top-20 cases, gold track appeared among neighbor gold tracks for only `6.25%` of turns and the gold artist for `28.13%`. Direct protected insertion is strongly negative: case-only `-0.26481`, lock5 `-0.07614`, lock10 `-0.03824`, and lock15 `-0.01788` nDCG@20 versus the current base on that slice. Stop case-neighbor direct candidate promotion; only keep case features as future research if time allows a proper fold-local LTR feature test.

## Response Text Audit

Script: `scripts/audit_response_text.py`

Gold-free Blind A checks:

- Primary `judge_clean_mix`: 80 rows, audit Distinct-2 `0.56394` with this script's tokenizer, word counts `41-112` with average `75.88`, no noisy phrase hits, no long/short rows.
- Lexplus backup: 80 rows, audit Distinct-2 `0.58673`, word counts `41-120` with average `72.05`, no noisy phrase hits, one long row at 120 words.
- Lexplus softened backup: 80 rows, audit Distinct-2 `0.58996`, word counts `41-112` with average `71.24`, no noisy phrase hits, no long/short rows.
- This audit is a response QA guard, not an official metric replacement. It catches banned/noisy tag leaks, overly long/short responses, and repeated opening phrases before spending a public submission.

## Response-Only Blender

Script: `scripts/blend_response_predictions.py`

The blender creates response-only artifacts from the primary and `lexplus_softened` predictions, refusing to run if ranking payload hashes differ. Five dev probes kept nDCG/catalog unchanged but all lexical scores stayed below `lexplus_softened`; keep the script as tooling, not as a promoted backup.

## Prediction Hashes

Script: `scripts/hash_predictions.py`

Primary and `lexplus_softened` Blind A artifacts have identical ranking hashes, confirming the backup is response-only:

```text
ranking_sha256 = b83bd55e238206fb22eeca4e950b2930d82d4762c178c4bf1a82b3c0df1a6f27
primary_response_sha256 = 8e3ce4fa69ce4669673e4894f115822d2ad4cf8df9774e77a1d057264472670b
lexplus_softened_response_sha256 = ef6739fc8cf568ed8055cd613973b330d326f94499708603b9124915d0744b28
```

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

## LightGBM LTR Reranking

Scripts:

- `scripts/probe_lgbm_ltr.py`
- `scripts/run_ltr_rerank.py`

Key validation:

- Candidate pool: top 300 fused candidates per dev turn.
- Model: LightGBM LambdaRank, binary exact-track labels, group = session turn.
- Training excludes dev rows whose gold track is not naturally present in the candidate pool.
- Original 260-tree five-fold OOF official dev score: `nDCG@20=0.180947`, catalog diversity `0.520958`.
- Tuned 120-tree five-fold OOF official dev score: `nDCG@20=0.182098`, catalog diversity `0.526524`.
- L2-regularized 120-tree five-fold OOF official dev score: `nDCG@20=0.183021`, catalog diversity `0.528011`; this is the current conservative single-model ranking choice.
- Blind-A-shaped 500-panel mean nDCG@20: `0.167368` versus `0.086485` for the legacy head20 baseline.
- Blind A local summary for `goalflow_ltr_head0_polished_v3`: 1497 unique tracks out of 1600 recommendation slots, catalog diversity `0.031803`, Distinct-2 `0.451235`.
- Blind A local summary for `goalflow_ltr120_head0_judge_v2_clean`: 1500 unique tracks out of 1600 recommendation slots, catalog diversity `0.031867`, Distinct-2 `0.488329`.
- Blind A local summary for `goalflow_ltr120_lambda2_head0_judge_v2_clean`: 1496 unique tracks out of 1600 recommendation slots, catalog diversity `0.031782`, Distinct-2 `0.485312`.
- Max-500 candidate expansion was rejected: held-out head0 nDCG@20 `0.182574`, below the max-300 single-fold score `0.184317` for the 120-tree model.
- Max-200 candidate reduction was rejected: held-out head0 nDCG@20 `0.181556`, and valid groups with positive candidates dropped from `787` to `737`.
- Extra lexical/entity/year aggregate features were rejected: held-out nDCG@20 fell to `0.179408`.
- A larger 63-leaf model was rejected after OOF: it won the first held-out fold (`0.184676` vs `0.184317`) but lost overall (`0.181239` vs `0.182098`).
- L2 `reg_lambda=0.1` was rejected after OOF: fold 0 improved to `0.184929`, but five-fold official `nDCG@20` was `0.181537`.
- L2 `reg_lambda=2` was accepted: five-fold official `nDCG@20` improved to `0.183021`.
- L1 regularization was rejected: all tested `reg_alpha` values underperformed `reg_alpha=0` on fold 0.
- Extra tree counts `140`, `160`, and `200` were rejected as single models: all won or nearly won fold 0 but lost five-fold OOF.
- `colsample_bytree=1.0` and `learning_rate=0.06` were rejected after OOF despite small fold 0 gains.
- `lambdarank_truncation_level=100` was rejected after OOF: fold 0 improved slightly, but official five-fold `nDCG@20` dropped to `0.182444`.
- RRF ensembling over the 120/140/200-tree L2 OOF predictions gives a small local gain, official `nDCG@20=0.183253`; adding the all-column 120-tree L2 variant improves the equal-weight ensemble to `0.183482`.
- Weighted RRF (`rrf_k=26`, weights `[1.0, 0.5, 1.3, 1.0]`) improves the four-model ensemble to official OOF `nDCG@20=0.183924`; Blind-A-shaped validation versus equal-weight is positive, with the 2000-panel run showing mean delta `+0.00162` and median delta `+0.00062`.
- Position-band weighted RRF was rejected: equal-head/weighted-tail and weighted-head/equal-tail variants did not beat the global weighted RRF, whose score remains `0.183923885`.
- Anchored weighted RRF was also rejected for promotion: adding a tiny equal-weight anchor (`k=26`, anchor `k=60`, `lambda=0.05`) reached `0.183956`, only `+0.000032` over the current package and far below the `+0.0005` acceptance bar for a new ranking package.
- Optional embedding LTR features are now implemented behind `--embedding-features`, but the first fold rejects promoting them: `track_cf + user_cf` scored `0.18341`, metadata seed cosine scored `0.18301`, and attributes seed cosine scored `0.17892` versus the stronger accepted 120-tree L2 baseline. Keep this as a research/probe path, not a submission path.
- `judge_mix` response style was added as a lower-risk text-upside package. It keeps the same ranking as `judge_v2`, raises official dev lexical diversity from `0.14874` to `0.15926`, and raises Blind A local Distinct-2 from `0.48531` to `0.52209`.
- `judge_brief`, `judge_compact_mix`, and `judge_clean_mix` were added after the response probe. `judge_brief` reaches official dev lexical `0.17431`; `judge_compact_mix` reaches `0.19493`; after duplicate/list-valued title, artist, and album display cleanup, `judge_clean_mix` reaches `0.19966` on the four-model ensemble ranking. Blind A local Distinct-2 is `0.60519` for the clean mix.
- `judge_v3` response style was added as a fuller explanation style. It may help LLM-as-a-Judge because it reads more naturally, but local lexical diversity is lower than `judge_v2`, so it is only a text backup.
- `judge_planned` and `judge_balanced_mix` were added after the Pro response-judge pass. They preserve the same ranking and make the explanations more natural; after album-display cleanup, `judge_balanced_mix` keeps usable lexical diversity (`dev 0.17624`, Blind A `0.55719`), while `judge_planned` is more prose-like but too low-lexical for a first submission.
- `judge_clean_mix_plus` was added after Pro round 8 as a response-only backup. It preserves the weighted RRF ranking, has official dev lexical `0.19928`, Blind A Distinct-2 `0.60729`, and no quick banned-tag/opening-duplication flags. It is not promoted above `judge_clean_mix` because it sometimes reintroduces less-clean broad tags such as `songsof2011` or `lobpreis`.

Current submission recommendation:

1. `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_clean/blindset_A/submission.zip`
2. `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_plus/blindset_A/submission.zip`
3. `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_rrf20_judge_clean_mix_clean/blindset_A/submission.zip`
4. `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_clean_mix_clean/blindset_A/submission.zip`
5. `experiments/goalflow_ltr120_lambda2_head0_judge_clean_mix_clean/blindset_A/submission.zip`
6. `experiments/goalflow_ltr120_lambda2_head0_judge_mix_clean/blindset_A/submission.zip`
7. `experiments/goalflow_ltr120_lambda2_head0_judge_v2_clean/blindset_A/submission.zip`
8. `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_balanced_mix/blindset_A/submission.zip`
9. `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_compact_clean/blindset_A/submission.zip`
10. `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_compact_broad_clean/blindset_A/submission.zip`
11. `experiments/goalflow_ltr120_lambda2_head0_judge_compact_mix_clean/blindset_A/submission.zip`
12. `experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_clean_mix_clean/blindset_A/submission.zip`
13. `experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_mix_clean/blindset_A/submission.zip`
14. `experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v2_clean/blindset_A/submission.zip`
15. `experiments/goalflow_ltr120_lambda2_head0_compact_clean/blindset_A/submission.zip`
16. `experiments/goalflow_ltr120_lambda2_head0_compact_broad_clean/blindset_A/submission.zip`
17. `experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_compact_clean/blindset_A/submission.zip`
18. `experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_compact_broad_clean/blindset_A/submission.zip`
19. `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_planned/blindset_A/submission.zip`
20. `experiments/goalflow_ltr120_lambda2_head0_judge_v3_clean/blindset_A/submission.zip`
21. `experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v3_clean/blindset_A/submission.zip`
22. `experiments/goalflow_ltr120_head0_judge_v2_clean/blindset_A/submission.zip`
23. `experiments/goalflow_ltr_head0_polished_v3/blindset_A/submission.zip`
24. `experiments/goalflow_head20_compact_broad/blindset_A/submission.zip`

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
- Historical response-only package: `experiments/goalflow_head20_compactresp_v2/blindset_A/submission.zip`
- Safer diversity backup: `experiments/goalflow_taildiv_head18_compactresp_v2/blindset_A/submission.zip`
- Aggressive diversity backup: `experiments/goalflow_taildiv_head15_compactresp_v2/blindset_A/submission.zip`
- Current weighted four-model ensemble text choice: `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_judge_clean_mix_clean/blindset_A/submission.zip`
- Previous weighted four-model ensemble text fallback: `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_rrf20_judge_clean_mix_clean/blindset_A/submission.zip`
- Equal-weight four-model ensemble text fallback: `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_judge_clean_mix_clean/blindset_A/submission.zip`
- Conservative single-model clean-mix fallback: `experiments/goalflow_ltr120_lambda2_head0_judge_clean_mix_clean/blindset_A/submission.zip`
- Lower-risk mixed-response single-model backup: `experiments/goalflow_ltr120_lambda2_head0_judge_mix_clean/blindset_A/submission.zip`
- Compact/broad mixed-response single-model backup: `experiments/goalflow_ltr120_lambda2_head0_judge_compact_mix_clean/blindset_A/submission.zip`
- Conservative fixed-style single-model package: `experiments/goalflow_ltr120_lambda2_head0_judge_v2_clean/blindset_A/submission.zip`
- Previous three-model clean-mixed-response ensemble backup: `experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_clean_mix_clean/blindset_A/submission.zip`
- Previous three-model compact/broad-mixed-response ensemble backup: `experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_compact_mix_clean/blindset_A/submission.zip`
- Previous three-model lower-risk mixed-response ensemble backup: `experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_mix_clean/blindset_A/submission.zip`
- Previous three-model fixed-style ensemble backup: `experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v2_clean/blindset_A/submission.zip`
- Clean compact high-lexical LTR backup: `experiments/goalflow_ltr120_lambda2_head0_compact_clean/blindset_A/submission.zip`
- Current high-lexical LTR backup: `experiments/goalflow_ltr120_lambda2_head0_compact_broad_clean/blindset_A/submission.zip`
- Weighted clean compact high-lexical backup: `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_compact_clean/blindset_A/submission.zip`
- Weighted highest-lexical ensemble backup: `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_w140half_w20013_rrf26_compact_broad_clean/blindset_A/submission.zip`
- Equal-weight clean compact high-lexical backup: `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_compact_clean/blindset_A/submission.zip`
- Equal-weight highest-lexical ensemble backup: `experiments/goalflow_ens_ltr120_140_200_col1_lambda2_rrf60_compact_broad_clean/blindset_A/submission.zip`
- Previous three-model clean compact high-lexical backup: `experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_compact_clean/blindset_A/submission.zip`
- Previous three-model high-lexical ensemble backup: `experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_compact_broad_clean/blindset_A/submission.zip`
- High-risk category-segmented package: `experiments/goalflow_segcat_ltr120_140_200_ens_judge_v2_clean/blindset_A/submission.zip`
- Category-segmented high-lexical backup: `experiments/goalflow_segcat_ltr120_140_200_ens_compact_broad_clean/blindset_A/submission.zip`
- Fuller-prose single-model backup: `experiments/goalflow_ltr120_lambda2_head0_judge_v3_clean/blindset_A/submission.zip`
- Fuller-prose ensemble backup: `experiments/goalflow_ens_ltr120_140_200_lambda2_rrf60_judge_v3_clean/blindset_A/submission.zip`
- Previous 120-tree LTR first-choice package: `experiments/goalflow_ltr120_head0_judge_v2_clean/blindset_A/submission.zip`
- Previous LTR first-choice package: `experiments/goalflow_ltr_head0_polished_v3/blindset_A/submission.zip`
- Current conservative legacy-rank backup: `experiments/goalflow_head20_compact_broad/blindset_A/submission.zip`
- Current legacy middle backup: `experiments/goalflow_taildiv_head19_compact_broad/blindset_A/submission.zip`
- Current legacy diversity backup: `experiments/goalflow_taildiv_head18_compact_broad/blindset_A/submission.zip`
- Experimental CF-tail package: `experiments/goalflow_head20_cf_tail19/blindset_A/submission.zip`

The `head20_compactresp_v2` package keeps `legacy_head_k=20` and therefore preserves the old safe ranking anchor while replacing the weak response text with compact metadata-grounded explanations. It is now a legacy-family fallback behind the LTR packages above.

The `head18` tail-diversity package uses `legacy_head_k=18`, `tail_diversity_start=18`, and `global_repeat_penalty=0.06`. It changes only the final two positions and is the safer diversity experiment. The `head15` package is more aggressive and should be held back unless the first response-focused retry already succeeds.

The `compact_broad` response style is the high-lexical backup for submissions. It keeps compact v2's high lexical diversity, blocks offensive/private tag artifacts such as `albums i own`, `seen live`, `lastfm`, and profanity-like tag variants, and avoids the lower Distinct-2 of the more natural long-response experiments.

The current RRF ensemble uses `scripts/ensemble_predictions.py` with the 120/140/200-tree L2 LTR packages plus the weaker 120-tree all-column L2 variant. Equal weights with `rrf_k=60` improve local OOF to `nDCG@20=0.183482`; the new weighted setting (`rrf_k=26`, weights `[1.0, 0.5, 1.3, 1.0]`) improves it again to `0.183924` and is positive on Blind-A-shaped panels, so it is now the first-choice package. The previous weighted, equal-weight four-model, and older three-model RRF packages remain fallbacks.

The category-segmented package uses `scripts/select_segmented_predictions.py` to choose among the single LTR models and the RRF ensemble by `conversation_goal.category`. It has the best non-nested official dev OOF score, `nDCG@20=0.184069`, and Blind A local diversity matches the single 120-tree package. A stricter nested segment-selection check, where four folds choose the segment map and the held-out fold evaluates it, drops to about `0.18235`; keep it as a high-risk experiment rather than a primary package.

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
- `research/pro_answers/round5/tab1_response_generator_design.txt`
- `research/pro_answers/round5/tab2_low_risk_ranking_improvements.txt`
- `research/pro_answers/round5/tab3_ranker_implementation_guidance.txt`
- `research/pro_answers/round5/tab4_embedding_cf_integration.txt`
- `research/pro_answers/round5/tab5_offline_validation_design.txt`
- `research/pro_answers/round6/tab1_rank20_rescue.txt`
- `research/pro_answers/round6/tab2_response_judge_optimization.txt`
- `research/pro_answers/round6/tab3_embedding_cf_shadow.txt`
- `research/pro_answers/round6/tab4_shadow_blind_ci.txt`
- `research/pro_answers/round6/tab5_next_best_target.txt`
- `research/pro_answers/round7/tab1_response_submission_choice.txt`
- `research/pro_answers/round7/tab2_low_risk_weighted_rrf.txt`
- `research/pro_answers/round7/tab3_train_context_usage.txt`
- `research/pro_answers/round7/tab4_embedding_query_features.txt`
- `research/pro_answers/round7/tab5_small_blind_validation.txt`
- `research/pro_answers/round8/tab1_submission_decision.txt`
- `research/pro_answers/round8/tab2_remaining_low_risk_ranking.txt`
- `research/pro_answers/round8/tab3_catalog_diversity_stop_or_tail.txt`
- `research/pro_answers/round8/tab4_embedding_ltr_features.txt`
- `research/pro_answers/round8/tab5_response_style_next.txt`
- `research/pro_answers/round9/tab1_response_lexplus_decision.txt`
- `research/pro_answers/round9/tab5_final_submission_sequence.txt`
- `research/pro_answers/round9/README.md`
- `research/pro_answers/round10/tab1_cross_encoder_audition.txt`
- `research/pro_answers/round10/tab1_ce_negative_stop_decision.txt`
- `research/pro_answers/round10/tab2_direct_query_embedding_path.txt`
- `research/pro_answers/round10/tab3_direct_embedding_guardrails.txt`
- `research/pro_answers/round10/tab4_consensus_fallback_calibration.txt`
- `research/pro_answers/round10/tab5_final_iteration_decision_tree.txt`
- `research/pro_answers/round10/tab6_case_based_distillation.txt`
- `research/pro_answers/round10/README.md`
- `research/pro_answers/round11/tab1_lexplus_softened_decision.txt`
- `research/pro_answers/round11/tab2_case_branch_stop_decision.txt`
- `research/pro_answers/round11/tab5_final_ranking_stop_decision.txt`
- `research/pro_answers/round11/README.md`
- `research/pro_answers/round12/tab1_response_blender_decision.txt`
- `research/pro_answers/round12/README.md`

Operational takeaways:

- RRF regression should be attacked with source gating or LTR, not more uncalibrated sources.
- Progress labels should be shifted: label `t` is feedback for recommendation `t-1`.
- Embedding work should start with seed metadata/attributes/cf and user-cf channels using Challenge embeddings, with per-channel masks and normalization.
- Track document augmentation dev reporting must stay train-only; train+dev augmentation is only a separately marked final blind retrain after freezing choices.
- LambdaRank groups are `session_id × turn_number`, with binary exact-track labels and hard negatives from per-source top candidates.
- Public blind feedback shifts the immediate priority toward richer metadata-grounded response realization. Tail diversification is secondary on 80-row Blind A because catalog diversity has a low ceiling.
- Round 5 Pro answers reinforce a conservative strategy: protect the BM25 head, use extra retrieval/ranking only as tail evidence, and validate changes on Blind-A-shaped panels.
- Round 7 Pro answers support `judge_clean_mix` over compact-broad for the next real submission, recommend anchored global weighted RRF as the lowest-risk ranking improvement, and warn that train/embedding signals should enter as gated/tail evidence until they pass strict blind-like uncertainty checks.
- Round 8 Pro answers still favor the weighted RRF ranking and warn against ranking-diversity tweaks on the tiny Blind A ceiling. They suggest response-only `judge_clean_mix_plus` as a possible judge-quality backup, but the local artifact check keeps plain `judge_clean_mix` first.
- Round 9 final-submission guidance still favors submitting the weighted RRF + `judge_clean_mix` package first. The response-style retry also keeps `judge_clean_mix` ahead of `lexplus` for the first submission because a small Distinct-2 gain is easier to lose through Gemini judge naturalness. Other round 9 browser questions are still pending or failed with empty ChatGPT responses, so they are recorded for retry rather than treated as research evidence.
- Round 10 Pro answers keep the current primary package unchanged, demote cross-encoder work to a strict offline-only audition, recommend any direct dense query retrieval go through Qwen3-compatible text channels with protected-head gates, and keep the consensus fallback only as a rare future-split repair.
- Round 11 Pro answers freeze all ranking changes for the final stage. They promote `lexplus_softened` over plain `lexplus` as the response-only backup, reject feature-only case distillation after the negative case probe, and mark direct Qwen3 current-query dense retrieval as no-go for the final package.
- Round 12 Pro answer rejects response blenders after local blend probes underperformed `lexplus_softened` on lexical diversity. The final response set is therefore the primary `judge_clean_mix` package plus the `lexplus_softened` response-only challenger.
