# Blind A Submission Decision

Recommended first retry after LTR OOF validation:

```text
experiments/goalflow_ltr_head0_polished_v3/blindset_A/submission.zip
```

Why this one:

- Five-fold out-of-fold dev validation reaches official `nDCG@20=0.18095`, versus `0.08587` for the conservative legacy head20 dev baseline.
- Blind-A-shaped 500-panel validation gives mean nDCG@20 `0.16737`, with mean delta `+0.08088` over head20.
- Local Blind A catalog diversity is `0.03180`, close to the 80-row ceiling `0.03399`.
- Local Blind A Distinct-2 is `0.45124`; this is lower than compact-broad but the response is much more natural for the Gemini judge.
- It directly attacks the previous public weak points: `lexical_diversity=0.0125` and `llm_judge_score=1.0`, while also improving local ranking validation.

High-lexical LTR backup:

```text
experiments/goalflow_ltr_head0_compact_broad/blindset_A/submission.zip
```

Use this if the first LTR polished package unexpectedly loses on LLM judge or if maximizing Distinct-2 becomes the only goal. It has the same LTR ranking and local Blind A catalog diversity, but Distinct-2 is higher at `0.69996` and the response is more mechanical.

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
