# Blind A Submission Decision

Recommended first retry:

```text
experiments/goalflow_head20_compactresp_v2/blindset_A/submission.zip
```

Why this one:

- It keeps the same ranking strategy that already scored public nDCG@20 `0.1935`.
- It changes only the response text, which was the clear weak point: public lexical diversity `0.0125`, LLM judge `1.0`.
- Local Blind A Distinct-2 rises to `0.6654`.
- It avoids spending submission risk on tail ranking changes whose catalog-diversity gain is small on an 80-row blind split.

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

Backup package:

```text
experiments/goalflow_taildiv_head18_compactresp_v2/blindset_A/submission.zip
```

Use it only after the response-focused package is tested. It changes positions 19-20 only, improves Blind A unique tracks from `1216` to `1268`, and keeps dev nDCG@20 close to safe (`0.08527` vs `0.08587`).

Hold back:

```text
experiments/goalflow_taildiv_head15_compactresp_v2/blindset_A/submission.zip
```

It has better diversity, but dev nDCG@20 drops to `0.08180`; the catalog gain is unlikely to pay for that risk on the public composite.
