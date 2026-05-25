# Blind A Submission Decision

Recommended first retry:

```text
experiments/goalflow_head20_compact_broad/blindset_A/submission.zip
```

Why this one:

- It keeps the same ranking strategy that already scored public nDCG@20 `0.1935`.
- It changes only the response text, which was the clear weak point: public lexical diversity `0.0125`, LLM judge `1.0`.
- Local Blind A Distinct-2 rises to `0.6642`.
- It keeps compact v2's lexical diversity while filtering bad/private tag artifacts such as `albums i own`, `seen live`, `lastfm`, and profanity-like tag variants.
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
