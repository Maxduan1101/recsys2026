# GoalFlow-MusicCRS 框架级研究 Brief

这份 brief 是给后续深度研究模型和我们自己看的。

目标不是继续做“稳定打榜小修小补”，而是重新站在系统设计角度，研究如何把 Music-CRS 推荐框架从当前水平往上扩展。

换句话说：

```text
不是从 2 到 3。
而是考虑从 2 到 10，从 2 到 100。
```

## 1. 项目背景

比赛是 RecSys Challenge 2026 的 Music-CRS，也就是 Conversational Music Recommendation。

系统输入：

```text
用户画像
conversation_goal
当前用户请求
历史对话
历史推荐歌曲
历史反馈 / goal_progress_assessments
```

系统输出：

```text
predicted_track_ids:
  官方曲库里的 top-20 track_id，有序、不能重复。

predicted_response:
  一段英文自然语言回复，解释为什么推荐。
```

平台打分：

```text
nDCG@20:
  推荐排序主指标。

catalog_diversity:
  推荐歌曲覆盖多少不同曲目。

lexical_diversity:
  回复文本二元词组多样性。

LLM-as-a-Judge:
  Gemini 等模型评估回复的个性化和解释质量。
```

Blind A public 结果：

```text
上一版:
  nDCG@20            0.1935
  catalog_diversity  0.0257
  lexical_diversity  0.0125
  llm_judge_score    1.0000
  composite_score    0.1006

当前主包:
  nDCG@20            0.1898
  catalog_diversity  0.0317
  lexical_diversity  0.6060
  llm_judge_score    1.5000
  composite_score    0.1962
```

解释：

```text
response 修复非常有效。
ranking 没有提升，甚至略降。
下一阶段真正要解决 ranking / recall。
```

## 2. 当前系统简述

当前系统大致是：

```text
ConversationState
  -> 多 BM25 index + 多 query variant
  -> candidate sources
  -> RRF / candidate pool
  -> LightGBM LambdaRank rerank
  -> 多模型 RRF ensemble
  -> top-20
  -> metadata-grounded response
```

已经做过的粗排 source：

```text
legacy_metadata
metadata_all
title_artist
album_artist
tags
enriched train-context docs
```

query variant：

```text
legacy_history
current
goal
current_goal
seed_current
quoted_entities
```

当前问题：

```text
这些 source 是有信号的。
但之前过度依赖 RRF 作为粗排融合和候选截断方式。
RRF 不应该被当作“正确粗排真理”。
```

## 3. 当前粗排现状

早期裸 RRF 结果：

```text
goalflow_bm25_aug_v1 nDCG@20 = 0.067874
goalflow_bm25_aug_v2 nDCG@20 = 0.074497
baseline              nDCG@20 = 0.085870
```

这只能说明：

```text
裸 RRF 的最终 top-20 排序变差。
```

后续 source diagnostics 才说明：

```text
legacy source hit@20      = 0.2303
best single source hit@20 = 0.4715
RRF fused hit@20          = 0.2595
```

解释：

```text
多 source 里确实有人能找到更多 gold。
但 RRF 没有学会什么时候该信哪个 source。
```

现在新增了专门的粗排评测脚本：

```text
scripts/evaluate_recall_pool.py
```

这个脚本不看 response，不训练 LTR，只评估：

```text
candidate_union coverage
per-source hit@K
RRF hit@K 仅作参考
平均 candidate pool size
gold 完全没进候选池的 miss list
```

下一步应当正式跑：

```text
retrieval_top_k = 100
retrieval_top_k = 260
retrieval_top_k = 500
retrieval_top_k = 1000
```

关键不是 top20，而是：

```text
candidate_union recall@100 / @300 / @500 / @1000 / @2000
```

## 4. Rerank 可接受候选范围

目前已有历史实验：

```text
max_candidates_per_group = 200
  train_rows ≈ 591400
  valid_groups_with_positive_candidates = 737
  nDCG@20 = 0.18156

max_candidates_per_group = 300
  train_rows ≈ 947400
  valid_groups_with_positive_candidates = 787
  nDCG@20 ≈ 0.18302 - 0.18432

max_candidates_per_group = 500
  train_rows ≈ 1709000
  valid_groups_with_positive_candidates = 859
  nDCG@20 = 0.18257
```

当前建议先把 rerank 容量预期定为：

```text
生产/稳态候选上限：300-500
研究/扩展候选上限：1000
```

也就是：

```text
先默认“1000 以内”是 rerank 可探索区。
但真正稳定提交未必直接用 1000。
```

原因：

```text
300 已经验证过稳定。
500 让更多 gold 进入候选，但排序噪声更大。
1000 可能能吃，但需要重新评估训练时间、内存、特征质量和 LTR 对噪声的处理能力。
```

重要原则：

```text
候选池大小不是死的。
粗排应该先尽量提高 candidate_union coverage。
如果 1000 候选明显提高 recall，后续应改进 rerank，而不是因为旧 LTR 处理不好就放弃召回。
```

## 5. 对之前 Pro 研究问题的复盘

之前问过 Pro 的相关问题主要有：

```text
round2/tab2_embedding-based_extension_design
round5/tab4_embedding_cf_integration
round7/tab4_embedding_query_features
round8/tab4_embedding_ltr_features
round10/tab2_direct_query_embedding_path
round10/tab3_direct_embedding_guardrails
```

这些问题多数偏向：

```text
如何低风险把 embedding 接进当前系统
如何不破坏 BM25/LTR 头部
如何做 tail rescue
如何把 embedding 做成 LTR feature
```

它们的价值：

```text
给了很多 guardrail 和低风险接入建议。
```

它们的不足：

```text
问题设定偏“保守打榜”。
没有从框架层面充分研究 recall-first candidate pool。
也没有把目标设定成“如何系统性用多模态 embedding 扩大召回上限”。
```

所以现在需要重新问两个框架级问题。

## 6. Pro 问题 1：Embedding 如何用于粗排

### 中文目的

官方给了很多 embedding，所有 AI 都会注意到这个信息。

我们不能只说“embedding 特征试了没用”，因为那只是很浅的接入方式。

现在要研究：

```text
如何把 metadata-qwen3、attributes-qwen3、lyrics-qwen3、audio CLAP、image SigLIP2、cf-bpr
系统性用于粗排 candidate generation。
```

重点是：

```text
召回，不是最终打榜保守后处理。
```

### 中文问题

```text
你在帮助设计 RecSys Challenge 2026 Music-CRS 的框架级推荐系统。

项目背景：
- 官方曲库约 47071 首歌。
- 输入是用户画像、conversation_goal、当前对话、历史 turn 和历史推荐反馈。
- 输出是 top-20 track_id 和自然语言回复。
- 推荐主指标是 nDCG@20，但我们现在专注粗排候选池，不是最终 top20。
- 当前系统已有多 BM25 index、多 query variant、train-context augmentation、LightGBM LambdaRank rerank。
- 目前 rerank 已验证能稳定处理每 turn 约 300 candidates，500 candidates 会增加 gold 覆盖但排序噪声变大；我们希望把研究上限先放到 1000 candidates 以内。

官方 embedding：
- metadata-qwen3
- attributes-qwen3
- lyrics-qwen3
- audio-laion_clap
- image-siglip2
- track cf-bpr
- user cf-bpr

已有浅层实验：
- seed-CF tail rescue 很保守，只替换尾部，收益极小或不稳定。
- embedding LTR features：track_cf+user_cf 有一点信号但没超过主线；metadata seed cosine 接近但没赢；attributes seed cosine 明显伤分。

现在不要给保守打榜建议。
请从框架设计角度回答：

1. 如何把这些 embedding 系统性用于粗排 candidate generation？
2. 哪些 embedding 应该做 query-to-track dense retrieval？
3. 哪些只能做 seed similarity？
4. 哪些适合做 user-personalized recall？
5. audio/image embedding 在没有明确兼容 query encoder 的情况下应该如何利用？
6. 每个 source 建议 topK 多少，如何组成 candidate union，使总候选量控制在 300/500/1000 三档？
7. 如何用 dev gold 评估 embedding source 的 incremental recall，而不是只看最终 nDCG？
8. 如何设计 ablation 表，判断 embedding 是没用、接入方式错了，还是 rerank 没吃好？
9. 如何避免 dense source 带来大量噪声，影响后续 rerank？
10. 给出一个从最小可行到完整多模态候选池的实施路线。

请用中文回答，面向一个第一次参加推荐系统比赛但能读技术解释的人。
```

## 7. Pro 问题 2：整体粗排系统如何框架级升级

### 中文目的

embedding 是一个方向，但不是全部。

还要问：

```text
如果 embedding 另有一条线在研究，整个粗排系统还应该怎么系统升级？
```

目标不是：

```text
保守打榜 +0.001
```

而是：

```text
设计一个优秀的粗排框架。
```

### 中文问题

```text
你在帮助设计 RecSys Challenge 2026 Music-CRS 的粗排候选池框架。

项目背景：
- 官方曲库约 47071 首歌。
- 每个样本是多轮对话音乐推荐。
- 系统最后只提交 top-20 track_id，但我们现在只研究粗排 candidate pool。
- 当前 rerank 是 LightGBM LambdaRank，大概已验证每 turn 300 candidates 稳定，500 candidates 有更多 gold 覆盖但排序噪声变大；我们愿意研究 1000 candidates 以内的候选池。
- 当前粗排已有 BM25 多 index、多 query variant、train-context augmentation、history seed query、quoted entity query。
- 早期 RRF top20 变差，但 source diagnostics 显示 best single source hit@20 明显高于 legacy，说明有召回潜力。
- 我们现在不想假定 RRF 正确。粗排输出应被看作 unordered candidate set，最终排序交给 rerank。
- embedding 如何用于粗排会作为单独问题研究，所以这里可以提到 embedding 但不要把答案全部押在 embedding 上。

请从框架级角度回答：

1. 一个优秀的 Music-CRS 粗排 candidate pool 应该包含哪些 source？
2. 如何按 intent 类型设计 source：specific_track、artist_exploration、album、mood_playlist、lyrics_theme、cover_art、similar_to_seed、era_genre？
3. 如何把 conversation_goal、当前 user query、历史对话、positive/negative seeds 分别转成查询？
4. BM25 多 index 应该如何设计？哪些字段组合值得保留，哪些会引入噪声？
5. train-context augmentation 应该如何安全使用，如何避免 dev/Blind 泄漏和过拟合？
6. 是否应该做 entity parser？如果做，具体抽取什么字段，如何用于候选召回？
7. 如何做 source budget allocation？例如每个 source topK 多少，总候选池控制在 300/500/1000 三档。
8. 粗排评测应该看哪些指标？请重点设计 candidate_union recall@K、incremental recall、miss taxonomy、source overlap、oracle upper bound。
9. 如何判断某个 source 是真的有用，还是只是在增加噪声？
10. 如何把粗排评测和 rerank 训练解耦？
11. 如果候选池 recall 很高但 rerank 不好，下一步该怎么改 rerank；如果候选池 recall 很低，下一步该怎么加 source？
12. 给出一个从当前系统演进到强 candidate pool 的路线图。

请用中文回答。目标是优秀框架设计，不是保守打榜微调。
```

## 8. 当前本地行动建议

在 Pro 回答之前，本地可以先做：

```text
1. 跑 scripts/evaluate_recall_pool.py
   retrieval_top_k = 100, 260, 500, 1000

2. 产出 candidate_union coverage 表
   按 overall / turn / category / specificity / intent 分组

3. 统计 pool size
   看 300 / 500 / 1000 三档候选预算是否合理

4. 列出 miss taxonomy
   gold 完全没进入候选池的样本到底是什么类型

5. 再决定：
   是补 embedding source？
   补 entity parser？
   补 train-context retrieval？
   还是改 rerank 去吃更大候选池？
```

