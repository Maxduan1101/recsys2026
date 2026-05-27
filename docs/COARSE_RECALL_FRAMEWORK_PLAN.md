# GoalFlow 粗排框架升级计划

这份文档只讨论 ranking 前面的粗排候选池。

它的目标不是解释过去每一行代码，而是回答一个更大的问题：

```text
如果我们不满足于保守打榜，想把系统从 2 推到 10、甚至 100，
粗排 candidate pool 应该怎么重新设计？
```

## 1. 项目背景

Music-CRS 是 Conversational Music Recommendation，也就是多轮对话音乐推荐。

系统输入：

```text
用户画像
conversation_goal
当前用户说的话
历史对话
历史推荐歌曲
历史反馈
```

系统输出：

```text
predicted_track_ids:
  20 个官方曲库里的 track_id，必须有序，不能重复。

predicted_response:
  一段英文回复，解释为什么推荐这些歌。
```

主推荐指标是 `nDCG@20`。

`nDCG@20` 不是只看有没有命中，而是非常看重正确歌曲排在多前面。

但是在系统设计上，我们要把推荐拆成两步：

```text
第一步：粗排 / 召回
  目标是把可能正确的歌放进候选池。

第二步：rerank / 精排
  目标是在候选池里把最可能正确的歌排到前面。
```

所以现在粗排阶段的核心问题不是：

```text
RRF top20 分数高不高？
```

而是：

```text
在给 rerank 之前，候选池到底有没有包含正确答案？
```

## 2. 当前粗排系统

当前粗排大致是：

```text
ConversationState
  -> 多个 BM25 index
  -> 多种 query variant
  -> 多个 source
  -> candidate union
  -> RRF 临时排序
  -> LTR / LightGBM rerank
```

这里的 `source` 是一个独立候选生成通道。

例如：

```text
metadata_all:current
```

意思是：

```text
用 current 这个 query，
去 metadata_all 这个索引里搜，
得到一批候选歌。
```

当前已有 index：

```text
legacy_metadata
metadata_all
title_artist
album_artist
tags
enriched train-context docs
```

当前已有 query variant：

```text
legacy_history
current
goal
current_goal
seed_current
quoted_entities
```

## 3. 现在已经确认的问题

早期我们直接用 RRF 融合多个 source，结果不好：

```text
goalflow_bm25_aug_v1 nDCG@20 = 0.067874
goalflow_bm25_aug_v2 nDCG@20 = 0.074497
baseline              nDCG@20 = 0.085870
```

这说明：

```text
裸 RRF 的最终 top20 排序不可靠。
```

后来做 source diagnostics，看到：

```text
legacy source hit@20      = 0.2303
best single source hit@20 = 0.4715
RRF fused hit@20          = 0.2595
```

这说明：

```text
多个 source 里面确实有 source 能找到更多 gold。
但 RRF 不知道什么时候该信哪个 source。
```

所以结论是：

```text
不要把 RRF 当粗排真理。
粗排更应该输出 unordered candidate set。
source rank / RRF rank 只是 rerank 的特征。
```

## 4. Rerank 候选容量

已有 LTR 历史实验：

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

当前先定：

```text
稳定候选池：300-500
研究候选池：1000 以内
```

注意：

```text
1000 不是最终提交承诺。
1000 是研究上限，用来判断粗排还有多少召回空间。
```

## 5. 粗排评测新结果

新脚本：

```text
scripts/evaluate_recall_pool.py
```

它不看 response，不训练 LTR，只看粗排候选池。

目前已跑：

```text
retrieval_top_k = 100
retrieval_top_k = 260
retrieval_top_k = 500
retrieval_top_k = 1000
```

结果：

```text
top100:
  candidate_union coverage = 0.6324
  mean_pool_size           = 1344.1
  best_single_source hit@20 = 0.4718
  best RRF hit@20           ≈ 0.2631

top260:
  candidate_union coverage = 0.7060
  mean_pool_size           = 2698.9
  best_single_source hit@20 = 0.4715
  best RRF hit@20           ≈ 0.2645

top500:
  candidate_union coverage = 0.7661
  mean_pool_size           = 4830.6
  best_single_source hit@20 = 0.4715
  best RRF hit@20           ≈ 0.2651

top1000:
  candidate_union coverage = 0.8261
  mean_pool_size           = 8571.4
  best_single_source hit@20 = 0.4715
  best RRF hit@20           ≈ 0.2644
```

解释：

```text
把每个 source 拿更多候选，确实能让更多 gold 进入 union。
top100 -> top260 -> top500 的 coverage 明显增加。
top1000 继续增加，但成本非常高。

但是 RRF hit@20 基本不涨。
这说明新增候选有用，但旧排序方法不会用。
```

这正好支持下一步：

```text
粗排阶段先追求 candidate coverage。
然后把 source 信息作为 feature 交给 rerank 学。
```

同时也暴露了新问题：

```text
就算 top100，平均 union 也已经 1344。
top500 更是 4831。
top1000 达到 8571。
这超过了当前 rerank 300/500/1000 的合理预算。
```

所以不能把所有 source 的 union 直接塞给 rerank。

必须做：

```text
source budget allocation
candidate tiering
intent-aware source selection
```

## 6. Pro 研究问题已经重新问过

这次不是问英文小补丁，而是问了两个中文框架级问题。

问题 1：

```text
Embedding 如何系统性用于粗排 candidate generation？
```

保存位置：

```text
research/pro_answers/round_framework_embedding_multisource.txt
```

核心建议：

```text
不要把所有 embedding 都当 query-to-track dense retrieval。
应该拆成几类 source：

1. query-to-track retrieval
   当前文本 / goal 直接搜 track。

2. seed-to-track retrieval
   用历史推荐过、用户正反馈过的歌曲找相似歌。

3. user-to-track retrieval
   用 user cf-bpr 和 track cf-bpr 做个性化召回。

4. multimodal expansion
   audio / image 不一定能直接吃文本 query，
   更适合从可信 seed 扩展。

5. candidate feature source
   不强行召回，只给 rerank 提供相似度和 source agreement 特征。
```

问题 2：

```text
不押宝 embedding，整体粗排候选池如何框架级升级？
```

保存位置：

```text
research/pro_answers/round_framework_coarse_source_gating.txt
```

核心建议：

```text
粗排应该输出 candidate_set(turn)。
不要让 RRF 直接决定最终 top20。

每个 candidate 应该带：
  track_id
  source_name
  source_rank
  source_score
  matched_query_variant
  matched_field
  candidate_tier
```

并且要按 intent 动态分配 source：

```text
specific_track:
  title / artist / quoted entity / exact match source 更重要。

artist_exploration:
  artist discography / same artist / artist context 更重要。

mood_playlist:
  tags / attributes / train-context / CF 更重要。

similar_to_seed:
  positive seed similarity / same artist / same album / embedding seed 更重要。

lyrics_theme:
  lyrics source / theme tags 更重要。

cover_art:
  image / cover-caption / visual tags 更重要。
```

## 7. 下一步不应该继续钻 RRF

现在已经能回答之前的问题：

```text
不是 RRF 参数 60 还是 26 的问题。
```

因为：

```text
top100 / top260 / top500 的 RRF hit@20 几乎不动。
但 candidate_union coverage 一直涨。
```

这说明主要矛盾不是：

```text
RRF_k 调多少？
```

而是：

```text
如何把大 union 压成 rerank 能吃的 300/500/1000 候选，
同时尽量不丢 gold？
```

## 8. 下一轮实施路线

下一轮应该做四件事。

### 8.1 先做 source-specific topK 调参

每个 source 的 topK 不应该一样。

原因很简单：

```text
有的 source 前 50 个就很准，后面基本是噪声。
有的 source 前 100 个一般，但 100-500 还能继续找到 gold。
有的 source 最大覆盖率高，但和其他 source 重叠很多。
```

所以第一步不是先把大 union 压到 300/500/1000，而是先画每个 source 自己的曲线：

```text
source hit@20
source hit@50
source hit@100
source hit@300
source hit@500
source hit@800
source hit@1200
```

新增脚本：

```text
scripts/analyze_source_topk_curves.py
```

它读取：

```text
experiments/recall_pool_top1000/recall_pool/recall_pool_summary.csv
```

输出：

```text
experiments/recall_pool_top1000/recall_pool/source_topk_curves/source_topk_curves.csv
experiments/recall_pool_top1000/recall_pool/source_topk_curves/source_topk_gain_report.md
```

当前初步结果：

```text
enriched:seed_current
  coverage = 0.6681
  hit@100 = 0.4555
  hit@500 = 0.6055
  recommended_k ≈ 1200

metadata_all:seed_current
  coverage = 0.6627
  hit@100 = 0.4614
  hit@500 = 0.5984
  recommended_k ≈ 1200

enriched:legacy_history
  coverage = 0.5801
  hit@100 = 0.4101
  hit@500 = 0.5301
  recommended_k ≈ 800

metadata_all:legacy_history
  coverage = 0.5560
  hit@100 = 0.4049
  hit@500 = 0.5044
  recommended_k ≈ 800

tags:seed_current
  coverage = 0.4721
  hit@100 = 0.3246
  hit@500 = 0.4721
  recommended_k ≈ 500
```

这说明：

```text
seed_current 相关 source 的 K 放大很有价值。
enriched / metadata_all 的 K 也值得放大。
tags 不是完全没用，但更像中大 K 的补充 source。
所有 source 用同一个 K 是不合理的。
```

注意：

```text
这只是 source 自身曲线。
下一步还要做 incremental recall：
某个 source 新找到的 gold，是不是其他 source 已经找到过？
```

### 8.2 再做 capped candidate pool 评测

source topK 调清楚以后，才做 capped pool。

也就是：

```text
给每个 source 不同 topK，
再构造最多 300 / 500 / 1000 个候选。
```

然后看：

```text
capped_pool_300 coverage
capped_pool_500 coverage
capped_pool_1000 coverage
```

目标不是先优化 nDCG，而是回答：

```text
在预算限制下，最多能保住多少 gold？
```

### 8.3 做 intent-aware source router

现在所有 turn 基本吃同一套 source。

下一步应该按 intent 分配预算：

```text
specific_track:
  title_artist / quoted_entities / legacy_history 优先。

mood_playlist:
  tags / enriched / goal / current_goal 优先。

similar_to_seed:
  seed_current / seed metadata / seed embedding 优先。

artist_exploration:
  artist exact / same artist / train artist context 优先。
```

### 8.4 给 rerank 加 source/tier 特征

候选进入 rerank 时要带：

```text
candidate_pool_tier:
  core300
  extra500
  extra1000

came_from_source_x
best_rank_in_source_x
source_count
best_source_rank
best_index_rank
best_query_rank
```

这样 rerank 才知道：

```text
这首歌为什么进候选池？
它是高精度 source 找到的，还是低置信 source 找到的？
```

### 8.5 再接 embedding

embedding 不是不用。

但下一步不能再做“尾部替换”这种弱接入。

应该按 Pro 建议做成独立 source：

```text
metadata_dense_query
lyrics_dense_query
user_cf_recall
positive_seed_cf_recall
positive_seed_audio_recall
positive_seed_attribute_recall
image_seed_or_cover_recall
```

然后评估：

```text
incremental recall:
  这个 source 是否找到了 BM25 union 找不到的 gold？

overlap:
  它是不是只重复找到了已有候选？

noise:
  它每新增 100 个候选，能多带来多少 gold？
```

## 9. 当前结论

现在最重要的结论是：

```text
粗排确实还有空间。
```

证据是：

```text
candidate_union coverage:
  top100 = 0.6324
  top260 = 0.7060
  top500 = 0.7661
  top1000 = 0.8261
```

但是：

```text
当前融合和 rerank 没有有效吃掉这个空间。
```

证据是：

```text
RRF hit@20 基本卡在 0.26 左右。
```

所以下一阶段的主线应该是：

```text
从“RRF 排序”切换到“预算化候选池 + source feature rerank”。
```
