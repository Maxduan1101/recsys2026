# GoalFlow-MusicCRS 从 Baseline 到 0.1962 的完整阶段日志（原时间线版）

这份文档面向第一次参加推荐系统比赛的人。它不讲代码细节，而是讲清楚：我们为什么做某件事、做了什么实验、结果是好还是坏、因此改变了什么策略。

当前项目目录：

```text
/Users/bytedance/generated_problems/recsys2026_music_crs/goalflow_musiccrs
```

最新 Blindset A 主提交包：

```text
blindset_A_PRIMARY_submission.zip
```

最新 Blindset A public 结果：

```text
nDCG@20            0.1898
catalog_diversity  0.0317
lexical_diversity  0.6060
llm_judge_score    1.5000
composite_score    0.1962
```

上一版 public 结果：

```text
nDCG@20            0.1935
catalog_diversity  0.0257
lexical_diversity  0.0125
llm_judge_score    1.0000
composite_score    0.1006
```

一句话总结：这轮没有提升推荐排序，`nDCG@20` 从 `0.1935` 小降到 `0.1898`；但回复文本和多样性大幅提升，把总分从 `0.1006` 拉到 `0.1962`，接近翻倍。

## 1. 先解释比赛到底在干什么

RecSys Challenge 2026 Music-CRS 是一个对话式音乐推荐比赛。

`RecSys` 是 recommender systems 的缩写，意思是推荐系统。比如音乐软件给你推荐歌、购物网站给你推荐商品，都是推荐系统。

`CRS` 是 conversational recommender system 的缩写，意思是对话式推荐系统。它不是只看一个搜索词，而是要看多轮聊天。

每条样本大概长这样：

```text
用户画像：这个用户来自哪里、喜欢哪类音乐文化
对话目标：用户想找一首具体歌，或者想找某种风格的歌
历史对话：前面用户怎么说，系统推荐过什么
当前请求：用户这一轮说的话
```

系统要输出两样东西：

```text
1. predicted_track_ids
   从官方曲库里选 20 首歌，按最可能正确到最不可能正确排序。

2. predicted_response
   给用户的一段英文回复，解释为什么推荐这些歌。
```

注意：20 首歌必须来自官方曲库，不能编歌名，不能重复 track_id。

## 2. 评测指标完整解释

### nDCG@20

`nDCG@20` 是最重要的推荐排序指标。

可以把它理解成：

```text
正确答案排第 1 名：分数最高
正确答案排第 2 名：分数低一点
正确答案排第 20 名：还有一点分
正确答案没进前 20：这一轮得 0
```

所以 top-1、top-5 非常重要。不能为了花哨的多样性，把最可能正确的歌挤到后面。

### Catalog Diversity

`Catalog Diversity` 可以翻译成曲库覆盖多样性。

如果系统总是推荐同一批热门歌，分数低；如果推荐覆盖更多不同歌曲，分数高。

Blindset A 只有 80 条，每条 20 首歌，所以最多只有：

```text
80 * 20 = 1600 个推荐位置
```

官方曲库大小是：

```text
47071 首歌
```

所以 Blindset A 的 catalog diversity 理论上限大约是：

```text
1600 / 47071 = 0.0340
```

这解释了为什么 `0.0317` 已经接近上限。

### Lexical Diversity / Distinct-2

`Lexical Diversity` 是文本多样性。这里基本等价于 Distinct-2。

`Distinct-2` 的意思是：把回复切成连续两个词的组合，看看有多少不同组合。

如果所有回复都写：

```text
Here are some songs you might enjoy.
```

那二元词组重复很多，分数会很低。

如果每条回复都具体提到不同歌名、艺人、年代、专辑、风格，分数会高很多。

这次我们的 lexical diversity 从 `0.0125` 涨到 `0.6060`，这是总分翻倍的主要原因。

### LLM-as-a-Judge

`LLM-as-a-Judge` 是让大语言模型当评委。这里 public 显示的 judge 平均分是 `1.5000`。

它主要看回复是不是：

```text
1. 个性化：有没有结合用户画像和历史反馈
2. 解释具体：有没有说清楚为什么推荐
3. 不胡编：有没有只引用 metadata 里能证明的信息
4. 自然：是不是像一个音乐推荐助手，而不是死板模板
```

这个分数现在还是主要瓶颈。它从 `1.0` 到了 `1.5`，但还不高。

### Composite Score

`Composite Score` 是最终总分。我们根据 public 结果推断它大致是：

```text
0.5 * nDCG@20
+ 0.1 * catalog_diversity
+ 0.1 * lexical_diversity
+ 0.3 * ((llm_judge_score - 1) / 4)
```

解释：

```text
nDCG 权重最大，占一半
catalog 和 lexical 各占 0.1
LLM judge 折算后占 0.3
```

所以后面想继续涨分，最重要是：

```text
1. 让 nDCG 回到 0.20+
2. 让 judge 从 1.5 提到 2.0 或更高
```

## 3. 我们一开始的 Baseline

`Baseline` 是官方或最初版本的基础方案。它的意义是：先跑通比赛流程，知道提交格式、数据读取、评测方式都没问题。

最早强 baseline 是 BM25-history。

`BM25` 是一种经典文本搜索算法。你可以把它理解成“关键词搜索增强版”。

例如用户说：

```text
I want a dreamy indie folk song.
```

BM25 会更偏向文档里包含 `dreamy`、`indie`、`folk` 的歌曲。

Baseline 的大概流程：

```text
对话历史
  -> 把历史推荐过的 track_id 转成歌曲信息
  -> 拼成搜索文本
  -> 用 BM25 搜曲库
  -> 输出 top-20
  -> 用模板或小模型生成回复
```

最早 dev 结果大概是：

```text
nDCG@20            0.085870
catalog_diversity  0.388966
lexical_diversity  0.000125
```

这说明：

```text
推荐排序还算有用
回复文本几乎完全不行
```

## 4. 项目独立化：先复制一份，不污染官方 baseline

用户要求“如果需要 baseline 要复制一份，不要直接覆盖”。

所以我们创建了独立项目：

```text
goalflow_musiccrs/
```

这样做的意义：

```text
1. 官方 baseline 保持干净
2. 我们可以大胆实验
3. 出问题时能回退
4. 后面文档、实验、提交包都在一个独立项目里
```

## 5. 总体系统设计

我们给系统起名：

```text
GoalFlow-MusicCRS
```

核心想法：

```text
先理解用户当前到底想找什么
再用多种方式从曲库召回候选歌曲
然后用学习型模型重新排序
最后生成具体、有解释的回复
```

总流程图：

```text
用户画像 + 当前请求 + 历史对话
        |
        v
ConversationState 当前会话状态
        |
        v
多路召回：BM25 / metadata / tags / train-context / feedback seeds
        |
        v
候选融合：RRF + 规则分数
        |
        v
LTR 学习排序模型
        |
        v
RRF 模型集成
        |
        v
top-20 track_id
        |
        v
metadata-grounded response
        |
        v
prediction.json / submission.zip
```

下面解释几个词。

`ConversationState` 是当前会话状态。它把用户画像、当前请求、历史推荐、历史反馈整理成一个结构化对象。

`Recall` 是召回。意思是从 47071 首歌里先粗略捞出一批可能相关的候选，不追求排序完美，只追求别漏掉正确答案。

`Candidate` 是候选歌曲。先召回几百或几千首，再从里面排 top-20。

`Rerank` 是重排。召回之后再用更强的模型重新排序。

`Metadata` 是歌曲元数据，比如歌名、艺人、专辑、标签、年份、流行度。

## 6. 第一阶段：多路 BM25 和 RRF 融合

我们先做了多种 BM25 索引。

`Index` 是索引，可以理解成一种搜索目录。不同索引搜不同字段：

```text
metadata_all: 歌名 + 艺人 + 专辑 + 标签 + 年份
title_artist: 歌名 + 艺人
album_artist: 专辑 + 艺人
tags: 风格标签
enriched: metadata + 训练对话增强文本
legacy_metadata: 接近官方 baseline 的历史搜索方式
```

又做了多种 query。

`Query` 是搜索输入。我们不只搜当前用户一句话，还搜：

```text
当前用户请求
conversation_goal
当前请求 + 目标
历史正反馈歌曲 + 当前请求
legacy history query
```

然后用 `RRF` 融合。

`RRF` 是 Reciprocal Rank Fusion，意思是倒数排名融合。简单说：

```text
多个搜索器都觉得某首歌靠前，那它整体分就高
某个搜索器把它排第 1，会加很多分
排第 100，也加一点分
```

实验结果：

```text
goalflow_bm25_aug_v1 nDCG@20 = 0.067874
goalflow_bm25_aug_v2 nDCG@20 = 0.074497
baseline              nDCG@20 = 0.085870
```

只看这三行，严格来说只能得出一个结论：

```text
多路 BM25 + RRF 的最终 top-20 排序，比 baseline 差。
```

不能只凭这三行就说“找到了更多候选”，也不能只凭这三行就说“一定是 RRF 冲散了 baseline”。这两个判断来自后面专门做的 `Source Diagnostics`，也就是把每一个召回源单独拆开看。

后续诊断发现：

```text
legacy source hit@20      = 0.2303
best single source hit@20 = 0.4715
RRF fused hit@20          = 0.2595
```

这里的 `hit@20` 意思是：正确歌曲有没有出现在某个 source 的前 20 名里。

`hit@20` 和 `nDCG@20` 不一样：

```text
hit@20 只问：正确歌有没有进前 20。
nDCG@20 还问：正确歌排第几。
```

如果正确歌排第 20，`hit@20` 算命中，但 `nDCG@20` 分数很低。这个区别很重要，因为我们的很多新 source 是“能找到”，但“不一定排得足够靠前”。

`legacy source` 是接近 baseline 的旧搜索源。

`best single source` 是诊断用的“事后最佳 source”。它不是一个可提交模型，而是问：如果我们事后知道每一轮哪个 source 最会找正确歌，那么所有 source 里最好的那个能不能把正确歌放进前 20。

这组数说明：

```text
旧 source 自己只能在 23.03% 的轮次把正确歌放进前 20。
但多路 source 里，确实有一些 source 能覆盖更多正确歌，事后最佳能到 47.15%。
实际 RRF 融合后只有 25.95%，远低于事后最佳。
```

所以更准确的结论是：

```text
新增 source 让候选池里出现了更多正确答案。
但简单 RRF 没有学会什么时候该相信哪个 source。
它一边带进了一些新命中，一边也把 baseline 原本靠前的正确答案挤下去了。
```

这才叫 head-rank dilution。意思是：头部排名被稀释了。

白话说，baseline 像一个保守但靠谱的老搜索器；多路 RRF 像让很多搜索器一起投票。新搜索器确实知道一些老搜索器不知道的歌，但投票规则太粗糙，于是有些老搜索器本来排第 1、第 2 的正确答案，被一堆“也有点相关”的候选挤到了后面。

策略改变：

```text
不能直接相信所有新召回源
要保护 baseline 的头部排序
```

## 7. 第二阶段：Legacy Head Protection

`Legacy` 是旧系统，也就是官方 BM25-history baseline。

`Head Protection` 是保护前几名。

我们做了：

```text
head10: 前 10 名沿用 legacy，后面用 GoalFlow
head20: 前 20 名都沿用 legacy，只改回复
```

结果：

```text
head10 nDCG@20 = 0.083776
head20 nDCG@20 = 0.085870
```

解释：

```text
head20 完全保住 baseline 排名
同时让回复变好
```

策略改变：

```text
早期提交应优先安全
先不冒险改 ranking，先修 response
```

## 8. 第三阶段：Tail Diversity

`Tail` 是排序后半段，比如第 16 到第 20 名。

`Tail Diversity` 是只改后几名，让推荐更丰富，尽量不伤前几名。

我们试了：

```text
taildiv_head10: 保护前 10，后 10 做多样性
taildiv_head15: 保护前 15，后 5 做多样性
taildiv_head18: 保护前 18，后 2 做多样性
taildiv_head19: 只改第 20 名
```

结果：

```text
head10 diversity 很高，但 nDCG 掉太多
head15 diversity 高，但 nDCG 仍损失明显
head18 / head19 更安全
```

典型结果：

```text
taildiv_head10 nDCG@20 = 0.072093
taildiv_head15 nDCG@20 = 0.081796
taildiv_head18 nDCG@20 = 0.085271
taildiv_head19 nDCG@20 = 0.085758
```

结论：

```text
catalog diversity 有用
但 Blindset A 上限只有 0.034
为了 catalog 多样性牺牲 nDCG 不划算
```

策略改变：

```text
多样性只能很保守地做
后面不再把 catalog diversity 当主攻方向
```

## 9. 第四阶段：回复生成大改

第一次 public 结果暴露了最大问题：

```text
nDCG@20            0.1935
catalog_diversity  0.0257
lexical_diversity  0.0125
llm_judge_score    1.0000
composite_score    0.1006
```

这说明：

```text
推荐排序其实还不错
回复文本非常弱
LLM judge 也只给最低档
```

这里也要注意证据边界：`nDCG@20=0.1935` 只能说明这个旧包在 public Blind A 的推荐排序不差，不能说明它在所有 split 都强；`lexical_diversity=0.0125` 和 `llm_judge_score=1.0` 才直接说明回复文本是当时最大的短板。

于是重点转到 response。

我们做了多种回复风格。

### compact response

`compact` 是短而信息密集的回复。

它会写：

```text
推荐第一首是什么
为什么匹配当前请求
使用了哪些标签、年代、专辑信息
历史正反馈或负反馈如何影响排序
后面两首备选是什么
```

效果：

```text
lexical diversity 从约 0.083 提到约 0.175+
```

### compact_broad

`broad` 是允许更宽的标签来源，但后来发现风险：某些 last.fm 风格标签很脏或很私人。

比如：

```text
albums i own
seen live
lastfm
songsof2011
lobpreis
```

这些写进回复会显得不专业，甚至像泄漏用户私有标签。

策略改变：

```text
要过滤 noisy tags
```

`Noisy tag` 是噪声标签，意思是曲库里有但不适合写给用户看的标签。

### judge_v2

`judge_v2` 是面向 LLM judge 的回复。

它更强调：

```text
1. 我为什么选第一首
2. 我引用了哪些 metadata
3. 我如何使用历史反馈
4. 我如何使用用户画像作为 tie-breaker
```

`Tie-breaker` 是打破平局的辅助依据。比如两首歌都像，用户画像可以帮助决定谁排前。

### judge_clean_mix

后来我们发现：

```text
太自然的回复 lexical diversity 低
太模板的回复 LLM judge 可能不喜欢
```

于是做了混合风格 `judge_clean_mix`。

它混合：

```text
judge_v2
judge_brief
compact
```

并且清理：

```text
重复标题
list-valued artist name
专辑 remaster 噪声
坏标签
过长回复
```

最终主包使用这个风格。

效果：

```text
Blind A local Distinct-2 约 0.606
Public LexDiv = 0.6060
Judge 从 1.0 提到 1.5
```

结论：

```text
这一阶段是总分提升的核心
```

这个判断来自 public 指标对比，而不是主观感觉：

```text
nDCG@20            0.1935 -> 0.1898   小降
catalog_diversity  0.0257 -> 0.0317   小幅上升
lexical_diversity  0.0125 -> 0.6060   巨幅上升
llm_judge_score    1.0000 -> 1.5000   上升
composite_score    0.1006 -> 0.1962   接近翻倍
```

所以这轮总分上涨不是因为 ranking 变强，而是因为 response 和 diversity 修复成功。

## 10. 第五阶段：Source Diagnostics

`Source Diagnostics` 是分析每个召回源到底有没有用。

我们问的问题是：

```text
哪个搜索源能找到 gold track？
哪个源能把 gold track 排得靠前？
为什么融合后反而变差？
```

`Gold track` 是官方正确答案歌曲。

诊断发现：

```text
best single source hit@20 = 0.4715
legacy source hit@20      = 0.2303
RRF fused hit@20          = 0.2595
```

解释：

```text
某些新 source 单独看能找到很多正确歌
但 RRF 融合没有把它们正确排到前面
```

策略改变：

```text
问题不是召回完全没用
问题是融合和排序不够聪明
下一步应该做学习排序 LTR
```

## 11. 第六阶段：Progress Label 语义审计

数据里有 `goal_progress_assessments`。

它表示前一轮推荐是否让用户更接近目标。

我们一开始不确定：

```text
label t 是评价第 t 轮推荐？
还是评价第 t-1 轮推荐？
```

通过审计样例发现：

```text
turn 1 没有 label
turn 2 的 label 是用户对 turn 1 推荐的反馈
```

也就是说 label 要后移使用。

策略改变：

```text
历史推荐 m 的反馈应该用 progress[m + 1]
```

这个修正让 positive seed / negative seed 更合理。

`Positive seed` 是历史上被用户认可的歌曲。

`Negative seed` 是历史上用户不满意的歌曲。

## 12. 第七阶段：Train-context Document Augmentation

`Document Augmentation` 是文档增强。

普通歌曲文档只有：

```text
歌名
艺人
专辑
标签
年份
```

我们增加了训练集中与这首歌相关的对话上下文：

```text
用户当时怎么描述这首歌
conversation_goal 怎么写
assistant 当时怎么解释
前面哪些歌曲是正反馈
```

目的：

```text
让系统学会“用户自然语言表达 -> 某首歌”
```

结果：

```text
enriched source 的召回有帮助
但直接 RRF 融合仍会稀释头部排名
```

这里的证据也来自 source diagnostics：

```text
legacy source hit@20       = 0.2303
index_any=enriched hit@20  = 0.3685
index_any=metadata hit@20  = 0.3635
RRF fused hit@20           = 0.2595
```

意思是：

```text
enriched 文档单独看，比 legacy source 更容易把正确歌放进前 20。
但混合到 RRF 后，最终 fused 排序没有接近 enriched 的单源潜力。
```

所以准确说法不是“augmentation 直接让最终系统变强”，而是“augmentation 提高了候选覆盖潜力，但需要更聪明的重排模型才能兑现”。

策略改变：

```text
文档增强保留为召回源和 LTR 特征来源
但不直接主导最终排名
```

## 13. 第八阶段：官方 Embeddings 探索

`Embedding` 是向量表示。可以理解成把文字、音频、图片、用户偏好变成一串数字，让相似的东西距离更近。

官方提供了多种 embedding：

```text
metadata-qwen3       歌曲元数据文本向量
lyrics-qwen3         歌词向量
attributes-qwen3     风格、情绪、属性向量
audio-laion_clap     音频向量
image-siglip2        封面图像向量
cf-bpr               协同过滤向量
```

`CF` 是 collaborative filtering，协同过滤。意思是根据“相似用户喜欢什么”来推荐。

`BPR` 是 Bayesian Personalized Ranking，一种常见协同过滤训练方式。

`CLAP` 是音频和文本对齐的模型，常用于音乐/声音检索。

`SigLIP` 是图像和文本对齐的模型，类似 CLIP。

我们先做了 embedding schema 检查，发现：

```text
必须用 Challenge 版本 embeddings
旧版 embedding 和当前 track UUID 不匹配
```

后续尝试：

```text
seed-CF tail rescue
embedding LTR features
metadata seed cosine
attributes seed cosine
user_cf / track_cf
```

`Cosine` 是余弦相似度，衡量两个向量方向是否相似。

结果：

```text
seed-CF tail rescue 只带来极小提升或不稳定
embedding LTR features 在 fold 0 反而更差
attributes seed cosine 明显伤分
```

结论：

```text
embedding 很有研究价值
但在最终 Blind A 阶段不是低风险提分点
```

策略改变：

```text
不把 embedding 直接用于最终 ranking
保留为未来 Blind B 或长期研究方向
```

## 14. 第九阶段：LTR 学习排序

这是项目最大的 ranking 升级。

`LTR` 是 Learning to Rank，学习排序。

普通 BM25 是手写规则；LTR 是让模型从数据里学习：

```text
什么样的候选歌曲应该排前面
什么样的候选歌曲应该排后面
```

我们用的是 LightGBM LambdaRank。

`LightGBM` 是一个树模型库，适合表格特征。

`LambdaRank` 是一种专门优化排序的训练目标。

它不是简单判断一首歌“相关/不相关”，而是学习如何让正确歌曲在一个候选列表里排得更靠前。

### 特征

模型看到的特征包括：

```text
各召回源排名
RRF 分数
歌名/艺人/专辑匹配
标签重叠
年份
流行度
历史正反馈相似度
历史负反馈相似度
conversation_goal 类别
turn_number
user profile
```

`Feature` 是特征，也就是模型用来判断的输入数字或类别。

### OOF

`OOF` 是 out-of-fold。

意思是：

```text
把 dev 切成 5 份
每次用 4 份训练
用剩下 1 份预测
最后拼起来评估
```

这样每条 dev 预测都来自“没有见过这条数据”的模型，比直接训练再测自己更可信。

### 结果

LTR 从 baseline 的：

```text
nDCG@20 = 0.085870
```

大幅提升到：

```text
nDCG@20 = 0.180947
```

再调参后：

```text
120 trees + reg_lambda=2
nDCG@20 = 0.183021
```

这说明：

```text
学习排序是真正有效的 ranking 方向
```

## 15. 第十阶段：LTR 调参

`Hyperparameter` 是超参数，也就是模型训练前设置的参数。

我们试了很多：

```text
n_estimators       树的数量
num_leaves         每棵树的复杂度
learning_rate      学习率
reg_lambda         L2 正则
reg_alpha          L1 正则
min_child_samples  叶子节点最少样本
subsample          行采样
colsample_bytree   列采样
lambdarank_truncation_level 排序优化关注前多少名
max_candidates_per_group 每轮候选数量
```

术语解释：

`Regularization / 正则化` 是防止模型过度记住训练数据。

`L2 正则` 会惩罚过大的模型权重，让模型更稳。

`Learning rate` 是每次学习迈多大步。太大容易过拟合，太小可能学不够。

`Overfit / 过拟合` 是在本地数据上好，在新数据上差。

调参结论：

```text
120 trees 最稳
31 leaves 比 63 leaves 更稳
learning_rate 0.04 比 0.06 更稳
reg_lambda=2 有帮助
candidate pool 300 比 200/500 更好
row bagging 没帮助
L1 没帮助
truncation 100 没帮助
```

很多单折看起来好的设置，五折 OOF 会变差。

策略改变：

```text
只相信五折 OOF
不因为某一个 fold 小涨就升级提交包
```

## 16. 第十一阶段：模型集成

`Ensemble` 是模型集成。意思是把多个模型的排序结果合起来，减少单个模型偶然犯错。

我们用 RRF 合并多个 LTR 模型：

```text
120-tree L2
140-tree L2
200-tree L2
120-tree colsample=1.0 L2
```

`colsample_bytree=1.0` 是让树使用全部特征列。它单独不是最强，但和其他模型有互补。

结果：

```text
single 120-tree L2       nDCG@20 = 0.183021
3-model ensemble         nDCG@20 = 0.183253
4-model ensemble         nDCG@20 = 0.183482
weighted 4-model RRF     nDCG@20 = 0.183924
```

最后选择：

```text
rrf_k=26
weights=[1.0, 0.5, 1.3, 1.0]
```

`Weight` 是权重。权重大，说明更信任该模型。

策略改变：

```text
主 ranking 使用 weighted four-model RRF
```

## 17. 第十二阶段：高风险 ranking 尝试与拒绝

我们没有只保留成功实验，也记录了大量失败实验。

这些失败很重要，因为它们告诉我们不要继续浪费提交次数。

### Category segmented selection

`Segmented selection` 是按类别选不同模型。

比如：

```text
category A 用 ensemble
category C/H/I/J 用 140-tree
category G/K 用 200-tree
其他用 120-tree
```

表面结果：

```text
非嵌套 OOF nDCG@20 = 0.184069
```

看起来超过主包。

但更严格的 nested validation 降到：

```text
约 0.18235
```

`Nested validation` 是更严格的验证方式，模拟“你不能用测试结果来选策略”。

结论：

```text
这是 dev 过拟合
拒绝升级
```

### Consensus fallback

`Consensus fallback` 是共识回退。

意思是：如果 ensemble 的 top1 没有任何单模型支持，就退回单模型。

结果：

```text
dev OOF 从 0.183924 到 0.183986
但 Blind A 改 0 条
```

结论：

```text
对 Blind A 没用
保留给未来 Blind B
```

### Batch repeat repair

`Batch repeat repair` 是全局去重复。想减少总是推荐同一首歌。

结果：

```text
catalog diversity 上升
nDCG 下降
```

结论：

```text
Blind A catalog 已接近上限，不值得牺牲 nDCG
```

### Quoted-title promotion

`Quoted-title promotion` 是如果用户话里有引号歌名，就把同名候选往前提。

结果：

```text
改动很多
只改善少数
伤害大量样本
nDCG 明显下降
```

结论：

```text
规则太粗暴，拒绝
```

### Cross-encoder reranking

`Cross-encoder` 是一种神经网络精排模型。它把 query 和 candidate 一起读，理论上更懂语义。

我们用 MiniLM 做零样本探测。

`Zero-shot` 是不针对本比赛训练，直接拿现成模型用。

结果：

```text
CE-only 大幅下降
lock15 仍下降 0.00912
```

`lock15` 是保护前 15 名不动，只让模型改后面。

结论：

```text
零样本 cross-encoder 不适合直接上最终提交
```

### Case-neighbor

`Case-neighbor` 是找训练集中相似对话，把它们的答案搬过来当候选。

结果：

```text
gold exact coverage 只有 6.25%
gold artist coverage 28.13%
lock15 仍明显伤 nDCG
```

结论：

```text
直接搬训练 case 不行
```

## 18. 第十三阶段：最终 response 选择

最终几个 response 方案：

```text
judge_clean_mix        主包，更自然
lexplus_softened       备选，词汇更多，但更机械
compact_clean          词汇更多，更像模板
compact_broad          词汇最高，但可能有脏标签风险
judge_clean_mix_plus   曾是备选，后来被审计淘汰
judge_clean_smooth     自然化实验，失败
```

主包本地：

```text
Blind A local Distinct-2 = 0.60604
noisy hits = 0
long rows = 0
short rows = 0
```

lexplus_softened 本地：

```text
Blind A local Distinct-2 = 0.63566
noisy hits = 0
long rows = 0
short rows = 0
```

为什么没有先交 lexplus_softened？

因为它更像 compact 模板。它可能提高 lexical，但也可能让 LLM judge 觉得不自然。

这次主包 public：

```text
lexical_diversity = 0.6060
llm_judge_score = 1.5
```

说明：

```text
主包 lexical 已经够高
现在更该提升 judge，而不是继续堆 lexical
```

## 19. 第十四阶段：Final Freeze Audit

`Final Freeze Audit` 是最终冻结审计。

它不是为了涨分，而是为了避免提交事故。

它检查：

```text
prediction.json 是否能解析
是否正好 80 条
每条是否 20 个 track_id
track_id 是否都合法
每条内部是否重复
zip 里是否只有 prediction.json
ranking hash 是否符合预期
response hash 是否符合预期
是否有 noisy tags
是否有过长或过短回复
top1 歌名和艺人是否出现在解释中
```

`Hash` 是文件或内容的指纹。只要内容变了，hash 就会变。

最终审计结果：

```text
primary 通过
lexplus_softened 通过
judge_clean_mix_plus 不通过
```

`judge_clean_mix_plus` 不通过原因：

```text
1 个 noisy tag 泄漏
2 个过长回复
```

策略改变：

```text
judge_clean_mix_plus 从备选列表删除
最终只保留 primary 和 lexplus_softened
```

## 20. 最新 public 结果解释

最新提交主包后得到：

```text
nDCG@20            0.1898
catalog_diversity  0.0317
lexical_diversity  0.6060
llm_judge_score    1.5000
composite_score    0.1962
```

与上一版对比：

```text
nDCG@20            0.1935 -> 0.1898   小降
catalog_diversity  0.0257 -> 0.0317   上升
lexical_diversity  0.0125 -> 0.6060   巨幅上升
llm_judge_score    1.0000 -> 1.5000   上升
composite_score    0.1006 -> 0.1962   接近翻倍
```

这说明：

```text
1. LTR ranking 在 Blind A 没有比旧 ranking 更好
2. response 改造非常有效
3. catalog diversity 已接近上限
4. judge 仍然偏低，是下一阶段重点
```

## 21. 这一路踩过的主要坑

### 坑 1：本地 dev 提升不一定等于 public 提升

LTR 在 dev OOF 很强，但 Blind A nDCG 没涨。

原因可能是：

```text
Blind A 分布和 dev 不完全一致
Blind A 只有 80 条，波动很大
LTR 学到的模式在 public 上没完全转移
```

### 坑 2：召回源越多不一定越好

新召回源能找到更多正确歌，但简单融合会打乱原来正确的头部排序。

### 坑 3：多样性不是越高越好

Catalog diversity 在 Blind A 上限很低，过度追它会伤 nDCG。

### 坑 4：回复越自然不一定 lexical 越高

自然化 smooth probe 看起来更人话，但 lexical diversity 掉到 `0.16081`，还变长。

### 坑 5：高 lexical 模板可能伤 judge

compact_broad 词汇最高，但机械感更强，也可能带脏标签，所以不能盲交。

### 坑 6：Pro 建议也要用本地实验验证

很多 Pro 方向有启发，但最终是否采用，要看：

```text
dev OOF
Blind-like panels
final audit
public feedback
```

## 22. 当前最重要结论

第一，response 问题已经大幅修复。

```text
LexDiv 0.0125 -> 0.6060
Judge 1.0 -> 1.5
```

第二，catalog diversity 已接近 Blind A 上限。

```text
0.0317 / 0.0340
```

第三，ranking 仍然是关键瓶颈。

```text
最新 nDCG@20 = 0.1898
上一版 nDCG@20 = 0.1935
```

第四，下一阶段不能继续盲目堆 lexical。

因为 lexical 已经很高，继续提升 lexical 的边际收益小。真正大头是 judge 和 nDCG。

## 23. 现在各提交包的意义

主包：

```text
blindset_A_PRIMARY_submission.zip
```

特点：

```text
ranking 使用 weighted four-model LTR RRF
response 使用 judge_clean_mix
public 总分 0.1962
```

备选包：

```text
blindset_A_BACKUP_lexplus_softened_submission.zip
```

特点：

```text
ranking 与主包完全一样
response 更 compact
lexical 本地更高
但可能更机械
```

现在不建议盲交备选，因为：

```text
主包 LexDiv 已经 0.6060
备选最多可能小涨 lexical
但如果 judge 降，损失更大
```

## 24. 下一步建议

下一步重点不是继续冲 lexical，而是：

```text
1. 分析 public nDCG 为什么没涨
2. 做 judge-oriented response，但保持 LexDiv 不崩
3. 找一种更稳的 Blind A/B ranking 策略
```

具体方向：

```text
方向 A：基于旧 public nDCG 较强的 legacy ranking 做 response-only 新包
方向 B：做 judge-focused response，不使用 compact-heavy 模板
方向 C：对 LTR 和 legacy 的差异行做 public-like 诊断
方向 D：重新研究哪些 session 类别 LTR 伤了 Blind A
方向 E：不要再无脑尝试 cross-encoder、case-neighbor、embedding 直接上榜
```

## 25. 给小白的最终复盘

我们一开始的问题是：

```text
baseline 能推荐，但回复很烂
```

我们先做了很多召回和融合，发现：

```text
多路召回有潜力，但简单融合会伤头部排序
```

然后做了 LTR，发现：

```text
学习排序在 dev 上非常强，但 public Blind A 没完全转移
```

接着 public 反馈显示：

```text
response 才是第一轮最大短板
```

于是集中修 response，结果：

```text
总分 0.1006 -> 0.1962
```

当前局面是：

```text
response 多样性已经打通
catalog 已接近上限
judge 还低
ranking 需要重新诊断
```

所以 0.1962 不是终点，但这一轮不是白跑。它证明：

```text
我们的工程流程、提交格式、审计体系、response 生成和多样性控制已经有效
下一阶段要从“能写多样回复”升级到“能写让 judge 更满意的回复”
同时要解决 LTR 在 Blind A 上 nDCG 不增反降的问题
```
