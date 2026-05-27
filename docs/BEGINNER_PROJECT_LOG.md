# GoalFlow-MusicCRS 小白版模块日志

这份文档是重新拆过的版本。

旧文档是按时间顺序写的：今天做了粗排，明天做回复，后天又回到 rerank。那种写法对做项目的人方便，但对第一次看推荐系统比赛的人很累。

所以这里改成按系统模块讲：

```text
0. 先讲比赛怎么提交和怎么打分
1. 再讲 baseline 到底是什么
2. 单独讲粗排/召回从头到尾怎么迭代
3. 单独讲 rerank 从头到尾怎么迭代
4. 单独讲 response 和 diversity 怎么迭代
5. 最后讲现在的提交包、结果、问题和下一步
```

原来的流水账没有删除，保存在：

```text
docs/BEGINNER_PROJECT_LOG_CHRONOLOGY.md
```

## 0. 先把比赛流程讲清楚

这个比赛不是让我们在线上服务器里实时跑模型。至少 Blind A 这个阶段，我们交的是一个结果文件。

你可以把它理解成考试：

```text
官方给题目：Blind A 输入数据
官方藏答案：每一轮真正的 gold track
我们写答题卡：prediction.json
平台来阅卷：CodaBench 用隐藏答案打分
```

`Prediction` 在这里就是“预测结果”。它不是训练集，也不是官方答案，而是我们给隐藏测试输入做出来的答题卡。

每条预测长这样：

```json
{
  "session_id": "某个对话 ID",
  "user_id": "某个用户 ID",
  "turn_number": 3,
  "predicted_track_ids": [
    "第 1 个候选 track_id",
    "第 2 个候选 track_id"
  ],
  "predicted_response": "给用户看的英文推荐回复"
}
```

其中：

```text
predicted_track_ids:
  必须是官方曲库里的 track_id。
  最多 20 个。
  不能重复。
  顺序很重要，第 1 个最重要。

predicted_response:
  给用户看的自然语言回复。
  会影响 lexical diversity 和 LLM judge。
```

最终提交包是：

```text
submission.zip
  └── prediction.json
```

所以平台打分时做的是：

```text
读取 prediction.json
检查格式是否合法
检查每个 session × turn 是否都有预测
检查 track_id 是否都在官方曲库
用隐藏 gold track 算 nDCG@20
统计 catalog diversity
统计 response 的 lexical diversity
用 LLM judge 看回复质量
```

这也解释了为什么“没有线上推理时间限制”不等于“可以无脑让 GPT 看所有歌”。

我们本地当然可以花很久生成 `prediction.json`，但是如果让大模型逐一判断：

```text
80 条 Blind A 样本 × 47071 首歌 ≈ 376 万个 query-track pair
```

这对 LLM 或 cross-encoder 来说成本极高，而且大模型也不一定知道官方 synthetic agent 当时会选哪首歌。

## 1. 指标先讲清楚

### nDCG@20

`nDCG@20` 是推荐排序主指标。

白话：

```text
正确歌排第 1：最好
正确歌排第 2：少一点分
正确歌排第 20：还有一点点分
正确歌没进前 20：这一轮 0 分
```

所以推荐系统里最重要的是：

```text
不要只把正确歌找出来
还要把它排到足够靠前
```

### hit@20

`hit@20` 只是问：

```text
正确歌有没有出现在前 20？
```

它不关心第 1 还是第 20。

这就是为什么有些 source 的 `hit@20` 很高，但最终 `nDCG@20` 不一定高。

如果正确歌经常排第 18、第 19、第 20，那么：

```text
hit@20 看起来不错
nDCG@20 仍然很低
```

### Catalog Diversity

`Catalog Diversity` 是曲库覆盖多样性。

如果系统总推荐那几首热门歌，分数低。如果 80 条提交里覆盖更多不同歌曲，分数高。

Blind A 只有：

```text
80 条 × 每条 20 首 = 1600 个推荐位置
```

官方曲库大约：

```text
47071 首歌
```

所以 Blind A 的 catalog diversity 理论上限大约是：

```text
1600 / 47071 ≈ 0.0340
```

我们这次 public 是：

```text
catalog_diversity = 0.0317
```

已经接近 Blind A 的上限，所以继续为 diversity 大幅牺牲 nDCG 不划算。

### Lexical Diversity / Distinct-2

`Lexical Diversity` 是文本多样性。

这里可以理解成 Distinct-2：统计回复里连续两个词组成的短语，有多少是不重复的。

如果 80 条回复都写：

```text
Here are some songs you might enjoy.
```

那就非常重复，lexical diversity 很低。

如果每条回复都具体提到不同歌名、艺人、专辑、年代、标签、历史反馈，lexical diversity 会高很多。

### LLM-as-a-Judge

`LLM-as-a-Judge` 是让大语言模型当裁判，看回复质量。

它不是直接评价歌曲排序，而是看文字回复是不是：

```text
个性化
解释具体
能承接历史反馈
不胡编 metadata 里没有的信息
像一个正常音乐推荐助手
```

我们现在 public judge 是：

```text
llm_judge_score = 1.5
```

这仍然偏低。

## 2. Baseline 到底是什么

`Baseline` 是最基础的可运行方案。它的意义不是最强，而是：

```text
能读数据
能搜曲库
能输出 prediction.json
能被 evaluator 正常打分
```

官方 baseline 的核心是：

```text
对话历史 -> BM25 文本搜索 -> top-20 歌曲 -> 生成回复
```

`BM25` 是经典关键词搜索算法。你可以把它想成“比普通关键词匹配更聪明的搜索器”。

它会看：

```text
query 里有哪些词
歌曲文档里有哪些词
这些词是不是少见
这些词在文档里出现得多不多
```

### Baseline 搜的歌曲文档格式

这个项目里的 baseline BM25 配置使用这些字段：

```text
track_name
artist_name
album_name
release_date
```

所以一首歌在索引里的文本固定类似：

```text
track_name: Big Poppa
artist_name: The Notorious B.I.G.
album_name: Ready To Die
release_date: 1994-09-13
```

这里的 `Big Poppa` 只是例子。真正内容来自官方 track metadata。

注意：这不是“或者”。格式是确定的。

### Baseline 搜索 query 怎么来

baseline 会把当前对话历史拼起来。

如果历史里出现过系统推荐的 `track_id`，它会把这个 `track_id` 转回歌曲信息，再拼进对话。

粗略像这样：

```text
user: I want something energetic.
assistant: track_id: xxx, track_name: ..., artist_name: ..., album_name: ..., release_date: ...
user: That's closer, but I want more punk.
```

然后把这整段作为 BM25 query，去搜刚才那种歌曲文档。

baseline 的早期 dev 结果：

```text
nDCG@20 = 0.085870
```

它说明 baseline 有一定能力，但还远不够。

## 3. 粗排/召回模块，从头到尾怎么迭代

这一章只讲粗排，也叫召回。

`Recall` 是召回。意思是：

```text
先从 47071 首歌里捞出一批可能相关的候选歌。
```

召回阶段不要求完美排序，但必须尽量别漏掉正确答案。

推荐系统常见结构是：

```text
全曲库 47071 首
  -> 召回几百到几千首候选
  -> rerank 再排出最终 top-20
```

如果召回阶段没把正确歌放进候选池，后面的 rerank 再强也没用。

### 3.1 粗排的初始目标

我们一开始想做的是：

```text
baseline 只用一个保守 BM25 搜索
GoalFlow 用多个搜索视角
希望多路搜索能覆盖更多正确歌
```

这不是荒唐想法。因为比赛里的用户请求很复杂：

```text
有人按歌名找
有人按艺人找
有人按专辑找
有人按年代找
有人按 mood 找
有人按历史反馈说“更像上一首，但别那么重”
有人按 cover/image clue 找
```

单一 BM25 很难同时覆盖这些情况。

### 3.2 我们建立了哪些 index

`Index` 是索引，也就是一种搜索目录。

同一首歌可以被写成不同文档，放到不同 index 里。这样同一个 query 可以从不同角度搜。

#### legacy_metadata

用途：

```text
接近 baseline 的老格式，作为保守核心 source。
```

格式：

```text
track_name: Big Poppa
artist_name: The Notorious B.I.G.
album_name: Ready To Die
release_date: 1994-09-13
```

它适合：

```text
用户明确提到歌名
用户明确提到艺人
用户明确提到专辑
用户明确提到年代
```

#### metadata_all

用途：

```text
比 legacy 更宽，把 tag 和 popularity 也加进去。
```

格式：

```text
track_name: Big Poppa
artist_name: The Notorious B.I.G.
album_name: Ready To Die
tag_list: hip hop, rap, east coast, 90s
release_date: 1994-09-13
popularity: 0.83
```

它适合：

```text
用户说 genre、mood、年代、风格标签
```

风险：

```text
tag 里有噪声，可能把不专业的标签也算进去。
```

#### title_artist

用途：

```text
专门强化歌名和艺人匹配。
```

格式：

```text
track_name: Big Poppa
artist_name: The Notorious B.I.G.
```

它适合：

```text
specific track
也就是用户想找一首具体歌
```

#### album_artist

用途：

```text
专门强化专辑和艺人。
```

格式：

```text
album_name: Ready To Die
artist_name: The Notorious B.I.G.
release_date: 1994-09-13
```

它适合：

```text
用户围绕某张专辑、某个时期、某个艺人作品找歌
```

#### tags

用途：

```text
只看标签。
```

格式：

```text
tag_list: hip hop, rap, east coast, 90s
```

它适合：

```text
用户说 dreamy、energetic、punk、jazz、sad、danceable 等风格或情绪
```

风险：

```text
只看 tag 容易变宽，可能找来很多同风格但不是正确答案的歌。
```

#### enriched

用途：

```text
metadata + 训练对话增强文本。
```

格式类似：

```text
track_name: Big Poppa
artist_name: The Notorious B.I.G.
album_name: Ready To Die
tag_list: hip hop, rap, east coast, 90s
release_date: 1994-09-13
popularity: 0.83

training_user_query: I'm looking for a classic 90s East Coast rap track...
training_goal: find one specific song with a smooth hip-hop feel...
training_goal_category: ...
music_selection_reason: ...
assistant_explanation: ...
```

`Document augmentation` 就是这个。

它的目的：

```text
不是只让歌靠 metadata 被搜到，
而是让歌也能靠训练集中用户曾经怎么描述它而被搜到。
```

### 3.3 我们用了哪些 query

`Query` 是搜索输入。

baseline 主要用历史对话拼起来的 query。

GoalFlow 增加了多个 query variant。

#### legacy_history

用途：

```text
复刻 baseline 风格。
```

格式类似：

```text
user: I want something energetic.
assistant: track_id: xxx, track_name: ..., artist_name: ..., album_name: ..., release_date: ...
user: That's closer, but more punk.
```

#### current

只搜当前用户这一句话：

```text
That's closer, but more punk.
```

优点：

```text
不被长历史干扰。
```

缺点：

```text
可能丢掉前文目标。
```

#### goal

只搜 `conversation_goal`。

`conversation_goal` 是官方数据里对整段对话目标的描述。

优点：

```text
通常比用户某一句话更完整。
```

风险：

```text
如果 goal 太笼统，可能带来宽泛候选。
```

#### current_goal

把当前用户请求和 conversation_goal 拼在一起。

目的：

```text
既保留当前微调，又保留全局目标。
```

#### seed_current

把历史正反馈/负反馈歌曲也拼进去。

`Positive seed` 是之前被判断为接近目标的歌。

`Negative seed` 是之前不接近目标的歌。

目的：

```text
用户说“上一首更接近了”时，继续沿着那首歌附近搜。
用户说“不对”时，尽量避开那个方向。
```

#### quoted_entities

如果用户话里有引号内容，就单独拿出来搜。

例如：

```text
I'm looking for "The Lock Down Denial"
```

那 quoted query 就是：

```text
The Lock Down Denial
```

### 3.4 什么叫 source

前面你问过 `source` 是什么。

在我们这里：

```text
source = 某个 index + 某个 query variant 的组合
```

例如：

```text
legacy_metadata:legacy_history
metadata_all:current
metadata_all:goal
title_artist:quoted_entities
enriched:current_goal
tags:seed_current
```

它们都是不同 source。

每个 source 都会返回一串候选歌排名。

### 3.5 第一版融合：裸 RRF

多个 source 各自返回排名后，需要合并。

我们一开始用 RRF。

`RRF` 是 Reciprocal Rank Fusion，倒数排名融合。

公式：

```text
某首歌在某个 source 的贡献 = weight / (rrf_k + rank)
```

总分：

```text
某首歌总分 = 所有 source 贡献相加
```

比如 `rrf_k = 60` 且 `weight = 1`：

```text
rank 1:   1 / 61  = 0.01639
rank 20:  1 / 80  = 0.01250
rank 100: 1 / 160 = 0.00625
```

`rrf_k` 越大，曲线越平，后面的排名也还有一定分。

`rrf_k` 越小，曲线越陡，前几名更值钱。

早期粗排结果：

```text
goalflow_bm25_aug_v1 nDCG@20 = 0.067874
goalflow_bm25_aug_v2 nDCG@20 = 0.074497
baseline              nDCG@20 = 0.085870
```

严格说，只看这三行只能证明：

```text
裸 RRF 最终 top-20 排序变差。
```

不能只凭这三行说“召回一定变好了”。

“召回有潜力”这个判断来自后面的 source diagnostics。

### 3.6 Source diagnostics 以后才看清楚问题

`Source diagnostics` 是把每个 source 单独拆开看。

它回答三个问题：

```text
1. 每个 source 单独能不能找到 gold track？
2. gold track 在每个 source 排第几？
3. RRF 融合后 gold track 排第几？
```

`Gold track` 是隐藏答案里的正确歌。devset 上我们知道 gold，所以可以诊断。

诊断结果：

```text
legacy source hit@20      = 0.2303
best single source hit@20 = 0.4715
RRF fused hit@20          = 0.2595
```

解释：

```text
legacy source:
  接近 baseline 的 source。
  它有 23.03% 的轮次能把正确歌放进前 20。

best single source:
  事后看，每一轮从所有 source 里挑那个最会找 gold 的 source。
  它能到 47.15%。
  这不是可提交模型，因为真实 blind 没有 gold，不知道该挑哪个 source。

RRF fused:
  实际融合后只有 25.95%。
```

这说明：

```text
新 source 里确实有人能找到更多正确歌。
但是裸 RRF 没学会该相信哪个 source。
```

这才是“多路召回有潜力，但融合方式不够聪明”的证据。

### 3.7 为什么 RRF 会把头部冲散

一个例子：

```text
legacy source:
1. Gold Song
2. Song A
3. Song B

tags source:
1. Song X
2. Song Y
3. Song Z

enriched source:
1. Song X
2. Song M
3. Song N

metadata_all source:
1. Song X
2. Song P
3. Song Q
```

Gold Song 只被 legacy 强力支持。

Song X 被三个新 source 都排第 1。

RRF 会让 Song X 总分超过 Gold Song。

这就叫：

```text
head-rank dilution
```

意思是：

```text
原来头部正确答案被多个弱相关 source 的投票稀释了。
```

### 3.8 Legacy Head Protection 和 response-only 对照

裸 RRF 伤了排序后，我们做了：

```text
head10: 前 10 名沿用 legacy，后面用 GoalFlow
head20: 前 20 名沿用 legacy，只改 response
```

结果：

```text
head10 nDCG@20 = 0.083776
head20 nDCG@20 = 0.085870
```

这里要特别小心。

`head20` 不是一个真正的粗排优化实验。

因为比赛最终只提交 top-20。如果 top-20 全部沿用 baseline，那么推荐排序就等于 baseline。

所以：

```text
head20 nDCG@20 = baseline nDCG@20
```

这不是发现了新排序方法，只是一个 response-only 对照组。

它的作用是隔离变量：

```text
ranking 不动
只看 response 改造会不会涨总分
```

真正有一点粗排信息量的是 `head10`。

`head10` 的意思是：

```text
前 10 名保护 baseline
后 10 名换成 GoalFlow
```

结果它从 `0.085870` 降到 `0.083776`。

这说明：

```text
当时的 GoalFlow 粗排即使只替换后 10 个位置，也没有带来推荐排序收益。
```

所以这节的准确结论不是“head20 保住 baseline 很厉害”，而是：

```text
head20 = baseline ranking freeze，用来测 response。
head10 = 粗排替换后半段，结果仍然伤 nDCG。
```

### 3.9 Tail Diversity 也是安全边界，不是召回突破

`Tail` 是后半段，比如第 16 到第 20 名。

我们试过只改尾部来换 diversity。

结果：

```text
taildiv_head10 nDCG@20 = 0.072093
taildiv_head15 nDCG@20 = 0.081796
taildiv_head18 nDCG@20 = 0.085271
taildiv_head19 nDCG@20 = 0.085758
```

结论：

```text
动得越多，nDCG 越容易掉。
只动第 20 名相对安全。
```

这说明 diversity 只能保守做，不能拿它替代召回能力。

### 3.10 Progress label 审计修正了 seed 用法

数据里有 `goal_progress_assessments`。

它描述前一次推荐是否让用户更接近目标。

我们一开始不确定：

```text
turn t 的 label 是评价 turn t？
还是评价 turn t-1？
```

审计后发现：

```text
turn 1 没有 label
turn 2 的 label 是用户对 turn 1 推荐的反馈
```

所以应该后移使用：

```text
历史推荐 m 的反馈 = progress[m + 1]
```

这个修正让 positive seed 和 negative seed 更可信。

### 3.11 Train-context augmentation 的结论

文档增强的想法是对的：

```text
把训练集中导致某首歌被推荐的用户说法，追加到这首歌的文档里。
```

诊断结果里：

```text
legacy source hit@20       = 0.2303
index_any=enriched hit@20  = 0.3685
index_any=metadata hit@20  = 0.3635
RRF fused hit@20           = 0.2595
```

这说明：

```text
enriched source 单独看，比 legacy 更容易把正确歌放进前 20。
但 RRF 没有把这个潜力转化成最终 top-20。
```

所以准确结论是：

```text
augmentation 增加了候选覆盖潜力。
但需要更强的 rerank 或 source gating 才能兑现。
```

### 3.12 Embedding 探索

`Embedding` 是向量表示。

可以把它理解成：

```text
把一首歌、一段歌词、一张封面、一个用户，变成一串数字。
如果两串数字方向接近，就表示它们在某种意义上相似。
```

官方提供了很多 embedding：

```text
metadata-qwen3
lyrics-qwen3
attributes-qwen3
audio-laion_clap
image-siglip2
cf-bpr
```

解释：

```text
metadata-qwen3:
  歌名、艺人、专辑等文本的语义向量。

lyrics-qwen3:
  歌词语义向量。

attributes-qwen3:
  风格、情绪、乐器等属性向量。

audio-laion_clap:
  音频向量。

image-siglip2:
  封面图像向量。

cf-bpr:
  协同过滤向量，根据用户行为学习“谁可能喜欢什么”。
```

这里要说清楚：我们不是把所有 embedding 方向都完整做透了。我们做的是几种低风险接入方式，看看它们能不能稳定超过当时主线。

#### embedding schema 检查

`Schema` 是数据结构。

这一步不是推荐实验，而是确认：

```text
官方 embedding 文件里有哪些列
每种向量是多少维
track_id 是否和当前官方曲库对得上
有没有用错旧版本 embedding
```

检查结果：

```text
all_tracks embedding 有 47071 行
和官方曲库 47071 首歌能对齐
cf-bpr 是 128 维
audio 是 512 维
image 是 768 维
metadata / lyrics / attributes 是 1024 维
```

这一步的作用是修地基：保证后面不会拿错向量。

#### seed-CF tail rescue

这个名字拆开看：

```text
seed:
  历史里用户反馈过的参考歌。

CF:
  collaborative filtering，协同过滤。
  大概意思是“喜欢这首歌的人，还喜欢哪些歌”。

tail rescue:
  只救尾部，不动前面高价值位置。
```

具体做法：

```text
1. 保护原 top19 不动。
2. 找最近的 positive seed。
3. 用 track_cf 向量找和这个 seed 相似的歌。
4. 如果找到一个没出现过、也不是 negative seed 的候选，就插到第 20 位附近。
```

为什么这么保守？

```text
因为 nDCG 很看重前几名。
直接让 embedding 改 top1/top5 风险太大。
先只试它能不能在尾部捞一点收益。
```

结果：

```text
收益极小或不稳定。
```

这说明：

```text
CF embedding 可能有信号，
但只靠“拿 seed 的相似歌插尾部”这个简单策略，不足以稳定涨分。
```

#### embedding LTR features

这一步不是直接把 embedding 搜出来的歌塞进提交。

它是把 embedding 做成 LTR 的特征，让 LightGBM 自己判断要不要用。

比如每个候选歌会多几个数字：

```text
这个候选和 positive seed 的 embedding 相似度有多高
这个候选和 negative seed 的 embedding 相似度有多高
positive 相似度 - negative 相似度是多少
这个用户的 user_cf 向量和这首歌的 track_cf 向量有多匹配
```

也就是说：

```text
embedding 没有直接决定排序。
它只是给 rerank 模型多提供几列信息。
```

我们试了几组：

```text
track_cf + user_cf:
  用协同过滤向量描述“用户可能喜欢什么”和“歌曲之间谁像谁”。

metadata seed cosine:
  用 metadata-qwen3 向量算候选歌和 positive/negative seed 的语义相似度。

attributes seed cosine:
  用 attributes-qwen3 向量算候选歌和 seed 在风格、情绪、属性上的相似度。
```

`Cosine` 是余弦相似度，可以理解成两个向量方向像不像。

验证结果：

```text
track_cf + user_cf      nDCG@20 = 0.183415
metadata seed cosine    nDCG@20 = 0.183013
attributes seed cosine  nDCG@20 = 0.178924
```

对比当时最终主 ranking：

```text
weighted four-model RRF nDCG@20 = 0.183924
```

所以结论不是“embedding 没用”。

更准确是：

```text
我们试过的这些低风险 embedding 接法，没有超过最终主线。
```

其中：

```text
track_cf + user_cf:
  有一点信号，但没有超过 ensemble 主包。

metadata seed cosine:
  基本接近单模型主线，但没有新增明显收益。

attributes seed cosine:
  明显伤分，说明这个属性相似度直接喂给当前 LTR 可能会误导模型。
```

#### 这里没有做透的 embedding 方向

还没有系统做完的包括：

```text
用 embedding 做真正的候选召回 source
按题型选择不同 embedding，例如 cover_art 用 image，mood 用 attributes/audio
给 query 本身编码，再和 track embedding 做 dense retrieval
把 lyrics / audio / image 多模态信号做成独立 source diagnostics
专门训练一个 embedding-aware reranker
```

所以这部分的真实结论应该是：

```text
embedding 方向有价值，但我们只做了保守接入和少量验证。
当前几种接法没有稳定超过主线，所以没有进 Blind A 主包。
它不是被证明“没用”，而是还没有被充分开发。
```

这也是为什么下一步如果继续冲 nDCG，embedding 应该回到“召回诊断”里重新做，而不是只当尾部补丁或几列 LTR 特征。

### 3.13 粗排模块当前真实评价

这部分要说实话。

我们不是没有做粗排，但粗排还没有被做到“成熟系统”的程度。

已经做了：

```text
多 index
多 query variant
RRF 融合
source diagnostics
train-context augmentation
progress feedback seeds
embedding 试探
legacy head protection
tail diversity 安全边界
```

发现了：

```text
新增 source 有候选覆盖潜力
裸 RRF 会伤头部排序
enriched source 有价值
embedding 直接上不稳
```

还没有完全补上的关键表：

```text
候选池 recall@20 / recall@100 / recall@300 / recall@1200
按 turn_number 分组
按 conversation_goal.category 分组
按 specificity 分组
按 query 类型分组
```

这就是下一轮真正应该优先做的粗排诊断。

## 4. Rerank 模块，从头到尾怎么迭代

`Rerank` 是重排。

召回阶段先拿到候选池，rerank 再决定最终顺序。

一句话：

```text
召回负责“别漏掉”
rerank 负责“排前面”
```

### 4.1 为什么需要 rerank

裸 RRF 暴露了一个问题：

```text
不同 source 的可信度不是固定的。
```

比如：

```text
specific track 题：
  title_artist 和 quoted_entities 更可信。

mood playlist 题：
  tags、attributes、history seed 可能更可信。

album/artist 题：
  album_artist、legacy_metadata 更可信。
```

RRF 不懂这些，它只会加排名分。

所以我们需要一个模型学习：

```text
在什么情况下应该相信哪个 source？
什么样的候选更像 gold track？
```

这就是 LTR。

### 4.2 LTR 是什么

`LTR` 是 Learning to Rank，学习排序。

我们用的是：

```text
LightGBM LambdaRank
```

解释：

```text
LightGBM:
  一个树模型库。
  很适合表格特征。

LambdaRank:
  一种专门为排序问题设计的训练目标。
  它不是只学“是不是正确”，而是学“应该排第几”。
```

### 4.3 LTR 的候选从哪里来

LTR 不是全曲库直接排。

它先拿粗排候选：

```text
多个 BM25 source -> RRF 合并 -> 最多保留 rerank_pool_size=1200
```

训练时每轮最多取：

```text
max_candidates_per_group = 300
```

也就是说：

```text
每个 session × turn 是一个 group。
这个 group 里有若干候选歌。
模型学会把 gold track 排到这些候选歌前面。
```

如果 gold track 不在候选池里，这个 group 对 rerank 来说就是无解的。

所以 rerank 的上限仍然受召回约束。

### 4.4 LTR 用了哪些特征

`Feature` 是特征，也就是模型看到的输入信息。

LTR 特征大致分几类。

#### 召回排名特征

```text
某首歌在 legacy_metadata:legacy_history 里排第几
在 title_artist:current 里排第几
在 enriched:current_goal 里排第几
RRF 总分是多少
有多少 source 支持它
最好的 source rank 是多少
```

这些特征告诉模型：

```text
哪些 source 支持这首歌
支持得有多强
```

#### 实体匹配特征

`Entity` 是实体，比如歌名、艺人、专辑、年份。

特征包括：

```text
歌名是否和 query 匹配
艺人是否匹配
专辑是否匹配
年份是否匹配
标签词是否重叠
```

#### 对话反馈特征

```text
候选歌是否接近 positive seed
是否接近 negative seed
是否同艺人
是否同专辑
当前 turn_number 是多少
```

#### 用户和目标特征

```text
conversation_goal.category
specificity
用户画像
当前请求长度
历史长度
```

### 4.5 OOF 验证

`OOF` 是 out-of-fold。

做法：

```text
把 dev 切成 5 份
每次用 4 份训练
用剩下 1 份预测
最后把 5 份预测拼起来算 nDCG
```

为什么要这样？

因为如果模型训练完又在训练数据上评估，分数会虚高。

OOF 保证：

```text
每条 dev 预测都来自没见过它的模型。
```

### 4.6 LTR 第一轮效果

LTR 让 dev OOF 排序大幅提升：

```text
legacy head20 baseline nDCG@20 = 0.085870
早期 LTR                 nDCG@20 = 0.180947
```

这说明：

```text
学习排序方向是有效的。
```

但注意：

```text
这是 dev OOF。
不等于 Blind A 一定涨。
```

### 4.7 LTR 调参

`Hyperparameter` 是超参数，也就是训练前设置的模型参数。

我们调过：

```text
n_estimators
num_leaves
learning_rate
reg_lambda
reg_alpha
min_child_samples
subsample
colsample_bytree
lambdarank_truncation_level
max_candidates_per_group
```

解释几个关键的：

```text
n_estimators:
  树的数量。

num_leaves:
  每棵树的复杂度。

learning_rate:
  每次学习迈多大步。

reg_lambda:
  L2 正则，防止模型过度记住训练数据。

overfit:
  过拟合。本地看起来好，换到新数据变差。
```

最终较稳设置：

```text
120 trees
reg_lambda=2
31 leaves
learning_rate=0.04
max_candidates_per_group=300
```

结果：

```text
120 trees + reg_lambda=2
nDCG@20 = 0.183021
```

一些被拒绝的尝试：

```text
num_leaves=63:
  单折赢，五折 OOF 掉到 0.18124。

max_candidates_per_group=200:
  候选太少，正样本丢得更多。

max_candidates_per_group=500:
  候选更多，但噪声更多。

reg_lambda=0.1:
  单折看着好，五折 OOF 不如 2。

L1 regularization:
  没赢。

subsample:
  没赢。

lambdarank_truncation_level=100:
  没赢。
```

重要经验：

```text
单 fold 涨不能信。
五折 OOF 才比较可信。
```

### 4.8 模型集成

`Ensemble` 是模型集成。

意思是：

```text
训练多个相近但不完全一样的模型，
把它们的排序结果合并，
降低单个模型偶然犯错的风险。
```

我们用 RRF 合并多个 LTR 模型：

```text
120-tree L2
140-tree L2
200-tree L2
120-tree colsample=1.0 L2
```

结果：

```text
single 120-tree L2       nDCG@20 = 0.183021
3-model ensemble         nDCG@20 = 0.183253
4-model ensemble         nDCG@20 = 0.183482
weighted 4-model RRF     nDCG@20 = 0.183924
```

最终主 ranking：

```text
rrf_k = 26
weights = [1.0, 0.5, 1.3, 1.0]
```

`weight` 是权重，权重大代表更信任该模型。

`rrf_k=26` 比 `60` 更重视模型前几名。

### 4.9 高风险 rerank 尝试

我们还试过一些更冒险的 rerank。

#### Category segmented selection

按 `conversation_goal.category` 选择不同模型。

表面结果：

```text
非嵌套 OOF nDCG@20 = 0.184069
```

但严格 nested validation 降到：

```text
约 0.18235
```

结论：

```text
过拟合，拒绝。
```

#### Consensus fallback

如果 ensemble top1 没有单模型支持，就回退。

结果：

```text
dev OOF 小涨到 0.183986
Blind A 改 0 条
```

结论：

```text
对 Blind A 没实际影响，保留给未来 split。
```

#### Cross-encoder reranking

`Cross-encoder` 是把 query 和 candidate 一起读的神经网络精排模型。

理论上更懂语义。

我们用 MiniLM 做零样本探测。

结果：

```text
lock15 保护前 15 仍下降 0.00912 nDCG@20
只有 3 行变好，11 行变差
```

结论：

```text
零样本 cross-encoder 不适合直接上最终包。
```

#### Case-neighbor

找训练集中相似对话，把它们的答案当候选。

结果：

```text
gold exact coverage 只有 6.25%
lock15 仍下降 0.01788
```

结论：

```text
直接搬训练 case 不行。
```

### 4.10 Rerank 模块当前真实评价

已经做成的：

```text
LTR 从 0.085870 提升到 0.183021 dev OOF
ensemble 提升到 0.183924 dev OOF
高风险 rerank 大多被拒绝
最终 ranking 选择 weighted four-model RRF
```

没有解决的：

```text
Blind A public nDCG 没涨，反而从旧包 0.1935 到主包 0.1898。
```

可能原因：

```text
Blind A 只有 80 条，波动很大
Blind A 分布和 dev 不完全一致
LTR 依赖粗排候选池，召回缺口仍在
dev OOF 最优不一定等于 public 最优
```

所以下一步不能只继续调 rerank。

更应该先判断：

```text
Blind A 这 80 条里，是候选没召回，还是召回了但 LTR 排错？
```

## 5. Response 和 diversity 模块，从头到尾怎么迭代

这一章只讲文字回复和多样性。

第一次 public 结果：

```text
nDCG@20            0.1935
catalog_diversity  0.0257
lexical_diversity  0.0125
llm_judge_score    1.0000
composite_score    0.1006
```

这告诉我们：

```text
当时推荐排序不差。
最大短板是 response。
```

### 5.1 compact response

`compact` 是短而具体的回复。

它会包含：

```text
第一首推荐是什么
为什么匹配
引用哪些 metadata
历史反馈如何影响
后面几首备选是什么
```

效果：

```text
lexical diversity 明显提高
```

### 5.2 compact_broad

`broad` 是引用更多标签。

优点：

```text
词汇更多，Distinct-2 更高。
```

问题：

```text
tag 里有噪声。
```

例如：

```text
albums i own
seen live
songsof2011
lobpreis
```

这些写给用户会显得奇怪。

### 5.3 judge_v2

`judge_v2` 是面向 LLM judge 的回复。

它更强调：

```text
我为什么选第一首
我引用了哪些 metadata
我如何使用历史反馈
我如何使用用户画像作为 tie-breaker
```

`Tie-breaker` 是打破平局的辅助依据。

### 5.4 judge_clean_mix

后来做了混合风格：

```text
judge_v2
judge_brief
compact
```

并清理：

```text
脏标签
重复标题
list-valued artist name
过长回复
过短回复
不自然拒答
```

最终主包使用：

```text
judge_clean_mix
```

public 结果：

```text
lexical_diversity 0.0125 -> 0.6060
llm_judge_score   1.0    -> 1.5
```

这就是总分从 `0.1006` 到 `0.1962` 的最大原因。

### 5.5 lexplus_softened

`lexplus_softened` 是备选回复方案。

特点：

```text
ranking 和主包完全一样
response 更追求词汇多样性
本地 Blind A Distinct-2 = 0.63566
```

为什么没先交它？

```text
它更像模板。
可能 lexical 更高，但 judge 未必更喜欢。
```

### 5.6 Diversity 的结论

我们做过多种 tail diversity 和 repeat repair。

最终结论：

```text
Blind A catalog diversity 已接近上限。
继续追 diversity 收益小。
如果因此伤 nDCG，不值得。
```

## 6. 当前提交包和 public 结果

主包：

```text
blindset_A_PRIMARY_submission.zip
```

特点：

```text
ranking:
  weighted four-model LTR RRF

response:
  judge_clean_mix
```

public 结果：

```text
nDCG@20            0.1898
catalog_diversity  0.0317
lexical_diversity  0.6060
llm_judge_score    1.5000
composite_score    0.1962
```

上一版 public：

```text
nDCG@20            0.1935
catalog_diversity  0.0257
lexical_diversity  0.0125
llm_judge_score    1.0000
composite_score    0.1006
```

解释：

```text
ranking 没涨，nDCG 小降。
diversity 小涨。
response 大涨。
最终总分接近翻倍。
```

备选包：

```text
blindset_A_BACKUP_lexplus_softened_submission.zip
```

特点：

```text
ranking 和主包完全一样
response 更高 lexical
但可能更机械
```

## 7. 现在最重要的真实结论

### 结论 1：粗排没有被充分收口

粗排做过很多探索，但还缺一个系统性的候选池诊断表。

下一步应该先做：

```text
候选池 recall@20 / @100 / @300 / @1200
按 category 分组
按 turn_number 分组
按 specificity 分组
按 source 分组
```

否则很容易继续在 rerank 上空转。

### 结论 2：LTR 在 dev 上强，但 Blind A 没转化

LTR dev OOF 从 `0.085870` 到 `0.183924`，说明学习排序是有效路线。

但 public：

```text
旧包 nDCG@20 = 0.1935
新主包 nDCG@20 = 0.1898
```

说明：

```text
本地 dev 最优不等于 Blind A 最优。
```

### 结论 3：response 修复是这轮最大收益

```text
lexical_diversity 0.0125 -> 0.6060
judge             1.0    -> 1.5
composite          0.1006 -> 0.1962
```

这个模块是成功的。

但 judge 仍低，后面还可以继续优化。

### 结论 4：diversity 不是主攻方向

Blind A 上限约 `0.0340`，我们已有 `0.0317`。

继续为了 diversity 牺牲 nDCG，性价比低。

## 8. 下一步应该怎么干

如果继续优化，不应该再按“想到什么试什么”推进。

应该按这个顺序：

```text
1. 先做粗排候选池诊断
   看 gold 到底有没有进入候选池。

2. 如果 gold 没进入候选池
   继续加强召回 source。

3. 如果 gold 进入候选池但排不上去
   继续优化 LTR / source gating。

4. 如果 nDCG 短期很难涨
   再做 judge-oriented response。

5. diversity 只做不伤头部的安全小修。
```

最应该补的表：

```text
source_name
hit@20
hit@100
hit@300
hit@1200
mean_gold_rank
nDCG@20_if_this_source_alone
```

现在已经补了一个专门的粗排评测脚本：

```text
scripts/evaluate_recall_pool.py
```

它不看 response，也不训练 LTR，只回答粗排问题：

```text
1. 各 source 单独能不能找到 gold
2. 所有 source 的 candidate union 能不能包含 gold
3. 不同 rrf_k 下，gold 在 RRF 粗排里排第几
4. candidate pool 平均有多大
5. 哪些样本 gold 完全没进候选池
```

推荐先跑几组：

```text
retrieval_top_k = 100
retrieval_top_k = 260
retrieval_top_k = 500
retrieval_top_k = 1000
```

每组都看：

```text
candidate_union coverage
RRF hit@20 / hit@100 / hit@300 / hit@1200
pool_size
miss list
```

如果 `candidate_union coverage` 不够高，说明召回还没做好。

如果 `candidate_union coverage` 很高，但 RRF hit@100 很低，说明候选找到了，但是粗排融合很差。

然后再按类型拆：

```text
specific_track
artist_exploration
mood_playlist
cover_art
era_genre
turn_number 1-8
specificity HH / HL / LH / LL
```

只有这张表出来，才能回答：

```text
到底应该继续做召回，还是继续做 rerank？
```

## 9. 一句话复盘

从 baseline 到现在，真正发生的是：

```text
baseline:
  一个保守 BM25 搜索器，排序有点用，但回复很烂。

粗排阶段:
  多 source 能带来更多候选覆盖潜力，
  但裸 RRF 会冲散头部，
  需要更系统的 recall@K 诊断。

rerank 阶段:
  LTR 在 dev OOF 上大幅提升，
  ensemble 进一步小涨，
  但 Blind A public 没转化成 nDCG 提升。

response 阶段:
  从模板废话升级到 metadata-grounded explanation，
  lexical 和 judge 明显上涨，
  这是 public 总分翻倍的主因。

当前瓶颈:
  ranking 仍然没被解决。
  下一步应该先回到粗排候选池诊断，而不是盲目继续堆 rerank 或 diversity。
```
