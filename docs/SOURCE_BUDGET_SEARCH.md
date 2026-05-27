# Source/K Budget Search

这部分专门解决粗排阶段的新目标：

```text
选择每个 source 的 K，
让每个 dev turn 的候选歌合并去重后，平均 size 接近目标值，例如 800，
同时让 gold track 被召回的 turn 数最大。
```

这里的 `turn` 是一条具体评测样本，也就是某个 `session_id + user_id + turn_number`。

## 为什么不用 sum(K)

`sum(K)` 只是理论取数：

```text
source A top100 + source B top200 = sum(K) 300
```

但如果两个 source 返回大量相同歌曲，真实候选池可能只有 180 首。

rerank 真正吃进去的是：

```text
unique(source A candidates ∪ source B candidates)
```

所以我们现在优化真实去重后的候选池大小，而不是优化 `sum(K)`。

## 输入数据

导出脚本：

```bash
PYTHONPATH=goalflow_musiccrs .venv/bin/python goalflow_musiccrs/scripts/export_source_candidate_matrix.py \
  --tid source_candidate_matrix_top800 \
  --source-limit 20 \
  --k-values 0,50,100,200,400,800 \
  --max-k 800
```

输出目录：

```text
goalflow_musiccrs/experiments/<tid>/source_candidate_matrix/
```

关键文件：

```text
meta.txt
  矩阵维度、K 候选值、文件名。

sources.tsv
  每个 source 的编号和名字。

examples.tsv
  每个 turn 的唯一样本信息和 gold track。

candidates.i32
  int32 二进制矩阵。
  形状是 [num_turns, num_sources, max_k]。
  值是压缩后的 track index。

counts.u16
  每个 turn/source 实际有多少候选。

gold.i32
  每个 turn 的 gold track index。

source_single_hit_stats.csv
  每个 source 在每个 K 下单独命中多少 turn。

source_pair_hit_overlap.csv
  两个 source 在同一个 K 下命中的 turn 是否重合。
  这里的重合是 hit gold 的重合，不是候选歌曲列表的重合。
```

## C++ 搜索器

编译：

```bash
mkdir -p goalflow_musiccrs/bin
clang++ -O3 -std=c++17 goalflow_musiccrs/cpp/source_budget_beam.cpp \
  -o goalflow_musiccrs/bin/source_budget_beam
```

运行：

```bash
goalflow_musiccrs/bin/source_budget_beam \
  --data-dir goalflow_musiccrs/experiments/source_candidate_matrix_top800/source_candidate_matrix \
  --target-avg 800 \
  --hard-avg-limit 850 \
  --p95-limit 1200 \
  --per-bucket 4
```

## 搜索逻辑

每个 source 可选：

```text
0, 50, 100, 200, 400, 800
```

对一个 choice：

```text
choice[enriched:seed_current] = 400
choice[metadata_all:legacy_history] = 100
choice[tags:seed_current] = 50
...
```

每个 turn 都计算真实候选池：

```text
U[t] = 所有被选 source 的候选 track_id 合并去重
```

指标：

```text
hit
  有多少 turn 的 gold track 在 U[t] 里面。

coverage
  hit / turn 总数。

avg_union_size
  每个 turn 平均候选池大小。

p95_union_size
  95% 的 turn 候选池不超过这个值。
```

## Beam Search

朴素枚举太大：

```text
6^20
```

所以用 beam search。

为了避免只保留“大 K 大池子”的状态，C++ 搜索器按 `avg_union_size` 分桶：

```text
0-300
300-500
500-700
700-800
800-850
850-900
900-1100
```

每个桶只保留若干个最好的状态。

这样可以同时保住：

```text
小候选池但效率高的方案
接近 800 的主目标方案
略超过 800 但 hit 明显更高的候选方案
```

这不是精确最优算法，而是启发式搜索。

`per-bucket` 越大，保留的中间状态越多，越不容易错过好组合，但内存和运行时间也会上升。

C++ 搜索器现在支持：

```bash
--progress-interval-sec 60
  每 60 秒输出一次当前数字，不输出大集合。

--time-limit-sec 240
  到时间后在完成当前 source 后写出已有最优解并退出。
```

## 当前冒烟结果

小样本命令：

```bash
PYTHONPATH=goalflow_musiccrs .venv/bin/python goalflow_musiccrs/scripts/export_source_candidate_matrix.py \
  --tid source_candidate_matrix_smoke \
  --dev-limit 3 \
  --source-limit 8 \
  --k-values 0,20,50,100 \
  --max-k 100

goalflow_musiccrs/bin/source_budget_beam \
  --data-dir goalflow_musiccrs/experiments/source_candidate_matrix_smoke/source_candidate_matrix \
  --target-avg 80 \
  --hard-avg-limit 100 \
  --p95-limit 150 \
  --per-bucket 3
```

结果：

```text
turns = 24
sources = 8
best hit = 13
coverage = 0.5417
avg_union_size = 79.5
p95_union_size = 120
```

最佳组合：

```text
enriched:seed_current      top50
enriched:legacy_history    top20
enriched:current_goal      top50
tags:seed_current          top20
```

这说明端到端链路已经跑通。

## 中等样本验证

命令：

```bash
PYTHONPATH=goalflow_musiccrs .venv/bin/python goalflow_musiccrs/scripts/export_source_candidate_matrix.py \
  --tid source_candidate_matrix_dev50_s12_k400 \
  --dev-limit 50 \
  --source-limit 12 \
  --k-values 0,50,100,200,400 \
  --max-k 400

goalflow_musiccrs/bin/source_budget_beam \
  --data-dir goalflow_musiccrs/experiments/source_candidate_matrix_dev50_s12_k400/source_candidate_matrix \
  --out-dir goalflow_musiccrs/experiments/source_candidate_matrix_dev50_s12_k400/source_candidate_matrix/beam_search_target800 \
  --target-avg 800 \
  --hard-avg-limit 850 \
  --p95-limit 1200 \
  --per-bucket 4
```

结果：

```text
turns = 400
sources = 12
best hit = 292
coverage = 0.7300
avg_union_size = 795.185
p95_union_size = 996
```

最佳组合：

```text
enriched:seed_current        top400
metadata_all:seed_current    top400
enriched:legacy_history      top100
metadata_all:legacy_history  top100
enriched:current_goal        top100
metadata_all:current_goal    top200
legacy_metadata:seed_current top50
tags:seed_current            top100
enriched:current             top50
album_artist:seed_current    top50
```

这个结果说明两个事情：

1. 低预算时，`seed_current` 是主力。
2. 当真实候选池目标接近 800 时，系统开始需要少量 history/current_goal/tags/album_artist 做补充，但这些 source 的 K 不应该无脑放大。

## 全 Dev 结果

全 dev 导出命令：

```bash
PYTHONPATH=goalflow_musiccrs .venv/bin/python goalflow_musiccrs/scripts/export_source_candidate_matrix.py \
  --tid source_candidate_matrix_full_s20_k800 \
  --source-limit 20 \
  --k-values 0,50,100,200,400,800 \
  --max-k 800
```

导出结果：

```text
turns = 8000
sources = 20
tracks = 47071
candidate_matrix = 488.3 MiB
```

### Budget 曲线：宽松版本

| setting | hit | coverage | avg_union_size | p95_union_size | note |
|---|---:|---:|---:|---:|---|
| target500, hard<=550 | 5109 | 0.6386 | 545.631 | 650 | 500 附近的宽松方案 |
| target800, hard<=850 | 5412 | 0.6765 | 802.611 | 1008 | 800 附近的宽松方案 |
| target800, hard<=800 | 5406 | 0.6758 | 796.931 | 997 | 严格不超过 800 的方案 |
| target1000, hard<=1050 | 5620 | 0.7025 | 1046.880 | 1244 | 1000 附近的宽松方案 |

### Budget 曲线：严格版本

这里的严格版本是：

```text
avg_union_size <= target_avg
```

| target_avg | hit | coverage | avg_union_size | p95_union_size |
|---:|---:|---:|---:|---:|
| 300 | 4634 | 0.5793 | 298.699 | 372 |
| 500 | 5006 | 0.6258 | 499.776 | 604 |
| 700 | 5297 | 0.6621 | 696.278 | 869 |
| 800 | 5406 | 0.6758 | 796.931 | 997 |
| 900 | 5470 | 0.6837 | 895.785 | 1139 |
| 1000 | 5546 | 0.6933 | 999.402 | 1205 |

边际收益：

```text
300 -> 500: +372 hit
500 -> 700: +291 hit
700 -> 800: +109 hit
800 -> 900: +64 hit
900 -> 1000: +76 hit
```

所以 800 不是魔法数字。它只是一个比较合理的折中点：

```text
500 以下还丢太多 gold。
800 之后继续加候选还能救，但边际收益明显变低。
```

`target800 strict` 用 `per-bucket=8` 复跑后仍然是：

```text
hit = 5406
coverage = 0.67575
avg_union_size = 796.931
```

说明当前这个解不是 `per-bucket=4` 太窄导致的明显坏解。

严格 `avg_union_size <= 800` 时的最佳组合：

```text
enriched:seed_current        top400
metadata_all:seed_current    top400
enriched:legacy_history      top200
enriched:current_goal        top100
metadata_all:current_goal    top50
tags:seed_current            top400
enriched:current             top50
tags:current_goal            top50
```

宽松 `avg_union_size ≈ 800` 时的最佳组合：

```text
enriched:seed_current        top400
metadata_all:seed_current    top400
enriched:legacy_history      top200
metadata_all:legacy_history  top50
enriched:current_goal        top100
tags:seed_current            top400
enriched:current             top50
tags:current_goal            top50
```

### Source 重合度观察

`source_pair_hit_overlap.csv` 里可以看到一些长期高度重复的 source pair。

例如在 `K=400`：

```text
legacy_metadata:seed_current 和 album_artist:seed_current
  overlap_coef = 0.973

legacy_metadata:legacy_history 和 album_artist:legacy_history
  overlap_coef = 0.968

metadata_all:seed_current 和 tags:seed_current
  overlap_coef = 0.966
```

`overlap_coef` 接近 1 的意思是：

```text
较小那个 source 命中的 turn，几乎都已经被另一个 source 命中了。
```

所以这类 source 不适合同时大 K。它们可以保留小 K 当补充，但大预算应该优先给更有独立新增 hit 的 source。

### 当前结论

粗排阶段不应该再用统一 K，也不应该用简单 RRF 当最终粗排答案。

目前更合理的 coarse recall 配置方向是：

```text
主力大 K:
  enriched:seed_current
  metadata_all:seed_current
  enriched:legacy_history
  tags:seed_current

小 K 补充:
  enriched:current_goal
  metadata_all:current_goal
  enriched:current
  tags:current_goal

不宜同时大 K:
  legacy_metadata:seed_current
  album_artist:seed_current
  legacy_metadata:legacy_history
  album_artist:legacy_history
  title_artist:legacy_history
```

下一步应该把严格 `avg<=800` 的 source/K 组合接入实际预测流程，生成候选池给 reranker，而不是再用旧的统一 `retrieval_top_k`。

## Miss 分析

分析脚本：

```bash
PYTHONPATH=goalflow_musiccrs .venv/bin/python goalflow_musiccrs/scripts/analyze_budget_choice_misses.py \
  --matrix-dir goalflow_musiccrs/experiments/source_candidate_matrix_full_s20_k800/source_candidate_matrix \
  --out-dir goalflow_musiccrs/experiments/source_candidate_matrix_full_s20_k800/source_candidate_matrix/miss_analysis_strict_budgets \
  --choice budget300:.../beam_search_target300_strict/best_choice.tsv \
  --choice budget500:.../beam_search_target500_strict/best_choice.tsv \
  --choice budget700:.../beam_search_target700_strict/best_choice.tsv \
  --choice budget800:.../beam_search_target800_strict/best_choice.tsv \
  --choice budget900:.../beam_search_target900_strict/best_choice.tsv \
  --choice budget1000:.../beam_search_target1000_strict/best_choice.tsv
```

输出目录：

```text
goalflow_musiccrs/experiments/source_candidate_matrix_full_s20_k800/source_candidate_matrix/miss_analysis_strict_budgets/
```

关键文件：

```text
choice_eval_summary.csv
  每个预算的 hit、coverage、avg size、p95 size。

miss_recovery_summary.csv
  漏掉的 turn 是否仍然能在某个 source top800 里找到。

miss_feature_summary.csv
  漏掉样本的 turn/category/specificity/tag/artist/year/popularity 分布。

misses_budget800.csv
  budget800 每个漏掉样本的明细。
```

`budget800` 的漏召回类型：

| recovery_type | miss_count | miss_share |
|---|---:|---:|
| not_in_any_exported_source_top800 | 1552 | 0.5983 |
| in_unselected_source_top800 | 544 | 0.2097 |
| in_selected_source_but_rank_above_selected_k | 498 | 0.1920 |

解释：

```text
not_in_any_exported_source_top800
  当前这 20 个 BM25 source 就算各自取到 top800，也找不到 gold。
  这部分不能靠调 K 解决，需要新 source，例如 embedding、entity/parser、歌词/属性/CF 等。

in_unselected_source_top800
  某个没选或没给预算的 source 其实能找到。
  这部分可以研究 source 组合是否还不够好。

in_selected_source_but_rank_above_selected_k
  已选 source 能找到，但 rank 太靠后。
  这部分可以通过增大某些 K 或改 rerank 前候选池预算解决。
```

`budget800` 的 miss 特征：

```text
specificity:
  LL 821
  LH 806
  HL 778
  HH 189

gold_popularity_bucket:
  25-49 943
  50-74 801
  0     333
  75+   224
  10-24 216
  1-9    77

gold_year_bucket:
  2010s 1341
  2000s 559
  1990s 342
  1980s 150
  1970s 140
```

这说明漏掉的不是单纯“冷门歌”。

真正的大头是：

```text
LL/LH/HL 这类不够具体或半具体的目标。
```

这类 query 往往不是“歌名/艺人精确匹配”能解决的，后续更像要靠：

```text
embedding semantic recall
user/CF recall
positive seed similarity
训练对话 augmentation 更强的 query-to-track 映射
```
