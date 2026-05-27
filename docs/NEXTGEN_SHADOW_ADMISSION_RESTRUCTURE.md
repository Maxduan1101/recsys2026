# Nextgen Shadow / Admission / Rerank 重构记录

## 当前判断

现在系统的主要问题不是“还能不能多召回一些歌”，而是“新增进来的歌里，有多少能被 reranker 排到前面”。所以后续指标不能只看 `gold_in_pool`，还要看：

- `extra_gold_over_base`: 比基础池多找到了多少正确歌。
- `extra_gold_rankable@50/@100`: 新增正确歌里，有多少被当前模型排到前 50/100。
- `rankable_density`: 每增加一批候选，换来多少可排序的新增正确歌。
- `avg/p95 group_size`: 每个 turn 的候选池大小，控制 reranker 成本和噪声。

这对应新的三层结构：

1. `shadow mega-pool`: 大而全，只用来探索上限和分析 source 价值。
2. `admission`: 按 family/source 证据、候选质量和预算压缩候选池。
3. `LTR rerank`: LightGBM LambdaRank 只负责最终排序，不再背锅解决召回缺失。

## 已落地的工程入口

### 1. Blind 候选池构建

`scripts/build_nextgen_candidate_pool.py` 已支持 `--mode blind`，可以用 Blind-A 的 80 个 turn 构建 nextgen pool。

已跑通产物：

```text
experiments/nextgen_candidate_pool_v2_seed_audio_attr_cap1400_blindA/
```

关键规模：

```text
rows = 101,953
groups = 80
avg group size = 1274.41
p95 group size = 1400
```

### 2. Blind rerank 特征缓存

新增：

```text
scripts/build_rerank_v2_feature_cache.py
```

用途：把任意 dev/blind candidate pool 转成 rerank_v2 特征缓存，并可直接加 v3 semantic interaction features。

已跑通产物：

```text
experiments/rerank_v2_nextgen_v3_cap1400_blindA_apply/features_nextgen_blind_v3.pkl
```

关键规模：

```text
rows = 101,953
groups = 80
feature columns = 436
```

### 3. Train -> Apply 提交侧脚本

新增：

```text
scripts/train_apply_rerank_v2.py
```

用途：从 labeled dev feature cache 训练 LightGBM Ranker，然后给 blind feature cache 打分，输出：

```text
prediction.json
submission.zip
ranked_apply_top100.jsonl
train_apply_summary.json
```

它支持：

- 单模型训练。
- 多模型 rank blend。
- `all/source/independent` 三种 feature set。
- 内存中追加 v3 semantic features。

### 4. Shadow / Admission 分析脚本

新增：

```text
scripts/analyze_shadow_admission_pool.py
```

用途：把 candidate pool 拆成 source/family 级别，输出：

```text
pool_shadow_summary.csv
source_family_shadow_summary.csv
extra_gold_rankability_diagnostics.csv
```

当传入 scored pkl 时，它会计算：

- `gold_hit_rankable_le20/le50/le100`
- `extra_gold_rankable_le50/le100`
- `rankable_density_le50/le100`

这正是报告里要求的 admission 主指标。

已对 Blind-A nextgen pool 跑通基础分析：

```text
experiments/shadow_admission_blindA_nextgen_cap1400/
```

Blind 没有 gold label，所以这里主要验证 pool size 和 family/source 分布；真正的 rankable 指标需要在 dev 上带 label 跑。

## 下一步还缺什么

### A. dev nextgen feature cache 需要重建

清理 260GB 垃圾缓存时，删除了 nextgen v2 的底层大 pkl，几个调参目录只剩断开的 symlink。代码和轻量实验记录还在，但如果要训练最终 blind 模型，需要重建 dev nextgen feature cache。

建议只重建当前主线配置，不再恢复所有旧垃圾实验：

```bash
PYTHONPATH=goalflow_musiccrs:goalflow_musiccrs/scripts .venv/bin/python \
  goalflow_musiccrs/scripts/build_nextgen_candidate_pool.py \
  --mode dev \
  --out-dir goalflow_musiccrs/experiments/nextgen_candidate_pool_v2_seed_audio_attr_cap1400_dev_rebuild \
  --old300-pkl goalflow_musiccrs/cache/ltr_candidate_frames/dev_7d9af67ef612.pkl \
  --matrix-dir goalflow_musiccrs/experiments/source_candidate_matrix_full_s20_k800/source_candidate_matrix \
  --beam800-choice goalflow_musiccrs/experiments/source_candidate_matrix_full_s20_k800/source_candidate_matrix/beam_search_target800_strict/best_choice.tsv \
  --seed-channels track_cf,attributes,audio \
  --seed-channel-ks track_cf:180,attributes:180,audio:180 \
  --cf-seed-k 180 \
  --tail-max-add 180 \
  --max-pool-size 1400 \
  --write-pkl
```

然后构建 dev v3 feature cache：

```bash
PYTHONPATH=goalflow_musiccrs:goalflow_musiccrs/scripts .venv/bin/python \
  goalflow_musiccrs/scripts/build_rerank_v2_feature_cache.py \
  --mode dev \
  --pool-name nextgen \
  --pool-pkl goalflow_musiccrs/experiments/nextgen_candidate_pool_v2_seed_audio_attr_cap1400_dev_rebuild/nextgen_pool.pkl \
  --output-pkl goalflow_musiccrs/experiments/rerank_v2_nextgen_v3_cap1400_dev_rebuild/features_nextgen_dev_v3.pkl \
  --meta-json goalflow_musiccrs/experiments/rerank_v2_nextgen_v3_cap1400_dev_rebuild/features_nextgen_dev_v3.json \
  --augment-v3
```

### B. 最终 blind 训练应用

dev feature cache 重建后，训练并生成 blind zip：

```bash
PYTHONPATH=goalflow_musiccrs:goalflow_musiccrs/scripts .venv/bin/python \
  goalflow_musiccrs/scripts/train_apply_rerank_v2.py \
  --train-feature-pkl goalflow_musiccrs/experiments/rerank_v2_nextgen_v3_cap1400_dev_rebuild/features_nextgen_dev_v3.pkl \
  --apply-feature-pkl goalflow_musiccrs/experiments/rerank_v2_nextgen_v3_cap1400_blindA_apply/features_nextgen_blind_v3.pkl \
  --apply-mode blind \
  --out-dir goalflow_musiccrs/experiments/blindA_nextgen_v3_submit \
  --feature-set all \
  --rank-blend \
  --model-specs 'v3:1.0:objective=lambdarank,n_estimators=800,learning_rate=0.03,num_leaves=31,min_child_samples=100,reg_lambda=5,reg_alpha=0,subsample=0.8,colsample_bytree=0.8,lambdarank_truncation_level=300;v2trunc400:0.75:objective=lambdarank,n_estimators=800,learning_rate=0.03,num_leaves=31,min_child_samples=100,reg_lambda=5,reg_alpha=0,subsample=0.8,colsample_bytree=0.8,lambdarank_truncation_level=400;leaves63:0.5:objective=lambdarank,n_estimators=800,learning_rate=0.03,num_leaves=63,min_child_samples=100,reg_lambda=5,reg_alpha=0,subsample=0.8,colsample_bytree=0.8,lambdarank_truncation_level=300'
```

## 风险

- 重建 dev nextgen feature cache 会重新占用较大磁盘空间，但只需要保留一份主线 dev cache。
- Blind-A 的 rerank 特征已经跑通；真正耗时的是 dev 8000 个 turn 的 feature rebuild 和 LightGBM 训练。
- 当前还没有真正 query-to-Qwen / query-to-CLAP / query-to-SigLIP 的新 query encoder source；现有 nextgen 主要是 BM25 tail rescue + seed embedding neighbor。
