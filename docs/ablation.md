# Ablation 实验手册

近期大量提交（多源候选、Router、STHGCN T/L、因果计数、目标协议等）已合入 `main`，但对 Acc@1 各自贡献尚不清楚。本文给出**固定协议 + 因子表 + 推荐顺序 + 可复制命令 + 结果登记表**，方便系统做对照实验。

配套说明：[`candidates.md`](candidates.md) · [`router-modes.md`](router-modes.md) · [`sthgcn-tl-features.md`](sthgcn-tl-features.md) · [`causal-counts.md`](causal-counts.md) · [`candidate-context.md`](candidate-context.md) · [`results.md`](results.md)。

---

## 1. 目的

一次只改一个（或一组明确声明的）因子，其余保持默认，记录 **Acc@1**（主指标）以及 NDCG@1 / MRR，回答「哪项改动真正有用」。

---

## 2. 固定协议（所有对照共用）

| 项 | 约定 |
| --- | --- |
| **主指标** | `all_targets` 下的 **Acc@1**；同时记 NDCG@1、MRR |
| **结果路径** | `results/<run_name>/eval_test/metrics.json`（及同目录 `report.txt`） |
| **checkpoint** | `runs/...`；metrics **只**看 `results/` |
| **SURD 先验** | **train-all、无 fold**：在整份训练集上估计 **逐 POI** SURD；**train / val / test 共用同一张** `final.surd.pt`；训练不再查 fold 特异表 |
| **同一 prepare 指纹** | 能复用同一 cache 的实验（Router / causal / transition / 训练负采样）**必须共用同一 `--cache`** |
| **须重 prepare** | 候选构造、目标协议、T/L 特征 schema、或 **SURD 协议/指纹**（已去掉 `--folds` / `--warmup`）任一变化 → 重新 `prepare`（对照时用**独立** `--out` 保留多套 cache；同路径默认覆盖，见下表「依赖 prepare?」） |
| **默认基线 cache（Tokyo）** | `runs/tokyo/cache`：默认多源 `p90` + popular@max + user-frequent + random50；`train/eval-targets=all`；`feature_method=sthgcn_tl_features_v1`；**仅** `final` SURD 表 |
| **默认训练 knobs（除非该行明确改）** | `--router-mode hard_prior`；因果计数 **on**；`--transition-attn bias`；`--negatives 63`；`--max-train-candidates -1`；`--device cuda` |
| **对比公平性** | 同一 cache、同一 seed（默认 42）、独立 `--out` / `results/<run>`，避免结果目录互相覆盖 |

读 Acc@1：

```bash
python -c "import json; m=json.load(open('results/tokyo/abl_router_hard/eval_test/metrics.json')); print(m['all_targets']['Acc@1'], m['all_targets']['NDCG@1'], m['all_targets'].get('MRR'))"
```

**自动登记：** `train` 结束（写完最终 test）以及独立 `evaluate` 结束时，会**自动 append** 一行到：

- `results/ablation_log.md`
- `results/ablation_log.csv`

（文件不存在则按模板表头创建。）列与下方模板一致：`date` / `run_name` / `cache` / `key flags` / `Acc@1` / `NDCG@1` / `MRR` / `notes`（`notes` 常含 best-val Acc@1 等）。无需再手工 `cp` 才能开始记；模板仍可作空表参考：

```bash
# 可选：从模板重置本地日志（会丢掉已自动写入的行）
cp docs/ablation_log.template.md results/ablation_log.md
cp docs/ablation_log.template.csv results/ablation_log.csv
```

**总表（推荐）：** 扫当前 `results/**/eval_test/metrics.json`（并合并已有 `ablation_log.*`），写出含 **known_only** 与 **run-name ↔ flags 校验** 的主表：

```bash
python -m tsmoe ablation-summary --results results
# → results/ablation_summary.md
# → results/ablation_summary.csv
# 可选：--split validation；--strict（有 name mismatch 时 exit 1）
```

详见 §6.1（`known_only` 定义）与 §6.2（命名校验）。

---

## 3. 因子 / 开关总表

| 因子 | 改什么 | 档位 / on·off | CLI | 依赖 prepare? | 建议 `run_name`（`--out` 末段） |
| --- | --- | --- | --- | --- | --- |
| **候选半径** | 局部邻域半径 | `p70` / `p80` / `p90`（默认）/ `mean` / `max` / `full` / `random` | `prepare --candidate-radius …` | **是** | `abl_cand_p90` / `abl_cand_p80` / `abl_cand_full` / `abl_cand_random` … |
| **多源：热门** | 大半径热门源 | on（默认 top-k=100）/ off（`--popular-top-k 0`）；半径 `max`/`p90`/…/`multiplier` | `--popular-top-k` `--popular-radius` `--popular-by` | **是** | `abl_cand_no_pop` / `abl_cand_pop_p80` |
| **多源：用户常去** | 用户训练常去 | on（默认 top-k=50）/ off（`--user-frequent-top-k 0`） | `--user-frequent-top-k` `--user-frequent-min-visits` | **是** | `abl_cand_no_uf` |
| **随机负例** | 池内额外均匀负例（或 `random` 模式下的整池抽样数） | `50`（默认）/ `0`（关；**不可**与 `random` 同用）/ 其它正整数 | `--random-negatives` `--random-negatives-seed` | **是** | `abl_cand_rnd0` / `abl_cand_rnd50` |
| **仅邻域+随机** | 关掉热门+用户常去 | 组合档 | `--popular-top-k 0 --user-frequent-top-k 0` | **是** | `abl_cand_nbhd` |
| **随机池 vs 多源基线** | 仅均匀随机候选 vs 默认多源 | 对照：`p90` 多源（默认）/ `random` | 基线复用默认 cache；`random` 须 `--candidate-radius random`（`--random-negatives >= 1`） | **是**（`random`） | `abl_cand_baseline` / `abl_cand_random` |
| **Router 模式** | 专家路由 π | `hard_prior`（默认）/ `surd_mlp` / `hybrid` | `train --router-mode …`；hybrid 可加 `--lambda-prior` `--hybrid-lambda` | 否 | `abl_router_hard` / `abl_router_surd_mlp` / `abl_router_hybrid` |
| **`--no-surd`** | 关掉 SURD 引导 | off（默认）/ on | `train --no-surd` | 否 | `abl_nosurd` |
| **SURD 表协议** | 先验如何估计 / 谁查哪张表 | **仅 train-all 共用**（协议已固定；无 fold 档可切） | `prepare`（**无** `--folds`/`--warmup`）只写 `final.surd.pt`；`train`/`evaluate`/`explain`/`inspect --table final` 皆查 `final` | **是**（旧 fold 缓存须重 prepare） | 记为「当前默认」；旧 fold 训练先验 **不再支持** |
| **STHGCN T/L** | T=48 / L=18 相对 Φ(Δt)/Φ(Δs) | **当前唯一 schema**（`sthgcn_tl_features_v1`）；旧 `basic_fixed_features_v1` **已移除、无 CLI 开关** | （无开关；换 schema 须改代码并 **重 prepare**） | **是**（若回退旧特征） | 记为「当前默认」；不可与旧 basic 同 cache 对比 |
| **因果计数整模块** | 在线访问/转移表 + 分箱 + transition attn | **on（默认）** / off | `--causal-counts` / `--no-causal-counts` | 否 | `abl_causal_on` / `abl_causal_off` |
| **transition-attn** | 转移如何进 P/C 交叉注意力 | `bias`（默认）/ `gate` / `rel_key` | `--transition-attn`；`--transition-lambda`（bias） | 否（须因果 **on**） | `abl_tr_bias` / `abl_tr_gate` / `abl_tr_rel_key` |
| **visit-bins** | T/L 用户累计访问等频分箱 | 默认 `10` | `--visit-bins` | 否（须因果 **on**） | `abl_visit_bins_10` |
| **训练目标协议** | 样本取「全部前缀」还是「仅末站」 | `all`（默认）/ `last` | `prepare --train-targets` | **是** | `abl_tgt_all` / `abl_tgt_last` / `abl_tgt_train_all_eval_last` |
| **评估目标协议** | val/test 样本协议 | `all`（默认）/ `last` | `prepare --eval-targets` | **是** | （同上） |
| **未知 user/POI 过滤** | STHGCN 式剔除冷启动用户与目录外目标 | **`off`（默认）** / `on` | `prepare --filter-unknown` | **是** | `abl_filter_unknown` |
| **训练负采样** | 从构造池抽多少负例 | `63`（默认）/ `-1`（池内其余全要） | `train --negatives` | 否 | `abl_neg_63` / `abl_neg_all` |
| **训练候选封顶** | 含正例的训练候选上限 | `-1`（默认不封）/ 如 `256` | `train --max-train-candidates` | 否 | `abl_maxcand_256` |
| **候选上下文 A** | 候选 cat emb + Δgeo → query | **`off`（默认）** / `on` | `--candidate-context` | 否 | `abl_ctx_off` / `abl_ctx_cat_geo` |
| **候选时段 A-time** | last_slot / causal_slot | **`off`（默认）** / `last_slot` / `causal_slot` | `--candidate-context-time` | 否† | `abl_ctx_cat_geo_time` |
| **会话摘要 B** | 窗口 cat hist / 长度桶 / 活动中心 | **`off`（默认）** / `on` | `--session-summary` | 否 | `abl_sess_on` |
| **用户偏好 C** | 因果 user–cat / user–hour → base | **`off`（默认）** / `on` | `--user-pref`（须因果 on） | 否† | `abl_upref_on` |
| **用户 ID 嵌入** | train `user_id` Embedding → base | **`off`（默认）** / `on` | `--user-id-emb` | 否 | `abl_user_id` |
| **转移 PMI D** | 转移 attn 用 PMI 替代 raw count | **`off`（默认）** / `on` | `--transition-pmi`（须因果 on） | 否† | `abl_tr_pmi` |
| **专家内交叉注意力 E** | E0 off / E1 hybrid / E2 full | **`off`（默认）** / `hybrid` / `full` | `--expert-cross-attn` | 否 | `abl_e1_hybrid` / `abl_e2_full` |
| **E3 稀疏** | 仅 top-k 专家跑 expert attn | **`on`（E≠off 时）** / `off` | `--expert-cross-attn-sparse` | 否 | `abl_e1_e3` |
| **转移分数偏置** | `λ·log1p(last→cand POI)` 加到候选分数 | **`0`（默认）** / `>0` | `--transition-score-bias`（须因果 on） | 否† | `abl_tsb_*` |
| **类别→POI 分数偏置** | `λ·log1p(last_cat→cand_poi)` 超图先验 | **`0`（默认）** / `>0` | `--category-score-bias`（须因果 on） | 否† | `abl_csb_*` |
| **热门分数偏置** | `λ·log1p(global visits)` | **`0`（默认）** / `>0` | `--pop-score-bias`（须因果 on） | 否† | `abl_psb_*` |
| **距离分数偏置** | `−λ·log1p(haversine km)` 近者优先 | **`0`（默认）** / `>0` | `--geo-score-bias`（须 A on） | 否 | `abl_gsb_*` |
| **SURD/SMI 偏置门控** | 用 SURD / Specific MI 控制 tsb / csb / attn / 直接分数偏置 | **`off`（默认）** | `--surd-bias-gate` + `--surd-score-bias` / `--tsb-gate-mode` / `--surd-attn-gate` | 否† | `abl_surd_gate_*` |
| **λ_bpr / margin** | hard-neg BPR + logit margin | **`0`（默认）** | `--lambda-bpr` / `--bpr-margin` | 否 | `abl_bpr_*` |

† `causal_slot` / `user-pref` / `transition-pmi` 扩展在线计数表（**无需** re-prepare）。新特征 **不进** `hard_prior`。文本 emb **未实现**（见 [`candidate-context.md`](candidate-context.md)）。

> **#20 状态：** 因果在线计数 + `--transition-attn` 已合入 `main`（[#20](https://github.com/ShiyinTan/TS_MOE/pull/20)）。上表因果相关 flag 均为当前主干能力，**无需**再 checkout 旧分支。

### 3.1 Overlay 对照表（叠在默认 train 上）

默认基线（不改 Acc@1 路径）：

```bash
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_ctx_off --device cuda
```

| setting | add flags（叠在默认 train/eval 上） | 建议 `run_name` |
| --- | --- | --- |
| A cat+geo | `--candidate-context on` | `abl_ctx_cat_geo` |
| A + time (`last_slot`) | `--candidate-context on --candidate-context-time last_slot` | `abl_ctx_cat_geo_time` |
| A + time (`causal_slot`) | `--candidate-context on --candidate-context-time causal_slot` | `abl_ctx_cat_geo_causal_slot` |
| B | `--session-summary on` | `abl_sess_on` |
| C | `--user-pref on` | `abl_upref_on` |
| D PMI | `--transition-pmi on` | `abl_tr_pmi` |
| E1 hybrid (+E3 sparse default) | `--expert-cross-attn hybrid` | `abl_e1_hybrid` |
| E1 + dense expert attn | `--expert-cross-attn hybrid --expert-cross-attn-sparse off` | `abl_e1_dense` |
| E2 full | `--expert-cross-attn full` | `abl_e2_full` |
| recommended: A+B | `--candidate-context on --session-summary on` | `abl_ctx_sess` |
| recommended: A+B+C | `--candidate-context on --session-summary on --user-pref on` | `abl_ctx_sess_upref` |
| recommended: A+E1 | `--candidate-context on --expert-cross-attn hybrid` | `abl_ctx_e1` |
| SURD gate `unique_p` residual + weak ssb | `--surd-bias-gate unique_p --tsb-gate-mode residual --surd-score-bias 0.26 --transition-score-bias 2.0` | `abl_surd_gate_up_res` |
| cat→poi score bias | `--category-score-bias 0.5` | `abl_csb_05` |
| pop score bias | `--pop-score-bias 0.3` | `abl_psb_03` |
| geo score bias (needs A) | `--candidate-context on --geo-score-bias 0.5` | `abl_gsb_05` |
| **LOCKED WIN (TM-34):** A+time + BPR + SURD residual | see §3.2 | `hc_sg_Atime_bpr029_ssb026` |

细节见：候选 [`candidates.md`](candidates.md)；Router [`router-modes.md`](router-modes.md)；STHGCN [`sthgcn-tl-features.md`](sthgcn-tl-features.md)；因果 [`causal-counts.md`](causal-counts.md)；overlays [`candidate-context.md`](candidate-context.md)。

### 3.2 LOCKED WIN — Tokyo vs STHGCN (known_only)

**Recipe name:** `hc_sg_Atime_bpr029_ssb026` · **cache:** `runs/tokyo/cache_abl_tgt_last` · **ep1** · `--checkpoint-metric acc1`

| | Acc@1 / R@1 | R@5 | R@10 | R@20 |
| --- | ---: | ---: | ---: | ---: |
| STHGCN | 0.255788 | 0.441345 | 0.511687 | 0.574311 |
| **TS_MOE** | **0.260274** | **0.441517** | **0.519494** | **0.586934** |

```bash
python -m tsmoe train \
  --cache runs/tokyo/cache_abl_tgt_last \
  --out runs/tokyo/hc_sg_Atime_bpr029_ssb026 \
  --seed 42 --device cuda \
  --router-mode hard_prior --dim 256 --heads 8 \
  --lambda-bpr 0.29 \
  --transition-score-bias 2.0 \
  --surd-bias-gate unique_p --tsb-gate-mode residual --surd-score-bias 0.26 \
  --candidate-context on --candidate-context-time last_slot \
  --epochs 1 --checkpoint-metric acc1
# → results/tokyo/hc_sg_Atime_bpr029_ssb026/eval_test/metrics.json  (read known_only)
```

Judge **`known_only`** Acc@1 / Recall@k (not `all_targets`). Flags default-off; see [`causal-counts.md`](causal-counts.md).

### 3.3 Acc climb peak — deltas vs WIN (not a full flag dump)

**Peak:** `on_rev_soft10r` · known_only Acc@1 **0.267650** (+0.007376 vs WIN).

| | |
| --- | --- |
| **Base** | LOCKED WIN §3.2 (`hc_sg_Atime_bpr029_ssb026`) |
| **Added** | `--transition-attn gate` · `--geo-score-bias 0.15` · `--revisit-score-bias 0.10` · `--soft-target-beta 0.10 --soft-target-prior revisit` |

Full Keep/Discard delta table: [`climb-delta-from-win.md`](climb-delta-from-win.md) (also mirrored under `results/tokyo/` on the GPU box).

---

## 4. 推荐顺序（便宜 → 贵）

同一 cache 上先扫 **训练期开关**（无需 re-prepare），再扫 **须重 prepare** 的构造，最后扫最贵的因果 / transition 细档（也可与 Router 同 cache 交叉，但建议一次一因子）。

| 步 | 做什么 | 是否 re-prepare | 说明 |
| ---: | --- | --- | --- |
| **0** | 准备基线 Tokyo cache | 一次 | `runs/tokyo/cache`（train-all `final` SURD；或明确记下指纹的等价 cache） |
| **1** | Router：`hard_prior` / `surd_mlp` / `hybrid` | 否 | 最便宜；同一 cache |
| **2** | 训练负采样：`negatives` / `max-train-candidates` | 否 | 同一 cache |
| **3** | 因果整模块：on vs `--no-causal-counts` | 否 | 同一 cache |
| **4** | transition：`bias` / `gate` / `rel_key`（因果 on） | 否 | 同一 cache |
| **4b** | Overlay A/B/C/D/E（§3.1；默认均 off） | 否 | 同一 cache；一次一因子或声明组合 |
| **5** | 候选变体：半径 / 关源 / 随机负例 / full / **random vs 基线** | **是** | 每种独立 cache + model；§5 含 `abl_cand_baseline`↔`abl_cand_random` |
| **6** | 目标协议：`all`/`last` 组合 | **是** | 独立 cache |
| **6b** | `--filter-unknown on`（STHGCN 式剔除未知 user/POI） | **是** | 独立 cache；**不是** `known_only`（见 §6.1） |
| — | STHGCN T/L | — | **无开关**；旧 basic 特征已去掉，不能做 CLI 消融 |

---

## 5. 可复制命令（Tokyo）

均在仓库根目录、`conda activate tsmoe`。按 §4 顺序执行；每组用独立 `--out`，metrics 落在 `results/tokyo/<run>/eval_test/`。

**SURD（与 README / [#24](https://github.com/ShiyinTan/TS_MOE/pull/24) 一致）：** `prepare` **没有** `--folds` / `--warmup`；每次 prepare 只写 **`final.surd.pt` / `final.quantizer.pt`**（整份训练集 per-POI）；**train / val / test / evaluate / explain** 都查同一张 `final`。合入 #24 后旧 fold cache **须重跑下面的 prepare**（同路径默认覆盖；对照用独立 `--out`）。

### 0) 基线 prepare（#24 后先重跑一次；之后 1–4 步复用同一 cache）

```bash
conda activate tsmoe
# 需已有 data/tokyo/*.csv（见 README §3.2）

# 构建样本 + 训练集全量 SURD（train/val/test 共用）+ 多源候选
# （勿再传 --folds / --warmup；CLI 已移除）
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache \
  --candidate-radius p90 --popular-radius max --popular-top-k 100 \
  --popular-by visit --user-frequent-top-k 50 --user-frequent-min-visits 2 \
  --random-negatives 50 --random-negatives-seed 0 \
  --train-targets all --eval-targets all
# 日志：`final: N reference targets (train-all shared SURD); scored=0`
# → 只写 final.surd.pt / final.quantizer.pt；无 fold_*.surd.pt
# 旧含 fold_* 的 cache：指纹已变，同路径默认覆盖后即可 train。
```

可选 Demo 烟雾（同一 no-fold 逻辑，小模型）：

```bash
python examples/make_demo.py --data-root data
python -m tsmoe prepare --data-root data --city demo --out runs/demo/cache
# → 同样只写 final.surd.pt（train/val/test 共用）；无 --folds / --warmup
```

### 1) Router 模式（同一 `runs/tokyo/cache`；先验均查 `final`）

```bash
# hard_prior（默认）；train/evaluate 查 cache 内 final.surd.pt（非 fold 表）
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_router_hard \
  --router-mode hard_prior --device cuda
python -m tsmoe evaluate --cache runs/tokyo/cache \
  --checkpoint runs/tokyo/abl_router_hard/best.pt --split test --device cuda
# → results/tokyo/abl_router_hard/eval_test/

# surd_mlp
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_router_surd_mlp \
  --router-mode surd_mlp --device cuda
python -m tsmoe evaluate --cache runs/tokyo/cache \
  --checkpoint runs/tokyo/abl_router_surd_mlp/best.pt --split test --device cuda

# hybrid
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_router_hybrid \
  --router-mode hybrid --lambda-prior 1 --hybrid-lambda 0.1 --device cuda
python -m tsmoe evaluate --cache runs/tokyo/cache \
  --checkpoint runs/tokyo/abl_router_hybrid/best.pt --split test --device cuda
```

Demo 可选：

```bash
python -m tsmoe train --cache runs/demo/cache --out runs/demo/abl_router_hard \
  --router-mode hard_prior --epochs 1 --dim 16 --heads 2 --layers 1
python -m tsmoe train --cache runs/demo/cache --out runs/demo/abl_router_surd_mlp \
  --router-mode surd_mlp --epochs 1 --dim 16 --heads 2 --layers 1
python -m tsmoe train --cache runs/demo/cache --out runs/demo/abl_router_hybrid \
  --router-mode hybrid --lambda-prior 1 --hybrid-lambda 0.1 \
  --epochs 1 --dim 16 --heads 2 --layers 1
```

### 2) 训练负采样 / 候选封顶（同一 cache）

```bash
# 默认 63 负例
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_neg_63 \
  --negatives 63 --max-train-candidates -1 --device cuda

# 池内其余全作负例 + 封顶 256
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_neg_all_max256 \
  --negatives -1 --max-train-candidates 256 --device cuda
```

### 3) 因果计数 on / off（同一 cache；已在 main）

```bash
# 默认 on + bias
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_causal_on \
  --causal-counts --transition-attn bias --device cuda

# 整模块关闭（恢复 shuffle）
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_causal_off \
  --no-causal-counts --device cuda
```

### 4) transition-attn 三档（因果 on）

```bash
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_tr_bias \
  --transition-attn bias --transition-lambda 1.0 --device cuda
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_tr_gate \
  --transition-attn gate --device cuda
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_tr_rel_key \
  --transition-attn rel_key --device cuda
```

### 4b) Overlay A/B/C/D/E（同一 `runs/tokyo/cache`；**无需** re-prepare）

全部叠在默认 knobs 上（`hard_prior` + 因果 on + `bias`）。新特征 **不进** `hard_prior`。完整表见 §3.1。

```bash
# 基线（全部 overlay off）
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_ctx_off --device cuda

# A：候选 cat + Δgeo
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_ctx_cat_geo \
  --candidate-context on --device cuda

# A + last_slot
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_ctx_cat_geo_time \
  --candidate-context on --candidate-context-time last_slot --device cuda

# A + causal_slot（须因果 on；扩展在线表，不 re-prepare）
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_ctx_cat_geo_causal_slot \
  --candidate-context on --candidate-context-time causal_slot --device cuda

# B：会话摘要
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_sess_on \
  --session-summary on --device cuda

# C：用户偏好（须因果 on）
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_upref_on \
  --user-pref on --device cuda

# D：转移 PMI
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_tr_pmi \
  --transition-pmi on --device cuda

# E1 hybrid（E3 sparse 默认 on）
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_e1_hybrid \
  --expert-cross-attn hybrid --device cuda

# E1 + E3 显式 / E1 dense
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_e1_e3 \
  --expert-cross-attn hybrid --expert-cross-attn-sparse on --device cuda
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_e1_dense \
  --expert-cross-attn hybrid --expert-cross-attn-sparse off --device cuda

# E2 full（更贵）
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_e2_full \
  --expert-cross-attn full --device cuda

# 推荐组合
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_ctx_sess \
  --candidate-context on --session-summary on --device cuda
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_ctx_sess_upref \
  --candidate-context on --session-summary on --user-pref on --device cuda
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_ctx_e1 \
  --candidate-context on --expert-cross-attn hybrid --device cuda
```

Demo 烟雾：

```bash
python -m tsmoe train --cache runs/demo/cache --out runs/demo/abl_ctx_cat_geo \
  --candidate-context on --epochs 1 --dim 16 --heads 2 --layers 1
python -m tsmoe train --cache runs/demo/cache --out runs/demo/abl_e1_hybrid \
  --expert-cross-attn hybrid --epochs 1 --dim 16 --heads 2 --layers 1
```

### 5) 候选变体（每种 **重新 prepare**；独立 `--out` 以免覆盖）

每条 prepare 同样**无** `--folds` / `--warmup`，只产出 train-all `final` SURD（与基线协议相同；变的是候选构造）。

#### 5a) 随机采样 candidates vs 当前多源基线（推荐先跑）

同一 seed / 默认 train knobs / Acc@1；**只改候选构造**。基线复用 §0 的 `runs/tokyo/cache`（无需再 prepare）；`random` 须独立 cache。

| run_name | 候选构造 | cache | 差异 |
| --- | --- | --- | --- |
| `abl_cand_baseline` | 默认多源：邻域 `p90` ∪ popular@max ∪ user-frequent ∪ random50 | `runs/tokyo/cache` | 当前默认 |
| `abl_cand_random` | **仅**均匀随机：GT + `--random-negatives 50`（同 seed `0`） | `runs/tokyo/cache_abl_random` | 无邻域 / 热门 / 用户常去；无 empty-union→全目录回退 |

```bash
# 基线：默认多源（复用 §0 cache；seed 默认 42）
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_cand_baseline \
  --seed 42 --device cuda
# → results/tokyo/abl_cand_baseline/eval_test/  （主看 Acc@1）

# 随机池：--candidate-radius random（须 random-negatives >= 1）
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_random \
  --candidate-radius random \
  --random-negatives 50 --random-negatives-seed 0 \
  --train-targets all --eval-targets all
python -m tsmoe train --cache runs/tokyo/cache_abl_random --out runs/tokyo/abl_cand_random \
  --seed 42 --device cuda
# → results/tokyo/abl_cand_random/eval_test/
```

**说明：** `random` 复用同一套 `--random-negatives` / `--random-negatives-seed` 抽样器；邻域 / popular / user-frequent **不参与**。池通常小于多源（约 GT+50），Acc@1 **未**按池大小归一化——若要对齐难度，可把 `--random-negatives` 调到接近基线 mean pool size（见 train 启动时的 `candidates … mean=`）。`ablation-summary` 已登记 `abl_cand_baseline` ↔ `p90`、`abl_cand_random` ↔ `random`。

#### 5b) 其它候选变体

```bash
# p80 / p70
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_p80 \
  --candidate-radius p80 --popular-radius max --popular-top-k 100 --random-negatives 50
# → …/cache_abl_p80/final.surd.pt（train/val/test 共用）
python -m tsmoe train --cache runs/tokyo/cache_abl_p80 --out runs/tokyo/abl_cand_p80 --device cuda

python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_p70 \
  --candidate-radius p70 --popular-radius max --popular-top-k 100 --random-negatives 50
python -m tsmoe train --cache runs/tokyo/cache_abl_p70 --out runs/tokyo/abl_cand_p70 --device cuda

# 关随机负例
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_rnd0 \
  --candidate-radius p90 --random-negatives 0
python -m tsmoe train --cache runs/tokyo/cache_abl_rnd0 --out runs/tokyo/abl_cand_rnd0 --device cuda

# 关热门
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_no_pop \
  --candidate-radius p90 --popular-top-k 0 --random-negatives 50
python -m tsmoe train --cache runs/tokyo/cache_abl_no_pop --out runs/tokyo/abl_cand_no_pop --device cuda

# 关用户常去
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_no_uf \
  --candidate-radius p90 --user-frequent-top-k 0 --random-negatives 50
python -m tsmoe train --cache runs/tokyo/cache_abl_no_uf --out runs/tokyo/abl_cand_no_uf --device cuda

# 仅邻域 + 随机
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_nbhd \
  --candidate-radius p90 --popular-top-k 0 --user-frequent-top-k 0 --random-negatives 50
python -m tsmoe train --cache runs/tokyo/cache_abl_nbhd --out runs/tokyo/abl_cand_nbhd --device cuda

# 全目录
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_full \
  --candidate-radius full
python -m tsmoe train --cache runs/tokyo/cache_abl_full --out runs/tokyo/abl_cand_full --device cuda
```

### 6) 目标协议（须重 prepare；SURD 仍为 train-all `final`）

```bash
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_tgt_all \
  --train-targets all --eval-targets all
# → final.surd.pt only（无 fold 表 / 无 --folds --warmup）
python -m tsmoe train --cache runs/tokyo/cache_abl_tgt_all \
  --out runs/tokyo/abl_tgt_all --device cuda

python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_tgt_last \
  --train-targets last --eval-targets last
python -m tsmoe train --cache runs/tokyo/cache_abl_tgt_last \
  --out runs/tokyo/abl_tgt_last --device cuda

python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_tgt_all_eval_last \
  --train-targets all --eval-targets last
python -m tsmoe train --cache runs/tokyo/cache_abl_tgt_all_eval_last \
  --out runs/tokyo/abl_tgt_train_all_eval_last --device cuda
```

### 6b) 未知 user/POI 过滤（STHGCN 式；须重 prepare；**默认 off**）

对齐 STHGCN `remove_unseen_user_poi`：在 **prepare** 时从 val/test **数据**中剔除（不是只改 eval 分母）。叠在默认基线 prepare 上：

```bash
# 独立 cache（勿覆盖默认 runs/tokyo/cache）
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_filter_unknown \
  --candidate-radius p90 --popular-radius max --popular-top-k 100 \
  --popular-by visit --user-frequent-top-k 50 --user-frequent-min-visits 2 \
  --random-negatives 50 --random-negatives-seed 0 \
  --train-targets all --eval-targets all \
  --filter-unknown on
# → final.surd.pt；manifest.audit 含 trails_dropped_unknown_user / samples_skipped_unknown_poi_target
python -m tsmoe train --cache runs/tokyo/cache_abl_filter_unknown \
  --out runs/tokyo/abl_filter_unknown --device cuda
# → results/tokyo/abl_filter_unknown/eval_test/
```

**中文对照 `known_only`：**

| | `--filter-unknown on` | eval `known_only` |
| --- | --- | --- |
| 何时生效 | **prepare**（写入 cache 的样本表） | **evaluate** 指标切片 |
| 未知用户 | val/test 中 **从未出现在 train** 的 user 整条 trail **删除** | 不处理用户 |
| 未知 POI（目录外真实下一站） | 该预测样本 **不进入** train/eval 数据 | 样本仍在；`all_targets` 记 miss，`known_only` 从分母去掉 |
| 默认 | **off**（现状不变） | 始终同时报告 |

---

## 6. 如何读 metrics / 比较

`train` 结束后会写 `results/<run>/eval_validation/` 与 `results/<run>/eval_test/`；或单独：

```bash
# evaluate 与 train 一样查 final.surd.pt（共享先验）
python -m tsmoe evaluate --cache runs/tokyo/cache \
  --checkpoint runs/tokyo/<run>/best.pt --split test --device cuda
# → results/tokyo/<run>/eval_test/{metrics.json,report.txt}

# 可选：核对某 POI 的 train-all 表（默认 --table final）
python -m tsmoe inspect --cache runs/tokyo/cache --table final --poi-id 3
```

**主键（`metrics.json`）：**

| 字段 | 含义 |
| --- | --- |
| `all_targets.Acc@1` | **主指标** |
| `all_targets.NDCG@1` | 并列对照 |
| `all_targets.MRR` | 全长 MRR（无截断） |
| `known_only.Acc@1` 等 | 仅 **目录内** 真实下一站样本（见 §6.1） |
| `candidate_construction` / `candidate_pool_stats` | 池构造与大小（核对 fingerprint） |
| `model_config` / `training_args` | 含 `router_mode`、因果 / overlay 等（以 checkpoint 为准） |

```bash
# 单跑
python -c "import json; m=json.load(open('results/tokyo/abl_router_hard/eval_test/metrics.json')); a=m['all_targets']; print(a['Acc@1'], a['NDCG@1'], a.get('MRR'))"

# 多跑总表（含 known_only + 命名校验；推荐）
python -m tsmoe ablation-summary --results results

# 或多跑并排（临时脚本）
python -c "
import json
from pathlib import Path
for p in sorted(Path('results/tokyo').glob('abl_*/eval_test/metrics.json')):
    a=json.load(open(p))['all_targets']
    print(f\"{p.parent.parent.name}\tAcc@1={a['Acc@1']:.4f}\tNDCG@1={a['NDCG@1']:.4f}\tMRR={a.get('MRR')}\")
"
```

Checkpoint 选型默认验证 **NDCG@10**（`--checkpoint-metric ndcg10`）；可用 `acc1` 按验证 Acc@1 选 `best.pt`。报告头条以 **Acc@1** 为准（见 [`results.md`](results.md)）。

### 6.1 `known_only`（know only）是什么？

**一句话（中文）：** `known_only` **不是**「只看候选池里有没有 target」（正例本来就会强制进池）；而是 **只保留「真实下一站 POI 属于训练 catalog（目录内）」的样本** 再算 Acc@1 / NDCG@1 / MRR。目录外 / 未知目标在 `all_targets` 记为 miss（rank=0，分母仍算），在 `known_only` 则从分母剔除。

细节：

| 切片 | 分母 | 目录外真实下一站 |
| --- | --- | --- |
| `all_targets`（主指标） | 全部 eval 样本 | rank 哨兵 `0`，计为 miss |
| `known_only` | 仅 `rank > 0` 的样本（目录内目标） | **不计入** |

实现见 `tsmoe/engine.py`：`metric_row` / `evaluate()` 返回的 `known_only=metric_row([r for r in ranks if r])`。报告里写作 `known_only (in-catalog targets)`。`all_targets.coverage` ≈ 目录内目标占比；总表的 `known_*` 列来自同一份 `metrics.json`。

若要在数据层面去掉未知 user/POI（STHGCN `remove_unseen_user_poi`），用 prepare `--filter-unknown on`（§3 因子表 / §5 命令 **6b**），**不要**把它和 `known_only` 混为一谈。

### 6.2 总表与 run-name ↔ flags 校验

```bash
python -m tsmoe ablation-summary --results results
# → results/ablation_summary.md + results/ablation_summary.csv
```

行为：

1. 扫描 `results/**/eval_test/metrics.json`（`--split validation` 则扫 `eval_validation`）。
2. 若存在 `results/ablation_log.csv` / `.md`，按 `run_name` 合并最新一行的 `key_flags` / `notes`。
3. 从 leaf 名（如 `tokyo/abl_ctx_cat_geo` → `abl_ctx_cat_geo`）查 §3 / §3.1 建议命名，与 `metrics.json` 的 `model_config` / `training_args` / `candidate_construction`（及 log 的 `key_flags`）对照。
4. `name_check` 列：`ok`（一致）/ `mismatch`（不一致，见 `mismatch` 列）/ `unknown_name`（不在命名表）/ `no_settings`（无记录可对）。
5. 主列仍是 **all_targets** Acc@1；另含 `known_Acc@1` / `known_NDCG@1` / `known_MRR` / `known_coverage`。

`--strict`：任一 `mismatch` 时进程 exit 1（便于 CI / 批量核对）。

---

## 7. 结果登记表（填写区）

**优先看自动日志：** 每次 `train` / `evaluate` 完成后，`results/ablation_log.md` 与 `.csv` 会追加一行（Acc@1 等来自本次 test / 所选 split；train 的 `notes` 含 best-val）。下面空表仍可手工对照；或从 [`ablation_log.template.md`](ablation_log.template.md) / [`.csv`](ablation_log.template.csv) 复制。`results/` 已 gitignore。

### 7.1 Router（同 cache）

| run_name | key flags | Acc@1 | NDCG@1 | MRR | notes |
| --- | --- | --- | --- | --- | --- |
| `abl_router_hard` | `--router-mode hard_prior` |  |  |  |  |
| `abl_router_surd_mlp` | `--router-mode surd_mlp` |  |  |  |  |
| `abl_router_hybrid` | `--router-mode hybrid --lambda-prior 1 --hybrid-lambda 0.1` |  |  |  |  |

### 7.2 训练负采样

| run_name | key flags | Acc@1 | NDCG@1 | MRR | notes |
| --- | --- | --- | --- | --- | --- |
| `abl_neg_63` | `--negatives 63 --max-train-candidates -1` |  |  |  |  |
| `abl_neg_all_max256` | `--negatives -1 --max-train-candidates 256` |  |  |  |  |

### 7.3 因果计数 / transition（#20 已在 main）

| run_name | key flags | Acc@1 | NDCG@1 | MRR | notes |
| --- | --- | --- | --- | --- | --- |
| `abl_causal_on` | `--causal-counts --transition-attn bias` |  |  |  | 默认 |
| `abl_causal_off` | `--no-causal-counts` |  |  |  |  |
| `abl_tr_bias` | `--transition-attn bias` |  |  |  |  |
| `abl_tr_gate` | `--transition-attn gate` |  |  |  |  |
| `abl_tr_rel_key` | `--transition-attn rel_key` |  |  |  |  |

### 7.4 候选变体（各有独立 cache）

| run_name | key flags | Acc@1 | NDCG@1 | MRR | notes |
| --- | --- | --- | --- | --- | --- |
| `abl_cand_baseline` | 默认多源（`--candidate-radius p90` + pop/uf/rnd50） |  |  |  | §5a；复用 `runs/tokyo/cache` |
| `abl_cand_random` | `--candidate-radius random --random-negatives 50` |  |  |  | §5a；re-prepare；仅随机池 |
| `abl_cand_p90`（基线 cache） | `--candidate-radius p90` + 默认多源 |  |  |  | 可用 `abl_router_hard` / `abl_cand_baseline` 同设置数字 |
| `abl_cand_p80` | `--candidate-radius p80` |  |  |  | re-prepare |
| `abl_cand_p70` | `--candidate-radius p70` |  |  |  | re-prepare |
| `abl_cand_rnd0` | `--random-negatives 0` |  |  |  | re-prepare |
| `abl_cand_no_pop` | `--popular-top-k 0` |  |  |  | re-prepare |
| `abl_cand_no_uf` | `--user-frequent-top-k 0` |  |  |  | re-prepare |
| `abl_cand_nbhd` | pop=0, uf=0, rnd=50 |  |  |  | re-prepare |
| `abl_cand_full` | `--candidate-radius full` |  |  |  | re-prepare |

### 7.5 目标协议

| run_name | key flags | Acc@1 | NDCG@1 | MRR | notes |
| --- | --- | --- | --- | --- | --- |
| `abl_tgt_all` | `--train-targets all --eval-targets all` |  |  |  | re-prepare |
| `abl_tgt_last` | `--train-targets last --eval-targets last` |  |  |  | re-prepare |
| `abl_tgt_train_all_eval_last` | `--train-targets all --eval-targets last` |  |  |  | re-prepare |

### 7.5b 未知 user/POI 过滤（STHGCN 式；独立 cache）

| run_name | key flags | Acc@1 | NDCG@1 | MRR | notes |
| --- | --- | --- | --- | --- | --- |
| `abl_filter_unknown` | `--filter-unknown on` |  |  |  | re-prepare；≠ `known_only` |

### 7.6 STHGCN T/L（无 CLI 消融）

| 项 | 状态 | notes |
| --- | --- | --- |
| `sthgcn_tl_features_v1` | **当前默认**（T=48, L=18） | 旧 `basic_fixed_features_v1` 已移除，**无开关**；勿与旧 cache 混用 |

### 7.7 SURD 表协议（无 CLI 消融）

| 项 | 状态 | notes |
| --- | --- | --- |
| train-all `final` 共用 | **当前默认 / 唯一协议** | 无 `--folds` / `--warmup`；训练也不再用 fold 表；旧 fold cache 须重 prepare |

### 7.8 Overlay A/B/C/D/E（同 cache；默认均 off）

| run_name | key flags | Acc@1 | NDCG@1 | MRR | notes |
| --- | --- | --- | --- | --- | --- |
| `abl_ctx_off` | （默认；无 overlay） |  |  |  | 基线 |
| `abl_ctx_cat_geo` | `--candidate-context on` |  |  |  | A |
| `abl_ctx_cat_geo_time` | `--candidate-context on --candidate-context-time last_slot` |  |  |  | A+time |
| `abl_ctx_cat_geo_causal_slot` | `--candidate-context on --candidate-context-time causal_slot` |  |  |  | A+causal_slot |
| `abl_sess_on` | `--session-summary on` |  |  |  | B |
| `abl_upref_on` | `--user-pref on` |  |  |  | C |
| `abl_tr_pmi` | `--transition-pmi on` |  |  |  | D |
| `abl_e1_hybrid` | `--expert-cross-attn hybrid` |  |  |  | E1+E3 |
| `abl_e1_dense` | `--expert-cross-attn hybrid --expert-cross-attn-sparse off` |  |  |  | E1 |
| `abl_e2_full` | `--expert-cross-attn full` |  |  |  | E2 |
| `abl_ctx_sess` | `--candidate-context on --session-summary on` |  |  |  | combo |
| `abl_ctx_sess_upref` | `… + --user-pref on` |  |  |  | combo |
| `abl_ctx_e1` | `--candidate-context on --expert-cross-attn hybrid` |  |  |  | combo |

### 7.9 SURD/SMI bias gates + LOCKED WIN（known_only；`cache_abl_tgt_last`）

Judge **`known_only`** Acc@1 / Recall@k vs STHGCN floors (0.255788 / 0.441345 / 0.511687 / 0.574311). Full recipe §3.2.

| run_name | key flags | Acc@1 | R@5 | R@10 | R@20 | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `hc_sg_Atime_bpr029_ssb026` | `dim256` `λ_bpr=0.29` `tsb=2` `unique_p` residual `ssb=0.26` A+time `ep1` `acc1` | **0.260274** | **0.441517** | **0.519494** | **0.586934** | **LOCKED WIN** · beats STHGCN Acc+R@1/5/10/20 |

---

## 8. 清单

- [ ] 基线 `runs/tokyo/cache` **按新指纹** prepare 完成（含 train-all `final` SURD；记下半径 / 源 / targets）
- [ ] Router 三档填入 §7.1
- [ ] 负采样填入 §7.2
- [ ] 因果 on/off + transition 三档填入 §7.3
- [ ] Overlay A/B/C/D/E 填入 §7.8（§3.1 表）
- [x] SURD-gate LOCKED WIN 填入 §7.9 / §3.2（`hc_sg_Atime_bpr029_ssb026`）
- [ ] 候选变体（含 §5a `abl_cand_baseline` / `abl_cand_random`）填入 §7.4
- [ ] 目标协议填入 §7.5
- [ ] `--filter-unknown on` 填入 §7.5b（独立 cache；对照 `known_only` 说明）
- [ ] 确认 `results/ablation_log.md` / `.csv` 已自动追加各跑（或从模板重置后再跑）
- [ ] 跑 `python -m tsmoe ablation-summary`，核对 `results/ablation_summary.*`（含 known_only 与 name_check）
