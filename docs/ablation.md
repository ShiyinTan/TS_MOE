# Ablation 实验手册

近期大量提交（多源候选、Router、STHGCN T/L、因果计数、目标协议等）已合入 `main`，但对 Acc@1 各自贡献尚不清楚。本文给出**固定协议 + 因子表 + 推荐顺序 + 可复制命令 + 结果登记表**，方便系统做对照实验。

配套说明：[`candidates.md`](candidates.md) · [`router-modes.md`](router-modes.md) · [`sthgcn-tl-features.md`](sthgcn-tl-features.md) · [`causal-counts.md`](causal-counts.md) · [`results.md`](results.md)。

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
| **同一 prepare 指纹** | 能复用同一 cache 的实验（Router / causal / transition / 训练负采样）**必须共用同一 `--cache`** |
| **须重 prepare** | 候选构造、目标协议、T/L 特征 schema 任一变化 → 重新 `prepare`（对照时用**独立** `--out` 保留多套 cache；同路径默认覆盖，见下表「依赖 prepare?」） |
| **默认基线 cache（Tokyo）** | `runs/tokyo/cache`：默认多源 `p90` + popular@max + user-frequent + random50；`train/eval-targets=all`；`feature_method=sthgcn_tl_features_v1` |
| **默认训练 knobs（除非该行明确改）** | `--router-mode hard_prior`；因果计数 **on**；`--transition-attn bias`；`--negatives 63`；`--max-train-candidates -1`；`--device cuda` |
| **对比公平性** | 同一 cache、同一 seed（默认 42）、独立 `--out` / `results/<run>`，避免结果目录互相覆盖 |

读 Acc@1：

```bash
python -c "import json; m=json.load(open('results/tokyo/abl_router_hard/eval_test/metrics.json')); print(m['all_targets']['Acc@1'], m['all_targets']['NDCG@1'], m['all_targets'].get('MRR'))"
```

可选：把填好的表另存为工作区本地日志（`results/` 已 gitignore）：

```bash
cp docs/ablation_log.template.md results/ablation_log.md
# 或
cp docs/ablation_log.template.csv results/ablation_log.csv
```

---

## 3. 因子 / 开关总表

| 因子 | 改什么 | 档位 / on·off | CLI | 依赖 prepare? | 建议 `run_name`（`--out` 末段） |
| --- | --- | --- | --- | --- | --- |
| **候选半径** | 局部邻域半径 | `p70` / `p80` / `p90`（默认）/ `mean` / `max` / `full` | `prepare --candidate-radius …` | **是** | `abl_cand_p90` / `abl_cand_p80` / `abl_cand_full` … |
| **多源：热门** | 大半径热门源 | on（默认 top-k=100）/ off（`--popular-top-k 0`）；半径 `max`/`p90`/…/`multiplier` | `--popular-top-k` `--popular-radius` `--popular-by` | **是** | `abl_cand_no_pop` / `abl_cand_pop_p80` |
| **多源：用户常去** | 用户训练常去 | on（默认 top-k=50）/ off（`--user-frequent-top-k 0`） | `--user-frequent-top-k` `--user-frequent-min-visits` | **是** | `abl_cand_no_uf` |
| **随机负例** | 池内额外均匀负例 | `50`（默认）/ `0`（关）/ 其它正整数 | `--random-negatives` `--random-negatives-seed` | **是** | `abl_cand_rnd0` / `abl_cand_rnd50` |
| **仅邻域+随机** | 关掉热门+用户常去 | 组合档 | `--popular-top-k 0 --user-frequent-top-k 0` | **是** | `abl_cand_nbhd` |
| **Router 模式** | 专家路由 π | `hard_prior`（默认）/ `surd_mlp` / `hybrid` | `train --router-mode …`；hybrid 可加 `--lambda-prior` `--hybrid-lambda` | 否 | `abl_router_hard` / `abl_router_surd_mlp` / `abl_router_hybrid` |
| **`--no-surd`** | 关掉 SURD 引导 | off（默认）/ on | `train --no-surd` | 否 | `abl_nosurd` |
| **STHGCN T/L** | T=48 / L=18 相对 Φ(Δt)/Φ(Δs) | **当前唯一 schema**（`sthgcn_tl_features_v1`）；旧 `basic_fixed_features_v1` **已移除、无 CLI 开关** | （无开关；换 schema 须改代码并 **重 prepare**） | **是**（若回退旧特征） | 记为「当前默认」；不可与旧 basic 同 cache 对比 |
| **因果计数整模块** | 在线访问/转移表 + 分箱 + transition attn | **on（默认）** / off | `--causal-counts` / `--no-causal-counts` | 否 | `abl_causal_on` / `abl_causal_off` |
| **transition-attn** | 转移如何进 P/C 交叉注意力 | `bias`（默认）/ `gate` / `rel_key` | `--transition-attn`；`--transition-lambda`（bias） | 否（须因果 **on**） | `abl_tr_bias` / `abl_tr_gate` / `abl_tr_rel_key` |
| **visit-bins** | T/L 用户累计访问等频分箱 | 默认 `10` | `--visit-bins` | 否（须因果 **on**） | `abl_visit_bins_10` |
| **训练目标协议** | 样本取「全部前缀」还是「仅末站」 | `all`（默认）/ `last` | `prepare --train-targets` | **是** | `abl_tgt_all` / `abl_tgt_last` / `abl_tgt_train_all_eval_last` |
| **评估目标协议** | val/test 样本协议 | `all`（默认）/ `last` | `prepare --eval-targets` | **是** | （同上） |
| **训练负采样** | 从构造池抽多少负例 | `63`（默认）/ `-1`（池内其余全要） | `train --negatives` | 否 | `abl_neg_63` / `abl_neg_all` |
| **训练候选封顶** | 含正例的训练候选上限 | `-1`（默认不封）/ 如 `256` | `train --max-train-candidates` | 否 | `abl_maxcand_256` |

> **#20 状态：** 因果在线计数 + `--transition-attn` 已合入 `main`（[#20](https://github.com/ShiyinTan/TS_MOE/pull/20)）。上表因果相关 flag 均为当前主干能力，**无需**再 checkout 旧分支。

细节见：候选 [`candidates.md`](candidates.md)；Router [`router-modes.md`](router-modes.md)；STHGCN [`sthgcn-tl-features.md`](sthgcn-tl-features.md)；因果 [`causal-counts.md`](causal-counts.md)。

---

## 4. 推荐顺序（便宜 → 贵）

同一 cache 上先扫 **训练期开关**（无需 re-prepare），再扫 **须重 prepare** 的构造，最后扫最贵的因果 / transition 细档（也可与 Router 同 cache 交叉，但建议一次一因子）。

| 步 | 做什么 | 是否 re-prepare | 说明 |
| ---: | --- | --- | --- |
| **0** | 准备基线 Tokyo cache | 一次 | `runs/tokyo/cache`（或明确记下指纹的等价 cache） |
| **1** | Router：`hard_prior` / `surd_mlp` / `hybrid` | 否 | 最便宜；同一 cache |
| **2** | 训练负采样：`negatives` / `max-train-candidates` | 否 | 同一 cache |
| **3** | 因果整模块：on vs `--no-causal-counts` | 否 | 同一 cache |
| **4** | transition：`bias` / `gate` / `rel_key`（因果 on） | 否 | 同一 cache |
| **5** | 候选变体：半径 / 关源 / 随机负例 / full | **是** | 每种独立 cache + model |
| **6** | 目标协议：`all`/`last` 组合 | **是** | 独立 cache |
| — | STHGCN T/L | — | **无开关**；旧 basic 特征已去掉，不能做 CLI 消融 |

---

## 5. 可复制命令（Tokyo）

均在仓库根目录、`conda activate tsmoe`。按 §4 顺序执行；每组用独立 `--out`，metrics 落在 `results/tokyo/<run>/eval_test/`。

### 0) 基线 prepare（只做一次，后续 1–4 步复用）

```bash
conda activate tsmoe
# 需已有 data/tokyo/*.csv（见 README §3.2）

python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache \
  --candidate-radius p90 --popular-radius max --popular-top-k 100 \
  --user-frequent-top-k 50 --user-frequent-min-visits 2 \
  --random-negatives 50 --random-negatives-seed 0 \
  --train-targets all --eval-targets all
```

可选 Demo 烟雾（同一逻辑，小模型）：

```bash
python examples/make_demo.py --data-root data
python -m tsmoe prepare --data-root data --city demo --out runs/demo/cache
```

### 1) Router 模式（同一 `runs/tokyo/cache`）

```bash
# hard_prior（默认）
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

### 5) 候选变体（每种 **重新 prepare**；独立 `--out` 以免覆盖）

```bash
# p80 / p70
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_p80 \
  --candidate-radius p80 --popular-radius max --popular-top-k 100 --random-negatives 50
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

### 6) 目标协议（须重 prepare）

```bash
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_abl_tgt_all \
  --train-targets all --eval-targets all
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

---

## 6. 如何读 metrics / 比较

`train` 结束后会写 `results/<run>/eval_validation/` 与 `results/<run>/eval_test/`；或单独：

```bash
python -m tsmoe evaluate --cache runs/tokyo/cache \
  --checkpoint runs/tokyo/<run>/best.pt --split test --device cuda
# → results/tokyo/<run>/eval_test/{metrics.json,report.txt}
```

**主键（`metrics.json`）：**

| 字段 | 含义 |
| --- | --- |
| `all_targets.Acc@1` | **主指标** |
| `all_targets.NDCG@1` | 并列对照 |
| `all_targets.MRR` | 全长 MRR（无截断） |
| `candidate_construction` / `candidate_pool_stats` | 池构造与大小（核对 fingerprint） |
| `config` | 含 `router_mode`、因果相关项等（以 checkpoint 为准） |

```bash
# 单跑
python -c "import json; m=json.load(open('results/tokyo/abl_router_hard/eval_test/metrics.json')); a=m['all_targets']; print(a['Acc@1'], a['NDCG@1'], a.get('MRR'))"

# 多跑并排（按需改路径列表）
python -c "
import json
from pathlib import Path
for p in sorted(Path('results/tokyo').glob('abl_*/eval_test/metrics.json')):
    a=json.load(open(p))['all_targets']
    print(f\"{p.parent.parent.name}\tAcc@1={a['Acc@1']:.4f}\tNDCG@1={a['NDCG@1']:.4f}\tMRR={a.get('MRR')}\")
"
```

Checkpoint 选型仍按验证 **NDCG@10**；报告头条以 **Acc@1** 为准（见 [`results.md`](results.md)）。

---

## 7. 结果登记表（填写区）

跑完后把数字填进下表（或复制 [`ablation_log.template.md`](ablation_log.template.md) / [`ablation_log.template.csv`](ablation_log.template.csv) 到 `results/ablation_log.*` 再填——`results/` 已 gitignore）。

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
| `abl_cand_p90`（基线 cache） | `--candidate-radius p90` + 默认多源 |  |  |  | 可用 `abl_router_hard` 同设置数字 |
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

### 7.6 STHGCN T/L（无 CLI 消融）

| 项 | 状态 | notes |
| --- | --- | --- |
| `sthgcn_tl_features_v1` | **当前默认**（T=48, L=18） | 旧 `basic_fixed_features_v1` 已移除，**无开关**；勿与旧 cache 混用 |

---

## 8. 清单

- [ ] 基线 `runs/tokyo/cache` prepare 完成，指纹记下（半径 / 源 / targets）
- [ ] Router 三档填入 §7.1
- [ ] 负采样填入 §7.2
- [ ] 因果 on/off + transition 三档填入 §7.3
- [ ] 候选变体（re-prepare）填入 §7.4
- [ ] 目标协议填入 §7.5
- [ ] （可选）复制模板到 `results/ablation_log.md` 或 `.csv` 长期保存
