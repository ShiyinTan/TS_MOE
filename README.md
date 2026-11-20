# TS-MoE · Massive-STEPS 下一 POI 推荐

Target-conditioned SURD-guided Mixture-of-Experts（TS-MoE）的 PyTorch 实现：从 Massive-STEPS 城市签到 CSV 构造同一 trail 内的下一跳预测样本，用训练数据估计逐 POI 的 26 维 SURD，再训练候选感知的 15 专家 Transformer。**默认候选集**是上一签到位置空间邻域、更大半径热门 POI、用户训练常去地点与随机负例的并集（去重；**目录内真目标必入池**，含 val/test；半径可选 `p70`/`p80`/`p90`/`mean`/`max`，或 `full` 退回全目录）。训练可从该池负采样 / 封顶；**评估对完整构造池排名**。另支持新轨迹推荐，以及从联合计数到单个候选得分的追踪。

算法设计见 [`TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md`](TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md)。本 README 只描述**当前代码**如何安装与运行。

**依赖：** Python 标准库 + NumPy + tqdm + PyTorch（`numpy>=1.26,<3`，`tqdm>=4.66,<5`，`torch>=2.4,<3`）。没有 `setup.py` / `pyproject.toml`：在仓库根目录执行 `python -m tsmoe ...`，不要 `pip install` 本包。

**路径约定：** 缓存 / checkpoint → `runs/`；评估 metrics / report → `results/`（**Acc@1** 为主指标；见 [`docs/results.md`](docs/results.md)）。多源候选见 [`docs/candidates.md`](docs/candidates.md)；Router 见 [`docs/router-modes.md`](docs/router-modes.md)；T/L ↔ STHGCN 见 [`docs/sthgcn-tl-features.md`](docs/sthgcn-tl-features.md)（换特征 schema 须 **重新 prepare**）。因果在线计数 + `--transition-attn` 见 [`docs/causal-counts.md`](docs/causal-counts.md)（§3.8；**默认 on**）。候选上下文 / 会话 / 用户偏好 / 专家内交叉注意力 overlays（**默认均 off**）：[`docs/candidate-context.md`](docs/candidate-context.md)。系统对照实验（因子表、overlay 命令表、结果登记）：[`docs/ablation.md`](docs/ablation.md)。

面向 **NVIDIA GPU 服务器**：conda 建环境 + pip 装 NumPy / tqdm / **CUDA 12.4** PyTorch。`train` / `evaluate` / `explain` / `recommend` 的 `--device` 默认为 `auto`（有 CUDA 用 GPU）。进度条默认开启（`tqdm`）。

---

## 目录

1. [安装](#1-安装)
2. [数据准备](#2-数据准备)
3. [运行（可复制命令）](#3-运行可复制命令)
   - [3.1 Demo](#31-demo无需下载)
   - [3.2 Tokyo](#32-tokyo下载--核对--prepare--cuda-训练)
   - [3.3 目标协议（`--train-targets` / `--eval-targets`）](#33-目标协议--train-targets--eval-targets)
   - [3.4 候选构造（多源 / 半径 / 随机负例 / GT）](#34-候选构造多源--半径--随机负例--gt)
   - [3.5 训练抽样](#35-训练抽样)
   - [3.6 Router 模式（`--router-mode`）](#36-router-模式--router-mode)
   - [3.7 STHGCN T/L 特征（须重 prepare）](#37-sthgcn-tl-特征须重-prepare)
   - [3.8 因果在线计数 + `--transition-attn`](#38-因果在线计数--transition-attn)
   - [3.8a 候选上下文 / 会话 / 偏好 / 专家内交叉注意力（默认 off）](#38a-候选上下文--会话--偏好--专家内交叉注意力默认-off)
   - [3.9 结果目录与 Acc@1](#39-结果目录与-acc1)
   - [3.10 独立评估](#310-独立评估)
   - [3.11 GPU 占用与中断](#311-gpu-占用与中断)
4. [配置 knobs](#4-配置-knobs)
5. [结果目录](#5-结果目录)
6. [仓库 / 代码布局](#6-仓库--代码布局)
7. [命令一览与其它入口](#7-命令一览与其它入口)
8. [算法要点（简要）](#8-算法要点简要)
9. [测试与东京烟雾参考](#9-测试与东京烟雾参考)
10. [当前局限](#10-当前局限)
11. [Ablation 实验手册](#11-ablation-实验手册)

---

## 1. 安装

需要 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) / Anaconda，以及可用的 NVIDIA 驱动（`nvidia-smi` 正常）。`environment.yml` 创建 **Python 3.12 + pip + NumPy + tqdm**；再用官方 **cu124** 索引装 GPU 版 PyTorch（**无需**本机完整 CUDA Toolkit）。

为何不用 `conda install pytorch pytorch-cuda`：conda 版 `libtorch_cpu.so` 动态链接 MKL，易与不兼容的 `mkl` / `intel-openmp` 冲突（`undefined symbol: iJIT_NotifyEvent`）。官方 **pip CUDA 轮子静态链接 MKL**，可避开该问题。

### 1.1 Conda + pip cu124（推荐）

```bash
cd TS_MOE   # 仓库根目录

conda env create -f environment.yml
conda activate tsmoe

# NumPy + tqdm（与 requirements.txt 一致；已有 env 缺依赖时可单独跑）
python -m pip install "numpy>=1.26,<3" "tqdm>=4.66,<5"

# CUDA 12.4 轮子
python -m pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cu124

# 确认：应打印 True，且 device 为 cuda:0（或同类）
python -c "import numpy, tqdm, torch; print(numpy.__version__, tqdm.__version__, torch.__version__, torch.cuda.is_available()); print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
```

只改了 `environment.yml`（Python / pip / NumPy / tqdm）时：

```bash
conda env update -f environment.yml --prune
```

PyTorch 请继续用上面的 `pip install ... --index-url .../cu124`（或下文 CPU / 其它 CUDA 索引），不要 `conda install pytorch`。

**CUDA 版本：** 默认 **cu124**。驱动偏旧时改用 `https://download.pytorch.org/whl/cu121` 或 `.../cu118`；对照 [PyTorch 安装页](https://pytorch.org/get-started/locally/)。

### 1.2 修复已有损坏环境（`iJIT_NotifyEvent`）

若曾用 conda 装过 `pytorch` / `pytorch-cuda` 并出现上述符号错误，在现有环境里改用 pip 轮子：

```bash
conda activate tsmoe
conda remove -y pytorch pytorch-cuda
python -m pip install --force-reinstall "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
```

仍报错时可再钉旧 OpenMP/MKL 后重装 torch：

```bash
conda activate tsmoe
conda install -y "mkl<2024.1" "intel-openmp<2024.1"
python -m pip install --force-reinstall "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cu124
```

仍不行则 `conda env remove -n tsmoe` 后按 §1.1 重建。

### 1.3 CPU / 无 GPU

```bash
conda activate tsmoe
python -m pip install "numpy>=1.26,<3" "tqdm>=4.66,<5"
python -m pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu
```

训练 / 评估等命令加 `--device cpu`（或依赖默认 `auto` 回退）。

### 1.4 备选：venv + pip（CUDA）

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install "numpy>=1.26,<3" "tqdm>=4.66,<5"
python -m pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cu124
```

`requirements.txt` 声明版本范围，不锁定 CUDA/CPU 轮子；GPU 服务器请用 cu124 索引装 torch。

---

## 2. 数据准备

所有命令在**仓库根目录**执行。`prepare` **没有** `--raw`：只读已切好的 Massive-STEPS 签到 CSV（Foursquare 风格字段）。

`prepare --data-root` 先找 `<data-root>/<city>_checkins_train.csv`；找不到再试 `<data-root>/<city>/<city>_checkins_train.csv`。因此下载到 `data/tokyo/` 时可用 `--data-root data --city tokyo`。

### 期望文件

```text
data/tokyo/
  tokyo_checkins_train.csv
  tokyo_checkins_validation.csv
  tokyo_checkins_test.csv
```

| CSV 字段 | 用途 |
| --- | --- |
| `trail_id`、`user_id` | 轨迹归属、组织事件、划分隔离 |
| `venue_id` | 历史 P、下一 POI 标签与候选目录中的原始整数 ID |
| `venue_category_id_code`、`venue_category_id` | 历史 C；优先训练集一致的 code |
| `timestamp` | 排序、STHGCN 小时/星期 one-hot、相对间隔 Φ(Δt)；因果计数按时间序 |
| `latitude`、`longitude` | 历史 L 的 Φ(Δs km) 与空间候选锚点 |

带时区时间戳转 UTC；等时事件按原文件行序稳定排序；以 `(user_id, trail_id, timestamp, venue_id)` 去重。审计写入缓存的 `manifest.json`。

### 多源候选池（prepare 时固定）

默认候选 = **邻域 ∪ 大半径热门 ∪ 用户常去 ∪ 随机负例**（去重；**目录内真目标必入池**，含 val/test）。`prepare --candidate-radius {p70,p80,p90,mean,max,full}`（**默认 `p90`**）定局部邻域；`--popular-radius` 可用 `max` / `p90` / `p80` / `p70` / `mean` / `multiplier`；另有 `--popular-top-k` / `--popular-by`、`--user-frequent-*`、`--random-negatives`（默认 **50**，`0` 关闭）均写入指纹。**train / evaluate 不覆盖**；改构造须 **重新 prepare**（同一 `--out` **默认覆盖**旧缓存；对照实验请用独立 `--out` 保留多套；`--no-overwrite` 可禁止覆盖）。邻域/热门/用户常去皆空才回退全目录。`train --negatives` / `--max-train-candidates` 仅训练抽样；**evaluate 用完整构造池**。缺坐标的目录 POI 会按 observed → adjacent_midpoint → trail_mean → global_mean 补全。`train` 启动时打印各 split 池大小统计并写入 `results/<run>/candidate_pool_stats.json`。详见 [`docs/candidates.md`](docs/candidates.md)。

`download_city.py` 支持城市：`bandung`、`beijing`、`istanbul`、`jakarta`、`kuwait_city`、`melbourne`、`moscow`、`new_york`、`palembang`、`petaling_jaya`、`sao_paulo`、`shanghai`、`sydney`、`tangerang`、`tokyo`。

完整可复制流水线见下一节。

---

## 3. 运行（可复制命令）

先 `conda activate tsmoe`，并在仓库根目录执行。仅 `train` / `evaluate` / `explain` / `recommend` 接受 `--device`（`auto` / `cpu` / `cuda` / `cuda:0` / `mps`）；`prepare` / `inspect` 为 CPU 侧。

### 3.1 Demo（无需下载）

`train` 结束会对最佳 checkpoint 自动跑 validation + **test**，并写入 `results/demo/model/eval_*`。

```bash
conda activate tsmoe

python examples/make_demo.py --data-root data
python -m tsmoe prepare --data-root data --city demo --out runs/demo/cache
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model \
  --epochs 1 --dim 16 --heads 2 --layers 1
# → checkpoint: runs/demo/model/best.pt
# → metrics:    results/demo/model/eval_validation/ 与 eval_test/
```

### 3.2 Tokyo：下载 → 核对 → prepare → CUDA 训练

默认邻域 `p90` + popular@max + user-frequent + `--random-negatives 50`。成功 prepare 后应有 `runs/tokyo/cache/manifest.json` 与各 split `.pt`。

```bash
conda activate tsmoe

# 1) 下载 Massive-STEPS Tokyo 三个 split（Hugging Face；需出网）
python examples/download_city.py --city tokyo --data-root data

# 2) 核对文件名与列（stdlib；不 import torch）
python scripts/verify_city_checkins.py --data-root data --city tokyo

# 3) 构建样本 + 训练集全量 SURD（train/val/test 共用）+ 多源候选
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache
# 等价显式：
# python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache \
#   --candidate-radius p90 --popular-radius max --popular-top-k 100 \
#   --popular-by visit --user-frequent-top-k 50 --user-frequent-min-visits 2 \
#   --random-negatives 50 --random-negatives-seed 0
# 日志：`final: N reference targets (train-all shared SURD); scored=0`
# → 只写 final.surd.pt / final.quantizer.pt；训练与 val/test 查同一张表（无 fold 表）。

# 4) GPU 训练（默认 --device auto；也可显式 cuda）
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model --device cuda
# → runs/tokyo/model/best.pt
# → results/tokyo/model/eval_validation/ 与 eval_test/
```

手动下载：从 [Massive-STEPS-Tokyo](https://huggingface.co/datasets/CRUISEResearchGroup/Massive-STEPS-Tokyo) 取得同名三个 CSV 放到 `data/tokyo/`（或扁平放在 `data/`），再跑上面第 2–4 步。固定版本：`python examples/download_city.py --city tokyo --data-root data --revision <commit-or-branch>`。

### 3.3 目标协议（`--train-targets` / `--eval-targets`）

**白话说明：** 一条轨迹有多个签到时，「预测目标」可以是**每个中间站的下一跳**（`all`），也可以**只预测该轨迹最后一个下一跳**（`last`）。这两个开关只属于 **`prepare`**：选好后写进缓存（`manifest` / 各 split 的 `start`/`end` 样本表）。之后的 **`train` / `evaluate` 不能改协议**——它们只读缓存里已经造好的样本。想换协议，**重新 prepare** 即可（同一 `--out` 默认覆盖；对照时用独立目录）。

| 协议 | 含义 | `prepare` 参数 |
| --- | --- | --- |
| 全部前缀训练 + 全部前缀评估（默认） | 轨迹上每个可用下一跳都是一条样本；val/test 同理 | `--train-targets all --eval-targets all` |
| 仅末站训练 + 仅末站评估 | 每条轨迹只保留最后一个下一跳 | `--train-targets last --eval-targets last` |
| 全部前缀训练 + 仅末站评估 | 训练用所有前缀；验证/测试只评末站 | `--train-targets all --eval-targets last` |

三种协议的完整可复制 prepare（Tokyo；候选用默认多源；每种协议独立 `--out`）：

```bash
conda activate tsmoe
# 需已有 data/tokyo/*.csv（见 §3.2）

# A) 默认：train=all，eval=all
python -m tsmoe prepare --data-root data --city tokyo \
  --out runs/tokyo/cache_targets_all \
  --train-targets all --eval-targets all

# B) 训练与评估都只看每条轨迹末站
python -m tsmoe prepare --data-root data --city tokyo \
  --out runs/tokyo/cache_targets_last \
  --train-targets last --eval-targets last

# C) 全部前缀训练，仅评估末站
python -m tsmoe prepare --data-root data --city tokyo \
  --out runs/tokyo/cache_train_all_eval_last \
  --train-targets all --eval-targets last
```

对应训练（缓存决定协议；train **没有** `--train-targets`）：

```bash
python -m tsmoe train --cache runs/tokyo/cache_targets_all \
  --out runs/tokyo/model_targets_all --device cuda
python -m tsmoe train --cache runs/tokyo/cache_targets_last \
  --out runs/tokyo/model_targets_last --device cuda
python -m tsmoe train --cache runs/tokyo/cache_train_all_eval_last \
  --out runs/tokyo/model_train_all_eval_last --device cuda
```

### 3.4 候选构造（多源 / 半径 / 随机负例 / GT）

**白话：** 候选池在 **prepare** 时按样本锚点（目标前一签）固定进缓存；train/evaluate **不覆盖**。默认 = **邻域 ∪ 大半径热门 ∪ 用户常去 ∪ 随机负例**（去重）。**训练目录内的真目标（GT）必入池**（含 val/test）。邻域半径可选 `p70` / `p80` / `p90`（默认）/ `mean` / `max`；`full` = 全目录。细节见 [`docs/candidates.md`](docs/candidates.md)。

对照多种半径 / 构造时用**独立** `--out`（及对应 train `--out`）以免互相覆盖；若只保留一套，对同一 `--out` 重新 prepare 即可（默认覆盖）。

```bash
conda activate tsmoe

# p90（默认局部邻域）+ popular@max + 随机负例 50
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_p90 \
  --candidate-radius p90 --popular-radius max --popular-top-k 100 \
  --random-negatives 50

# p80 / p70
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_p80 \
  --candidate-radius p80 --popular-radius max --popular-top-k 100 \
  --random-negatives 50
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_p70 \
  --candidate-radius p70 --popular-radius max --popular-top-k 100 \
  --random-negatives 50

# mean / max 局部邻域
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_mean \
  --candidate-radius mean --random-negatives 50
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_max \
  --candidate-radius max --random-negatives 50

# popular 半径也可用 p80 / p70（或 multiplier）
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_pop_p80 \
  --candidate-radius p90 --popular-radius p80 --popular-top-k 100
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_pop2x \
  --candidate-radius p90 --popular-radius multiplier --popular-radius-multiplier 2.0

# 仅邻域 + 随机负例（关掉 popular / user-frequent）
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_nbhd \
  --candidate-radius p90 --popular-top-k 0 --user-frequent-top-k 0 \
  --random-negatives 50

# full = 全训练目录（无空间 / 随机源）
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_full \
  --candidate-radius full
```

| `--candidate-radius` | 局部邻域 |
| --- | --- |
| `p90`（默认） | 连续签到距离的 90 分位（km） |
| `p80` / `p70` | 80 / 70 分位 |
| `mean` / `max` | 均值 / 最大值 |
| `full` | 关闭多源过滤；候选 = 整个训练 POI 目录 |

| `--popular-radius` | 大半径热门 |
| --- | --- |
| `max`（默认） / `p90` / `p80` / `p70` / `mean` | 同上统计量 |
| `multiplier` | `local_radius_km × --popular-radius-multiplier` |

### 3.5 训练抽样

训练从**完整构造池**里抽负例；**评估仍用完整池**（含 GT）。Acc@1 等写入 `results/`。

```bash
conda activate tsmoe

# 默认：每例 63 个负例；不额外封顶
python -m tsmoe train --cache runs/tokyo/cache_p90 --out runs/tokyo/model_p90 \
  --negatives 63 --max-train-candidates -1 --device cuda

# 池内其余全部作负例（仍可用 max-train-candidates 封顶）
python -m tsmoe train --cache runs/tokyo/cache_p90 --out runs/tokyo/model_p90_allneg \
  --negatives -1 --max-train-candidates 256 --device cuda

# 小模型连通性烟雾（见 §9）
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model \
  --epochs 1 --dim 16 --heads 2 --layers 1 --negatives 15 \
  --batch-size 64 --eval-batch-size 32 --chunk 256 --threads 3 --device cuda
```

### 3.6 Router 模式（`--router-mode`）

三种专家路由；**默认 `hard_prior`**。模式写入 checkpoint 的 `config`，`evaluate` / `explain` / `recommend` 自动与训练一致（无需再传 flag）。详见 [`docs/router-modes.md`](docs/router-modes.md)。

| 模式 | 路由 π | 序列是否进 Router |
| --- | --- | --- |
| `hard_prior`（默认） | 对候选 POI 的 **U/S SURD 质量**（`prior[..., :4]`∪`prior[..., 15:]`，15 维专家序）做 `ω · log1p(15 · mass)`，再 `/T` softmax；**无序列输入** | 否 |
| `surd_mlp` | `π = softmax(MLP(SURD_poi) / T)`；MLP 只看置信度缩放的 26 维先验 + ρ/α 摘要 | 否 |
| `hybrid` | `logits = λ_prior · surd_bias + λ_hybrid · learned(seq×cand×SURD)`；`--lambda-prior` 默认 **0.1**，`--hybrid-lambda`（ε）默认 **0.1** | 是（learned 项） |

**Demo（三种模式，可复制）：**

```bash
conda activate tsmoe
# 需已有 runs/demo/cache（见 §3.1）

# 1) hard_prior（默认；可省略 --router-mode）
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_hard \
  --router-mode hard_prior --epochs 1 --dim 16 --heads 2 --layers 1
python -m tsmoe evaluate --cache runs/demo/cache \
  --checkpoint runs/demo/model_hard/best.pt --split test
# → results/demo/model_hard/eval_test/

# 2) surd_mlp
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_surd_mlp \
  --router-mode surd_mlp --epochs 1 --dim 16 --heads 2 --layers 1
python -m tsmoe evaluate --cache runs/demo/cache \
  --checkpoint runs/demo/model_surd_mlp/best.pt --split test

# 3) hybrid（SURD bias + 小的序列学习项；ε = --hybrid-lambda）
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_hybrid \
  --router-mode hybrid --lambda-prior 1 --hybrid-lambda 0.1 \
  --epochs 1 --dim 16 --heads 2 --layers 1
python -m tsmoe evaluate --cache runs/demo/cache \
  --checkpoint runs/demo/model_hybrid/best.pt --split test
```

**Tokyo（三种模式，可复制）：**

```bash
conda activate tsmoe
# 需已有 runs/tokyo/cache（见 §3.2）

python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model_hard \
  --router-mode hard_prior --device cuda
python -m tsmoe evaluate --cache runs/tokyo/cache \
  --checkpoint runs/tokyo/model_hard/best.pt --split test --device cuda
# → results/tokyo/model_hard/eval_test/{metrics.json,report.txt}

python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model_surd_mlp \
  --router-mode surd_mlp --device cuda
python -m tsmoe evaluate --cache runs/tokyo/cache \
  --checkpoint runs/tokyo/model_surd_mlp/best.pt --split test --device cuda

python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model_hybrid \
  --router-mode hybrid --lambda-prior 1 --hybrid-lambda 0.1 --device cuda
python -m tsmoe evaluate --cache runs/tokyo/cache \
  --checkpoint runs/tokyo/model_hybrid/best.pt --split test --device cuda
```

### 3.7 STHGCN T/L 特征（须重 prepare）

当前 `main` 的 T/L **输入特征**已与 Massive-STEPS 基线 **STHGCN** 对齐（T=48、L=18；`feature_method=sthgcn_tl_features_v1`）。旧 `basic_fixed_features_v1` 缓存**不兼容**——须 **重新 prepare**（同一 `--out` 默认覆盖；无需先 `rm -rf`），再 train。详见 [`docs/sthgcn-tl-features.md`](docs/sthgcn-tl-features.md)。

```bash
conda activate tsmoe

# 新 schema：对现有目录重新 prepare（指纹不同则覆盖）；对照旧 cache 时再用独立 --out
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model_sthgcn --device cuda
# → results/tokyo/model_sthgcn/eval_validation/ 与 eval_test/
```

### 3.8 因果在线计数 + `--transition-attn`

运行时特征（**无需重新 prepare**）。默认 **开启**。配套说明：[`docs/causal-counts.md`](docs/causal-counts.md)。

**白话：** 运行时维护「预测时刻之前」已见过的访问 / 转移计数（先打分，再写入真实下一跳）。开启时训练按**预测时间戳升序、不 shuffle**；每个 epoch 重置计数表；val/test 从 **训练结束时的表**（checkpoint `count_tables`，或按序重放 train）克隆后继续同样协议。**无需重新 prepare**。

| Flag | 默认 | 含义 |
| --- | --- | --- |
| `--causal-counts` / `--no-causal-counts` | **on** | 整模块开关；关闭则恢复 shuffle 小批量 |
| `--transition-attn` | `bias` | 转移计数如何进入 P/C 交叉注意力：`bias` / `gate` / `rel_key` |
| `--transition-lambda` | `1.0` | `bias` 模式下 `λ · log1p(count)` 的尺度 |
| `--visit-bins` | `10` | 用户累计访问的**等频分箱**数 → 嵌入加到 T/L（不用原始计数） |

P 源用全局 **POI→POI**；C 源用 **cat→POI**。写入 checkpoint `config`（及 `count_tables` / `visit_bin_edges`），`evaluate` / `explain` / `recommend` 与训练一致。

**Demo（可复制）：**

```bash
conda activate tsmoe
# 需已有 runs/demo/cache（见 §3.1）

# 默认：因果计数 on + transition-attn=bias（可省略这两个 flag）
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_causal \
  --epochs 1 --dim 16 --heads 2 --layers 1

# 三种 transition 注入模式
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_bias \
  --transition-attn bias --transition-lambda 1.0 --visit-bins 10 \
  --epochs 1 --dim 16 --heads 2 --layers 1
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_gate \
  --transition-attn gate --epochs 1 --dim 16 --heads 2 --layers 1
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_rel_key \
  --transition-attn rel_key --epochs 1 --dim 16 --heads 2 --layers 1

# 关闭整个因果计数模块（恢复 shuffle）
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_nocounts \
  --no-causal-counts --epochs 1 --dim 16 --heads 2 --layers 1
```

**Tokyo：**

```bash
conda activate tsmoe
# 需已有 runs/tokyo/cache（见 §3.2）

python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model_causal \
  --transition-attn bias --device cuda
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model_gate \
  --transition-attn gate --device cuda
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model_rel_key \
  --transition-attn rel_key --device cuda
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model_nocounts \
  --no-causal-counts --device cuda
# → results/tokyo/model_*/eval_*/
```

| 模式 | 含义 |
| --- | --- |
| `bias`（默认） | 注意力 logit 加 `λ · log1p(转移计数)`；P 用 POI→POI，C 用 cat→POI |
| `gate` | `[历史; 候选; φ(count)]` → sigmoid 门控缩放注意力权重 |
| `rel_key` | `MLP(φ(count))` 注入 key（随候选变化） |

### 3.8a 候选上下文 / 会话 / 偏好 / 专家内交叉注意力（默认 off）

运行时 overlays（**无需 re-prepare**；**默认全部 off**，不改今日 Acc@1 路径）。新信号进 base / 共享 query / 专家 attn，**不进** `hard_prior`。说明与 defaults 矩阵：[`docs/candidate-context.md`](docs/candidate-context.md)；叠在默认 train 上的对照命令表：[`docs/ablation.md`](docs/ablation.md) §3.1。

| Flag | 默认 | 含义 |
| --- | --- | --- |
| `--candidate-context` | `off` | 候选 cat emb + Δgeo → 融合进候选 query |
| `--candidate-context-time` | `off` | `last_slot` / `causal_slot`（后者须因果 on） |
| `--session-summary` | `off` | 会话摘要 → shared base/context |
| `--user-pref` | `off` | 因果 user–cat / user–hour → base（须因果 on） |
| `--transition-pmi` | `off` | 转移 attn 用 PMI（须因果 on） |
| `--expert-cross-attn` | `off` | `hybrid`（E1）/ `full`（E2） |
| `--expert-cross-attn-sparse` | `on` | E≠off 时仅 top-k 源跑 expert attn（E3） |

```bash
# 示例：A + E1（同一 cache）
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_ctx_e1 \
  --candidate-context on --expert-cross-attn hybrid --device cuda
```

### 3.9 结果目录与 Acc@1

| 位置 | 内容 |
| --- | --- |
| `runs/<city>/cache/` | prepare 缓存 |
| `runs/<city>/model/` | `best.pt`、`history.json`（**不含**最终 metrics） |
| `results/<run>/eval_validation/` | train 结束后最佳模型的 validation |
| `results/<run>/eval_test/` | train 最终 test，或独立 `evaluate --split test`；默认另有 `epoch_*.json` |
| `results/<run>/history.json` | 每 epoch 的 val（+ test）Acc@1 等摘要 |
| `results/ablation_log.md` / `.csv` | train / evaluate 结束后自动追加一行 |
| `results/ablation_summary.md` / `.csv` | `python -m tsmoe ablation-summary` 总表（含 known_only） |
| `results/<run>/candidate_pool_stats.json` | 各 split 候选池大小统计 |

报告头条主指标为 **Acc@1**（另报 NDCG@1、MRR）；checkpoint 仍按验证 **NDCG@10** 选取。默认 **每个 epoch 也跑 test**（`--eval-test-every-epoch`；`--no-eval-test-every-epoch` 关闭）。完整约定见 [`docs/results.md`](docs/results.md)。

```bash
# 读 Acc@1（示例：§3.2 默认 Tokyo 跑完后）
python -c "import json; m=json.load(open('results/tokyo/model/eval_test/metrics.json')); print(m['all_targets']['Acc@1'])"
grep -E 'summary|Acc@1' results/tokyo/model/eval_test/report.txt
```

### 3.10 独立评估

对已有 cache + checkpoint 重跑排名（**完整构造池**；目录内 GT 必在池内）；默认写出 `results/<run>/eval_<split>/`。路由模式从 checkpoint 读取。

```bash
conda activate tsmoe

python -m tsmoe evaluate --cache runs/tokyo/cache \
  --checkpoint runs/tokyo/model/best.pt --split test --device cuda
# → results/tokyo/model/eval_test/{metrics.json,report.txt}

# 可选：覆盖输出目录，并额外镜像一份 JSON
python -m tsmoe evaluate --cache runs/tokyo/cache \
  --checkpoint runs/tokyo/model/best.pt --split test --device cuda \
  --out results/tokyo/eval_test --output results/tokyo/test.json
```

### 3.11 GPU 占用与中断

```bash
# 多卡时常显式指定
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model --device cuda:0
python -m tsmoe evaluate --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt --device cuda:0
```

用 `nvidia-smi` 对照占用；日志找 `Using device=cuda`。若 `--device auto` 落到 CPU，会打印 `WARNING: --device auto fell back to CPU...`。显存紧时减小 `--eval-batch-size` / `--chunk`（评估）或 `--batch-size` / `--negatives` / `--max-train-candidates`（训练）。**Ctrl+C** 在当前 step 边界干净退出（再按一次强制）。

---

## 4. 配置 knobs

与 `__main__.py` 默认值一致。入口：`python -m tsmoe <command> --help`。

| 参数 | 默认与含义 |
| --- | --- |
| `prepare --max-length`、`--hash-dim`、`--tau-pos` | 50、64、10 |
| `prepare --clusters` | 4（SURD 无 fold：去掉 `--folds` / `--warmup`；须重 prepare） |
| `prepare --candidate-radius` | `p70` / `p80` / `p90` / `mean` / `max` / `full`；局部邻域；写入缓存 |
| `prepare --popular-radius` | `max` / `p90` / `p80` / `p70` / `mean` / `multiplier`（默认 `max`） |
| `prepare --popular-radius-multiplier` | 默认 `2.0`（仅 `multiplier`） |
| `prepare --popular-top-k` / `--popular-by` | 默认 `100` / `visit`（`0` 关闭热门源） |
| `prepare --user-frequent-top-k` / `--user-frequent-min-visits` | 默认 `50` / `2`（`0` 关闭用户常去） |
| `prepare --random-negatives` / `--random-negatives-seed` | 默认 `50` / `0`（`0` 关闭随机负例） |
| `prepare --train-targets` / `--eval-targets` | `all` 或 `last`（默认均为 `all`）；**仅 prepare**；见 §3.3 |
| `prepare --no-overwrite` | 禁止覆盖已有 `--out`（默认允许：指纹不同或缓存不完整时就地重建） |
| `train --dim`、`--heads`、`--layers` | 128、4、2 |
| `train --epochs`、`--batch-size`、`--lr` | 20、32、0.001 |
| `train --top-k`、`--temperature` | 4、1.0 |
| `train --router-mode` | `hard_prior`（默认）/ `surd_mlp` / `hybrid`；写入 checkpoint |
| `train --lambda-prior`、`--hybrid-lambda` | 0.1、0.1（hybrid：`λ_prior·surd_bias + λ_hybrid·learned`） |
| `train --lambda-surd`、`--lambda-balance` | 0.05、0.001 |
| `train --negatives` | 每例 63 个负例（从构造池抽样）；`-1` 为池内其余全部 |
| `train --max-train-candidates` | 训练时每例候选总数上限（含正例）；`-1` 不额外封顶；**eval 不抽样** |
| `train --no-surd` | 关闭统计先验 / 偏置与两项 SURD 监督 |
| `train --causal-counts` / `--no-causal-counts` | 因果在线计数 + transition 交叉注意力（**默认 on**；见 §3.8） |
| `train --transition-attn` | `bias`（默认）/ `gate` / `rel_key` |
| `train --candidate-context` 等 overlays | **默认 off**；见 §3.8a / [`docs/candidate-context.md`](docs/candidate-context.md) |
| `train --transition-lambda` | bias 项尺度，默认 `1.0` |
| `train --visit-bins` | T/L 用户累计访问等频分箱数，默认 `10` |
| `train --eval-test-every-epoch` / `--no-eval-test-every-epoch` | 每 epoch 跑 test 并写 `results/.../eval_test/epoch_*.json`（**默认 on**） |
| `--device`（train / evaluate / explain / recommend） | 默认 `auto`；也可 `cpu`、`cuda`、`cuda:0`、`mps` |
| `--eval-batch-size`、`--chunk` | 评估批大小 16、每块候选 128 |
| `--threads` | PyTorch CPU 线程数，默认 2 |

改变簇数、窗口、**目标协议**（§3.3）、候选构造（§3.4）或 **SURD 协议**（train-all `final`；已无 `--folds`/`--warmup`）时，须 **重新 prepare**（同一 `--out` 默认覆盖；对照实验用独立目录），并用配套缓存训练与评估。**`train` 无法覆盖 prepare 已写入的目标协议、候选构造或 SURD 表。**

训练用 AdamW；按验证 **NDCG@10** 选 `best.pt`；终端 / 报告头条为 **Acc@1**（及 NDCG@1、MRR）。完整 `history.json` 在 `runs/.../model/`；紧凑版（含每 epoch test）在 `results/.../history.json`。训练结束自动 append `results/ablation_log.*`。

**`groups` 切片（非训练特征）：** 按真实下一站汇总 Acc@1 / NDCG@1 / MRR：

| 键前缀 | 依据 | 桶 |
| --- | --- | --- |
| `poi/` | 目标在训练 SURD 表中的 `target_count` | `outside_catalog`；`no_train_target`；`tail_1_2`；`mid_3_9`；`head_10_plus` |
| `history/` | 观测前缀长度 | `1`；`2_5`；`6_plus` |
| `last_category/` | 历史最后一签类别频次 | `0`；`1_9`；`10_plus` |

单目标下一 POI：`Acc@K = Hit@K = Recall@K`，`MAP@K = MRR@K`；另报无截断 `MRR`、`mean_rank`、`median_rank`（仅目录内目标）。K ∈ {1,5,10,20}。

---

## 5. 结果目录

与 §3.9 相同，摘要如下。完整说明：[`docs/results.md`](docs/results.md)。候选池与 Acc@1：[`docs/candidates.md`](docs/candidates.md)。

| 位置 | 内容 |
| --- | --- |
| `runs/<city>/cache/` | prepare 缓存（`manifest.json`、split `.pt`、SURD、候选构造） |
| `runs/<city>/model/` | `best.pt`、`history.json`（**不含**最终 metrics） |
| `results/<run>/eval_validation/` | train 结束后最佳模型的 validation metrics |
| `results/<run>/eval_test/` | 最终 test + 默认 `epoch_*.json`；或独立 `evaluate --split test` |
| `results/<run>/history.json` | 每 epoch val（+ test）Acc@1 摘要 |
| `results/ablation_log.md` / `.csv` | train / evaluate 自动追加 |
| `results/ablation_summary.md` / `.csv` | `python -m tsmoe ablation-summary` 生成的总表（Acc@1 + known_only + 命名校验） |
| `results/<run>/candidate_pool_stats.json` | train 启动时各 split 候选池大小（min/max/mean/median） |

每个 `eval_*` 含 `metrics.json` + `report.txt`。`<run>` 由 train `--out` / checkpoint 路径去掉前缀 `runs/` 得到，例如 `runs/tokyo/model` → `results/tokyo/model/`。可用 `evaluate --out <dir>` 覆盖。

---

## 6. 仓库 / 代码布局

```text
TS_MOE/
  environment.yml
  requirements.txt
  docs/results.md             # results/ 与 runs/ 分工、Acc@1
  docs/candidates.md          # 多源候选、抽样、回退、train/eval 一致
  docs/router-modes.md        # --router-mode hard_prior / surd_mlp / hybrid
  docs/sthgcn-tl-features.md  # T/L ↔ STHGCN Φ(Δt)/Φ(Δs) 对齐
  docs/causal-counts.md       # 因果在线计数 + --transition-attn
  docs/candidate-context.md   # overlays A/B/C/D/E（默认 off）
  docs/ablation.md            # 消融协议、因子表、可复制命令、结果登记
  docs/ablation_log.template.md  # 结果表模板（可复制到 results/）
  docs/ablation_log.template.csv
  tsmoe/                      # 包；入口 python -m tsmoe
  examples/                   # download_city、make_demo
  scripts/                    # verify_city_checkins 等
  tests/
  data/<city>/                # 签到 CSV（下载或自备）
  runs/<city>/cache/          # prepare 缓存
  runs/<city>/model/          # checkpoint（非 metrics）
  results/<city>/model/       # metrics.json / report.txt
```

| 路径 | 内容 |
| --- | --- |
| `environment.yml` | Conda 骨架（Python 3.12 + pip + NumPy + tqdm；torch 用 pip cu124） |
| `requirements.txt` | `numpy`、`tqdm`、`torch` 版本范围 |
| `docs/results.md` | 评估结果目录约定 |
| `docs/candidates.md` | 多源候选（邻域 + 热门 + 用户常去 + 随机负例；GT 必入池） |
| `docs/router-modes.md` | `--router-mode`：hard_prior / surd_mlp / hybrid |
| `docs/sthgcn-tl-features.md` | T/L 与 STHGCN 对齐；换 schema 须重 prepare |
| `docs/causal-counts.md` | 因果在线计数、等频分箱、`--transition-attn` |
| `docs/candidate-context.md` | 候选上下文 / 会话 / 用户偏好 / PMI / 专家内交叉注意力 |
| `docs/ablation.md` | 消融实验手册（开关总表、顺序、命令、结果登记） |
| `docs/ablation_log.template.md` / `.csv` | 结果登记模板（运行时自动写 `results/ablation_log.*`） |
| `examples/download_city.py` | 从 Hugging Face 拉某城市三个 CSV |
| `examples/make_demo.py` | 合成 demo CSV |
| `scripts/verify_city_checkins.py` | 核对 prepare 所需文件名与列 |
| `tsmoe/data.py` | CSV 读取、类别对齐、时间与地理特征 |
| `tsmoe/candidates.py` | 距离统计、多源池、训练负采样 |
| `tsmoe/quantizer.py` | 四源摘要、标准化、K-means |
| `tsmoe/surd.py` | 联合频数表、26 项 SURD、冻结先验 |
| `tsmoe/prepare.py`、`tsmoe/inspect.py` | 训练全量 SURD、缓存、逐样本追踪 |
| `tsmoe/counts.py` | 因果在线访问 / 转移计数表、等频分箱 |
| `tsmoe/model.py` | Transformer、稀疏专家、Router、损失、transition 交叉注意力 |
| `tsmoe/engine.py`、`tsmoe/__main__.py` | 训练、评估、推荐、解释与 CLI |

数据来源：[Massive-STEPS](https://github.com/CRUISEResearchGroup/Massive-STEPS) · [next-POI 划分脚本](https://github.com/CRUISEResearchGroup/Massive-STEPS/blob/main/src/create_next_poi_dataset.py) · [Tokyo 数据集](https://huggingface.co/datasets/CRUISEResearchGroup/Massive-STEPS-Tokyo)。

---

## 7. 命令一览与其它入口

| 命令 | 作用 |
| --- | --- |
| `prepare` | CSV → 样本、train-all SURD（`final`）、候选半径 → `--out` 缓存 |
| `train` | 训练（共用 `final` 先验）；按验证 NDCG@10 存 `best.pt`；每 epoch 默认 test；写 `results/.../eval_*` 与 `ablation_log.*` |
| `evaluate` | 在缓存候选集上排名；默认 `results/.../eval_<split>/` |
| `ablation-summary` | 扫描 `results/**/eval_*/metrics.json`（+ `ablation_log.*`）→ `results/ablation_summary.md/.csv`（含 known_only 与 run-name↔flags 校验） |
| `inspect` | 查看某 POI 的 26 维 SURD |
| `recommend` | 对新历史 CSV 中每条 trail 做下一站 Top-N |
| `explain` | 审计一个候选的先验、路由与得分 |

```bash
# 离线 SURD
python -m tsmoe inspect --cache runs/tokyo/cache --table final --poi-id 3 \
  --sample-index 0 --output runs/tokyo/offline_trace.json

# 推荐 / 案例解释（--history 字段与官方签到 CSV 相同）
python -m tsmoe recommend --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt \
  --history my_history.csv --top-n 10 --output recommendations.json --device cuda
python -m tsmoe explain --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt \
  --split test --sample-index 0 --output case.json --device cuda
```

`explain` 默认解释真实目标；目标不在训练目录时解释最高分候选。可用 `--poi-id` 指定目录内候选。

---

## 8. 算法要点（简要）

与设计文档一致、当前实现已落地：

1. 四源历史摘要（P/C 哈希；T/L 为 STHGCN 对齐数值 T=48、L=18）→ 每源标准化 + K-means 状态。
2. 单一联合频数表边缘化 → 26 维（4 Unique + 11 Redundancy + 11 Synergy）；15 个 U/S 专家在线执行。
3. **SURD 先验（无 fold）：** 在整份训练集上估计逐 POI 的 26 维表，写入 `final.surd.pt`；**train / val / test 共用同一张表**（不再为时间折各算一套）。改协议后须重 prepare。
4. 在线：四源 Transformer 编码历史（T/L MLP→`dim`）；**Router 默认 `hard_prior`**（仅候选 U/S SURD → π，无序列）；可选 `surd_mlp` / `hybrid`；Top-K 稀疏专家；`base + 门控 MoE 残差`。
5. 候选：prepare 写入的多源池（邻域 + 热门 + 用户常去 + 随机负例，或 `full`）；目录内 GT 必入池；train 可抽样，evaluate 用完整池。
6. **因果在线计数**（默认 on）：按时间戳训练；用户/全局访问与转移表先读后写；POI/C 嵌入加计数 MLP；T/L 加等频访问分箱；P/C 交叉注意力默认 `bias` 注入转移（可选 `gate` / `rel_key`）。见 [`docs/causal-counts.md`](docs/causal-counts.md) 与 §3.8。
7. **Overlays（默认 off）**：候选上下文（cat+Δgeo±时段）、会话摘要、用户偏好、转移 PMI、专家内交叉注意力（E1 hybrid / E2 full / E3 sparse）；均不进 `hard_prior`。见 [`docs/candidate-context.md`](docs/candidate-context.md) 与 §3.8a。

缓存 `manifest.json` 记录字段约定、`feature_method=sthgcn_tl_features_v1`、目录、SURD `final` 表、候选构造与文件摘要。旧 `basic_fixed_features_v1` 或含 fold SURD 的缓存不兼容，须重新 prepare（默认覆盖同路径旧缓存）。

---

## 9. 测试与东京烟雾参考

```bash
python -m unittest discover -s tests -v
python -m tsmoe.surd
```

**东京烟雾**（[`examples/tokyo_smoke.json`](examples/tokyo_smoke.json)）：训练 / 验证 / 测试前缀 5,855 / 808 / 1,694；训练目录 2,494 POI。连通性示例：

```bash
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model \
  --epochs 1 --dim 16 --heads 2 --layers 1 --negatives 15 \
  --batch-size 64 --eval-batch-size 32 --chunk 256 --threads 3 --device cuda
```

该设置下测试集 1,453 / 1,694 目标在目录内；全体目标 Recall@10 ≈ 0.1582，NDCG@10 ≈ 0.0920（单轮小模型，非完整训练指标）。

---

## 10. 当前局限

- **未打包安装**：无 `pip install -e .`；须在仓库根目录用 `python -m tsmoe`。
- **无独立配置文件**：超参均通过 CLI；无 YAML/JSON 训练配置入口。
- **无服务化推理 API**：入口是 `recommend` / `explain` CLI。
- **默认 pip 索引为 cu124**：更旧驱动需改用 cu121 / cu118；不自动探测本机 CUDA。
- **无多进程 / 多卡数据并行开关**：可选单卡 `--device cuda:0`；无 `--gpus` / DDP CLI。
- **大数据内存**：大型城市 prepare 峰值随训练前缀增长；无分片 / 外存统计路径。
- **换候选构造或目标协议须重 prepare**：train / evaluate 不能单独改 `--candidate-radius` / popular / user-frequent / random-negatives，也不能覆盖 `--train-targets` / `--eval-targets`（见 §3.3、§3.4）。
- **换 T/L 特征 schema 须重 prepare**：旧 `basic_fixed_features_v1` 与当前 `sthgcn_tl_features_v1` 不兼容（见 §3.7）。

---

## 11. Ablation 实验手册

近期改动多，应用对照实验分清贡献：固定同一 prepare 指纹（能复用则复用）、以 **Acc@1** 为主、结果写入 `results/<run>/`。

`train` / `evaluate` 结束后会**自动 append** 到 `results/ablation_log.md` 与 `.csv`（无则创建）。扫已有结果生成总表：

```bash
python -m tsmoe ablation-summary --results results
```

**`known_only`：** 不是「候选池里有没有 target」，而是只统计真实下一站在训练 catalog 内的样本（见 [`docs/ablation.md`](docs/ablation.md) §6.1）。

完整内容（因子/开关总表、推荐顺序、Tokyo 可复制命令、结果登记空表、如何读 `metrics.json`、命名校验）：

- [`docs/ablation.md`](docs/ablation.md)
- 结果模板（可选 `cp` 到 `results/` 作空表 / 重置）：[`docs/ablation_log.template.md`](docs/ablation_log.template.md) · [`docs/ablation_log.template.csv`](docs/ablation_log.template.csv)
