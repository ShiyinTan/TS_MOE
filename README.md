# TS-MoE · Massive-STEPS 下一 POI 推荐

Target-conditioned SURD-guided Mixture-of-Experts（TS-MoE）的 PyTorch 实现：从 Massive-STEPS 城市签到 CSV 构造同一 trail 内的下一跳预测样本，用训练数据估计逐 POI 的 26 维 SURD，再训练候选感知的 15 专家 Transformer。支持全目录评估、新轨迹推荐，以及从联合计数到单个候选得分的追踪。

算法设计见仓库根目录 [`TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md`](TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md)。本 README 只描述**当前代码**如何安装与运行。

**依赖是 Python 标准库 + NumPy + tqdm + PyTorch**（`numpy>=1.26,<3`，`tqdm>=4.66,<5`，`torch>=2.4,<3`）。没有 `setup.py` / `pyproject.toml`：请在仓库根目录执行 `python -m tsmoe ...`，不要 `pip install` 本包。

面向 **NVIDIA GPU 服务器**：用 conda 建 Python 环境，再用 **pip** 装 NumPy、tqdm 与 **CUDA 12.4 轮子**的 PyTorch。`train` / `evaluate` / `explain` / `recommend` 的 `--device` 默认为 `auto`（检测到 CUDA 时用 GPU）；训练开始会打印 `Using device=...`。东京连通性烟雾曾在 CPU 上跑通；GPU 上用同一 CLI，换 `--device` 即可。进度条默认开启（`tqdm`；无 TTY 时仍会按行刷新）。

---

## 1. 用 Conda 创建 CUDA 环境（推荐）

需要已安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda，以及可用的 NVIDIA 驱动（`nvidia-smi` 正常）。`environment.yml` 创建 **Python 3.12 + pip + NumPy + tqdm**；随后从 PyTorch 官方 cu124 索引安装 GPU 轮子（**无需**本机再装完整 CUDA Toolkit）。

为何不用 `conda install pytorch pytorch-cuda`：conda 版 `libtorch_cpu.so` 动态链接 MKL / `intel-openmp`，solver 若拉到不兼容的 MKL（常见于 2024.1+ / 2025.x）会在 `import torch` 时报 `undefined symbol: iJIT_NotifyEvent`。官方 **pip CUDA 轮子静态链接 MKL**，可避开该冲突。

```bash
cd TS_MOE   # 仓库根目录

conda env create -f environment.yml
conda activate tsmoe

# NumPy + tqdm（PyPI；与 requirements.txt 一致）。已有 env 缺依赖时可单独跑。
python -m pip install "numpy>=1.26,<3" "tqdm>=4.66,<5"

# CUDA 12.4 构建；与 requirements.txt 的版本范围一致
python -m pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cu124

# 确认：应打印 True，且 device 为 cuda:0（或同类）
python -c "import numpy, tqdm, torch; print(numpy.__version__, tqdm.__version__, torch.__version__, torch.cuda.is_available()); print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
```

之后若只改了 `environment.yml`（Python / pip / NumPy / tqdm）：

```bash
conda env update -f environment.yml --prune
```

PyTorch 本身请继续用上面的 `pip install ... --index-url .../cu124`（或下文 CPU / 其它 CUDA 索引）管理，不要再 `conda install pytorch`。

**CUDA 版本假设：** 默认按常见 GPU 服务器选用 **cu124**。若驱动偏旧，把索引改成 `https://download.pytorch.org/whl/cu121` 或 `.../cu118` 后重装 torch；对照 [PyTorch 安装页](https://pytorch.org/get-started/locally/) 选择匹配组合。

### 修复已有损坏的 `tsmoe` 环境（`iJIT_NotifyEvent`）

若已按旧 README 用 conda 装过 `pytorch` / `pytorch-cuda`，且出现：

`ImportError: .../torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`

在**现有**环境里卸掉 conda 的 PyTorch 栈，改用 pip 轮子（不必删整个 env）：

```bash
conda activate tsmoe

# 卸掉旧 README 用 conda 装上的动态链接 torch 栈
conda remove -y pytorch pytorch-cuda

# 用官方 CUDA 12.4 轮子重装（静态链接 MKL）
python -m pip install --force-reinstall "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
```

若仍报同样符号错误，多半是环境里残留了冲突的 OpenMP/MKL；可再执行：

```bash
conda activate tsmoe
conda install -y "mkl<2024.1" "intel-openmp<2024.1"
python -m pip install --force-reinstall "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
```

仍不行则重建：`conda env remove -n tsmoe` 后按本节开头的「创建」步骤重来。

### 在 GPU 上跑（与 CLI 一致）

仅 `train`、`evaluate`、`explain`、`recommend` 接受 `--device`（`prepare` / `inspect` 为 CPU 侧数据与 SURD 构建，无该参数）。合法取值与 `__main__.py` 一致：`auto`（默认）、`cpu`、`cuda`、`cuda:0`、`mps`。

```bash
conda activate tsmoe

# --cache 须先由 prepare 写出（含 manifest.json），例如：
#   python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache
# 默认 --device auto：有 CUDA 则用 GPU；启动应看到 Using device=cuda ... gpu='...'
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model
python -m tsmoe evaluate --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt --split test

# 显式指定 GPU（多卡时常用 cuda:0）
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model --device cuda
python -m tsmoe evaluate --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt --device cuda:0
```

用 `nvidia-smi` 对照进程 GPU 占用；日志里找 `Using device=cuda` / `model_param_device=cuda:...`。若 `--device auto` 静默落到 CPU，会打印 `WARNING: --device auto fell back to CPU...`。训练 / 评估过程中 **Ctrl+C** 会在当前 step 结束后干净退出（再按一次强制退出）。

显存紧张时可减小 `--eval-batch-size` / `--chunk`（评估）或 `--batch-size` / `--negatives`（训练）；这些标志已存在于 CLI。

### CPU / 无 GPU 回退

无 GPU 或只需 CPU 时：仍可用同一 conda 环境，对训练与评估等命令传 `--device cpu`。装 CPU 版 PyTorch：

```bash
conda activate tsmoe
python -m pip install "numpy>=1.26,<3" "tqdm>=4.66,<5"
python -m pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu
```

### 备选：venv + pip（CUDA）

不需要 conda 时：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install "numpy>=1.26,<3" "tqdm>=4.66,<5"
# 与上文 conda 路径相同的 CUDA 12.4 轮子；其它版本见 pytorch.org
python -m pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cu124
```

`requirements.txt` 声明 `numpy>=1.26,<3`、`tqdm>=4.66,<5` 与 `torch>=2.4,<3`，不锁定 CUDA/CPU 轮子；GPU 服务器请用上面的 cu124 索引装 torch。

---

## 2. 目录约定

所有命令均在**仓库根目录**执行。典型布局：

```text
TS_MOE/
  environment.yml
  requirements.txt
  tsmoe/                 # 包；入口 python -m tsmoe
  examples/              # 下载城市数据、生成 demo CSV
  scripts/               # 布局核对等辅助脚本（不改 prepare CLI）
  tests/
  data/<city>/           # 下载或自备的签到 CSV（可按城市分子目录）
  runs/<city>/cache/     # prepare 产出的样本与 SURD 缓存
  runs/<city>/model/     # train 产出的 best.pt、history.json
```

`prepare --data-root` 会先找 `<data-root>/<city>_checkins_train.csv`；找不到则再试 `<data-root>/<city>/<city>_checkins_train.csv`。因此下载脚本写到 `data/tokyo/` 时，可直接用 `--data-root data --city tokyo`。东京逐步命令见 **§4 Prepare Tokyo data**（`prepare` 无 `--raw`）。

期望的城市签到文件：

```text
data/tokyo/
  tokyo_checkins_train.csv
  tokyo_checkins_validation.csv
  tokyo_checkins_test.csv
```

| CSV 字段 | 用途 |
| --- | --- |
| `trail_id`、`user_id` | 轨迹归属、组织事件、划分隔离、案例定位 |
| `venue_id` | 历史 P、下一 POI 标签与候选目录中的原始整数 ID（保留有效值 0） |
| `venue_category_id_code`、`venue_category_id` | 历史 C；优先训练集中一致的 code；支持只含其中一个字段的 CSV |
| `timestamp` | 排序、周期特征、间隔、时间折边界 |
| `latitude`、`longitude` | 历史 L 的球面坐标、位移、方向及有效位 |

采用官方预处理生成的**逐签到 CSV**（Massive-STEPS / Foursquare 风格字段）。带时区的时间戳转成 UTC，无时区按 UTC 解释。等时事件按原文件行序稳定排序。以 `(user_id, trail_id, timestamp, venue_id)` 去重（保留第一条）。不可解析时间戳整行隔离；缺失类别或坐标仍保留事件并编码。审计写入缓存的 `manifest.json`。

---

## 3. 最快跑通：合成 demo（无需下载）

```bash
conda activate tsmoe

python examples/make_demo.py --data-root data
python -m tsmoe prepare --data-root data --city demo --out runs/demo/cache
# train / evaluate 默认 --device auto（GPU 可用则走 CUDA）
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model --epochs 1 --dim 16 --heads 2 --layers 1
# 评估结果默认写入 <checkpoint 目录>/eval_<split>/{metrics.json,report.txt}
python -m tsmoe evaluate --cache runs/demo/cache --checkpoint runs/demo/model/best.pt
# 等价显式写法：在上述两条命令末尾加 --device cuda 或 --device cuda:0
```

---

## 4. Prepare Tokyo data

`prepare` **没有** `--raw`（或其它“从原始 dump 转换”）标志。它只读已经切好的 Massive-STEPS 签到 CSV（Foursquare 风格字段，与设计文档 [`TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md`](TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md) §2.1 一致）。仓库不附带东京原始数据；请先下载公开 CSV，再本地跑 `prepare`。

### 期望目录与文件名

`prepare` 先找 `<data-root>/tokyo_checkins_train.csv`；没有则用 `<data-root>/tokyo/`（`examples/download_city.py` 默认写这里）：

```text
data/tokyo/
  tokyo_checkins_train.csv
  tokyo_checkins_validation.csv
  tokyo_checkins_test.csv
```

每个 CSV 至少需要：`trail_id`、`user_id`、`venue_id`、`latitude`、`longitude`、`timestamp`，以及 `venue_category_id_code` 和/或 `venue_category_id`。公开东京文件还含 `name`、`address`、`venue_category`、`venue_city*`、`venue_country` 等展示字段（`prepare` 不依赖它们）。

### 从公开文件到 cache（复制即可）

在**仓库根目录**、已 `conda activate tsmoe`（且已装好 torch）时：

```bash
# 1) 下载 Massive-STEPS Tokyo 三个 split（Hugging Face；需出网）
python examples/download_city.py --city tokyo --data-root data

# 2) 可选：核对文件名与列（stdlib；不 import torch）
python scripts/verify_city_checkins.py --data-root data --city tokyo

# 3) 构建样本 + 折外 SURD 缓存（真实 CLI；无 --raw）
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache
```

成功后应看到 `runs/tokyo/cache/manifest.json` 以及 `train.pt` / `validation.pt` / `test.pt` 等。同一配置可重复跑 `prepare`（指纹一致则复用缓存）。

### 手动下载（跳过 `download_city.py`）

从 [Massive-STEPS-Tokyo](https://huggingface.co/datasets/CRUISEResearchGroup/Massive-STEPS-Tokyo) 取得同名三个 CSV，放到 `data/tokyo/`（或扁平放在 `data/` 下且文件名仍为 `tokyo_checkins_*.csv`），然后只跑上面的第 2–3 步。固定版本可：`python examples/download_city.py --city tokyo --data-root data --revision <commit-or-branch>`。

### 下载之后：训练 / 评估

```bash
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model --device cuda
# 默认写出 runs/tokyo/model/eval_test/{metrics.json,report.txt}；也可用 --out / --output
python -m tsmoe evaluate --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt \
  --split test --device cuda
python -m tsmoe evaluate --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt \
  --split test --out runs/tokyo/eval_test --output runs/tokyo/test.json --device cuda
```

省略 `--device` 时默认为 `auto`。其它城市把 `--city` / 路径里的 `tokyo` 换成对应前缀即可；`download_city.py` 支持：`bandung`、`beijing`、`istanbul`、`jakarta`、`kuwait_city`、`melbourne`、`moscow`、`new_york`、`palembang`、`petaling_jaya`、`sao_paulo`、`shanghai`、`sydney`、`tangerang`、`tokyo`。

### 选择预测目标

默认 `--train-targets all --eval-targets all`（与设计文档一致）。一条 `A → B → C → D` 产生三个样本，目标分别为 B、C、D；每个样本输入止于前一事件。训练 / 验证 / 测试沿用官方 CSV 的轨迹成员关系。

| 协议 | `prepare` 参数 |
| --- | --- |
| 每个前缀均评分（默认） | `--train-targets all --eval-targets all` |
| 训练与评估均用每条轨迹末站 | `--train-targets last --eval-targets last` |
| 全部前缀训练，仅评估末站 | `--train-targets all --eval-targets last` |

改变簇数、窗口或目标选取时，请换新的 `--out` 缓存目录，并用配套缓存训练与评估。

---

## 5. 命令一览

入口：`python -m tsmoe <command> ...`（`python -m tsmoe --help` / `python -m tsmoe <command> --help`）。

| 命令 | 作用 |
| --- | --- |
| `prepare` | 读 CSV → 样本、折外量化器、逐 POI SURD 表 → 写入 `--out` 缓存 |
| `train` | 用缓存训练；按验证集 NDCG@10 保存 `best.pt`；结束时写 `eval_validation/`，并对最佳模型跑一次 test → `eval_test/`（摘要以 Acc@1 打头） |
| `evaluate` | 全目录精确排名；终端 / `report.txt` 以 Acc@1 / NDCG@1 / MRR 为头条；完整 Acc/Hit/Recall、NDCG、MRR、MAP @1/5/10/20 + groups |
| `inspect` | 查看某 POI 的 26 维 SURD；可选某训练前缀的簇与联合计数 |
| `recommend` | 对新历史 CSV 中每条 trail 做下一站 Top-N |
| `explain` | 审计一个候选的先验、路由与得分贡献 |

### 离线 SURD 检查

```bash
python -m tsmoe inspect --cache runs/tokyo/cache --table final --poi-id 3 --sample-index 0 --output runs/tokyo/offline_trace.json
```

### 推荐与案例解释

`--history` 需与官方签到 CSV 字段相同。每条 trail 至少一条已观测签到；输出使用原始 `venue_id`。

```bash
python -m tsmoe recommend --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt \
  --history my_history.csv --top-n 10 --output recommendations.json --device cuda
python -m tsmoe explain --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt \
  --split test --sample-index 0 --output case.json --device cuda
```

`explain` 默认解释真实目标；若目标不在训练目录，则解释最高分候选并记录真实目标。可用 `--poi-id` 指定目录内候选。

### 重要配置（与代码默认值一致）

| 参数 | 默认与含义 |
| --- | --- |
| `prepare --max-length`、`--hash-dim`、`--tau-pos` | 50、64、10 |
| `prepare --clusters`、`--folds`、`--warmup` | 4、3、0.2 |
| `train --dim`、`--heads`、`--layers` | 128、4、2 |
| `train --epochs`、`--batch-size`、`--lr` | 20、32、0.001 |
| `train --top-k`、`--temperature` | 4、1.0 |
| `train --lambda-prior`、`--lambda-surd`、`--lambda-balance` | 0.1、0.05、0.001 |
| `train --negatives` | 每例 63 个均匀负例；`-1` 为全目录 |
| `train --no-surd` | 关闭统计先验 / 偏置与两项 SURD 监督（结构仍保留） |
| `--device`（仅 train / evaluate / explain / recommend） | 默认 `auto`：`cuda.is_available()` 则为 CUDA，否则 CPU（并 WARNING）；也可 `cpu`、`cuda`、`cuda:0`、`mps`；启动打印 `Using device=...` |
| `--eval-batch-size`、`--chunk` | 评估批大小 16、每块候选 128（可按显存下调） |
| `--threads` | PyTorch CPU 线程数，默认 2 |

训练使用 AdamW；每轮损失、专家派发与验证指标写入 `history.json`。验证集仅用于按 NDCG@10 选 `best.pt`（选模指标不变）；每轮与最终报告的**默认头条指标是 Acc@1**（同时打印 NDCG@1 与无截断 MRR）。训练结束后对最佳 checkpoint 再跑一次 **test**，并打印表格。结果目录：`eval_validation/`（最佳验证）与 `eval_test/`（最终测试），各含 `metrics.json` + `report.txt`。`train` / `evaluate` / `prepare` 默认显示 `tqdm` 进度（epochs、steps、评估批、SURD）。独立 `evaluate` 默认写出 `<checkpoint 所在目录>/eval_<split>/`（可用 `--out` 覆盖；`--output` 可再镜像一份 JSON）。JSON / 报告含 `all_targets` 与 `known_only`、`groups`、`primary_metrics`、`group_notes` / `group_labels`、`device` / `cache` / `cache_id` / `timestamp_utc` / 模型配置等复现字段。并列得分按目录行升序排名。单目标下一 POI：`Acc@K = Hit@K = Recall@K`，`MAP@K = MRR@K`；另报无截断 `MRR`、`mean_rank`、`median_rank`（仅对目录内目标）。K ∈ {1,5,10,20}。终端打印紧凑表格与分组摘要，避免整份 JSON 刷屏。Ctrl+C 在当前 step 边界退出（再按一次强制）。

**`groups` 含义（评估切片，非训练特征）：** 每个样本按真实下一站与上下文落入三组桶，每桶单独汇总 Acc@1 / NDCG@1 / MRR：

| 键前缀 | 依据 | 桶 |
| --- | --- | --- |
| `poi/` | 真实目标 POI 在训练 SURD 表中的 `target_count`（作为 next-POI 目标出现次数） | `outside_catalog`（不在训练目录）；`no_train_target`（在目录但 0 次）；`tail_1_2`（1–2，长尾）；`mid_3_9`（3–9）；`head_10_plus`（≥10，热门） |
| `history/` | 观测前缀长度 | `1`；`2_5`；`6_plus` |
| `last_category/` | 历史最后一签类别在训练集中的出现频次 | `0`；`1_9`；`10_plus` |

`report.txt` / 终端对上述键附带可读标签（如 `poi/head — ≥10 train targets (popular)`）；JSON 仍保留稳定短键，并写入 `group_labels` / `group_notes`。

---

## 6. 离线 SURD 与在线模型（简要）

与设计文档一致、当前实现已落地：

1. 四源历史摘要（P/C 哈希；T/L 数值）→ 每源标准化 + K-means 状态。
2. 单一联合频数表边缘化 → 26 维（4 Unique + 11 Redundancy + 11 Synergy）；15 个 U/S 专家在线执行。
3. 折外估计：默认最早 20% 训练样本暖启动，其后 3 个时间块；验证 / 测试 / 新轨迹用全训练 `final` 表。
4. 在线：四源 Transformer 编码历史；Router 读候选 + 上下文 + 置信度缩放的 26 维先验；Top-K 稀疏专家；`base + 门控 MoE 残差`。

缓存 `manifest.json` 记录字段约定、目录、分折与文件摘要。离线摘要与标准化矩阵在主存中；较大城市峰值内存随训练前缀数增长。

---

## 7. 测试与东京连通性参考

```bash
python -m unittest discover -s tests -v
python -m tsmoe.surd
```

自测覆盖解析分布、折外边界、稀疏派发、零置信度、候选分块、目录外目标与完整 CLI 流程。

**东京烟雾结果**（详见 [`examples/tokyo_smoke.json`](examples/tokyo_smoke.json)）：训练 / 验证 / 测试全部前缀样本 5,855 / 808 / 1,694；训练目录 2,494 个 POI。连通性命令示例（原记录在 CPU；GPU 服务器把 `--device cpu` 改成 `cuda` 或省略以用默认 `auto`）：

```bash
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model \
  --epochs 1 --dim 16 --heads 2 --layers 1 --negatives 15 \
  --batch-size 64 --eval-batch-size 32 --chunk 256 --threads 3 --device cuda
```

该设置下测试集 1,453 / 1,694 目标在目录内；全体目标 Recall@10 ≈ 0.1582，NDCG@10 ≈ 0.0920（单轮小模型，非完整训练指标）。

---

## 8. 文件索引

| 路径 | 内容 |
| --- | --- |
| `environment.yml` | Conda 环境骨架（Python 3.12 + pip + NumPy + tqdm；torch 用 pip cu124 安装） |
| `requirements.txt` | pip 版本约束（`numpy`、`tqdm`、`torch`；CUDA 轮子见上文索引 URL） |
| `examples/download_city.py` | 从 Hugging Face 拉取某城市三个 `*_checkins_*.csv` |
| `examples/make_demo.py` | 合成 demo CSV（无需下载） |
| `scripts/verify_city_checkins.py` | 核对 prepare 所需文件名与列（不 import torch） |
| `tsmoe/data.py` | CSV 读取、类别对齐、时间与地理特征、窗口 |
| `tsmoe/quantizer.py` | 四源摘要、标准化、K-means |
| `tsmoe/surd.py` | 联合频数表、26 项 SURD、冻结先验 |
| `tsmoe/prepare.py`、`tsmoe/inspect.py` | 折外估计、缓存、逐样本追踪 |
| `tsmoe/model.py` | Transformer、稀疏专家、Router、损失 |
| `tsmoe/engine.py`、`tsmoe/__main__.py` | 训练、评估、推荐、解释与 CLI |

数据格式与来源：[Massive-STEPS](https://github.com/CRUISEResearchGroup/Massive-STEPS) · [next-POI 划分脚本](https://github.com/CRUISEResearchGroup/Massive-STEPS/blob/main/src/create_next_poi_dataset.py) · [Tokyo 数据集](https://huggingface.co/datasets/CRUISEResearchGroup/Massive-STEPS-Tokyo)（Foursquare 风格签到字段，见设计文档 §2.1）。

---

## 9. 当前局限（与代码一致）

- **未打包安装**：无 `pip install -e .`；必须在仓库根目录用 `python -m tsmoe`。
- **无独立配置文件**：超参均通过 CLI 传入；无 YAML/JSON 训练配置入口。
- **无服务化推理 API**：推理入口是 `recommend` / `explain` CLI，不是 HTTP 服务。
- **默认 pip 索引为 cu124**：更旧驱动需改用 cu121 / cu118 索引重装 torch；本仓库不自动探测本机 CUDA。
- **无多进程 / 多卡数据并行开关**：`--device cuda:0` 等可选单卡；没有 `--gpus`、DDP 或 DataParallel 专用 CLI。
- **大数据内存**：大型城市 prepare 的峰值内存随训练前缀增长；本仓库未提供分片 / 外存统计路径。
