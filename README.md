# TS-MoE · Massive-STEPS 下一 POI 推荐

Target-conditioned SURD-guided Mixture-of-Experts（TS-MoE）的 PyTorch 实现：从 Massive-STEPS 城市签到 CSV 构造同一 trail 内的下一跳预测样本，用训练数据估计逐 POI 的 26 维 SURD，再训练候选感知的 15 专家 Transformer。支持全目录评估、新轨迹推荐，以及从联合计数到单个候选得分的追踪。

算法设计见仓库根目录 [`TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md`](TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md)。本 README 只描述**当前代码**如何安装与运行。

**依赖只有 Python 标准库 + PyTorch**（`torch>=2.4,<3`）。没有 `setup.py` / `pyproject.toml`：请在仓库根目录执行 `python -m tsmoe ...`，不要 `pip install` 本包。

面向 **NVIDIA GPU 服务器**：`environment.yml` 安装带 CUDA 的 PyTorch。`train` / `evaluate` / `explain` / `recommend` 的 `--device` 默认为 `auto`（检测到 CUDA 时用 GPU）。东京连通性烟雾曾在 CPU 上跑通；GPU 上用同一 CLI，换 `--device` 即可。

---

## 1. 用 Conda 创建 CUDA 环境（推荐）

需要已安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda，以及可用的 NVIDIA 驱动（`nvidia-smi` 正常）。仓库根目录的 `environment.yml` 安装 **CUDA 12.4** 构建的 PyTorch（`pytorch-cuda=12.4`；无需本机再装完整 CUDA Toolkit）。

```bash
cd TS_MOE   # 仓库根目录

conda env create -f environment.yml
conda activate tsmoe

# 确认：应打印 True，且 device 为 cuda:0（或同类）
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
```

之后若环境或依赖有变动：

```bash
conda env update -f environment.yml --prune
```

**CUDA 版本假设：** 默认按常见 GPU 服务器选用 `pytorch-cuda=12.4`。若驱动偏旧，编辑 `environment.yml` 把 `pytorch-cuda=12.4` 改成 `12.1` 或 `11.8` 后重新 `conda env create` / `update`；对照 [PyTorch 安装页](https://pytorch.org/get-started/locally/) 选择匹配组合。

### 在 GPU 上跑（与 CLI 一致）

仅 `train`、`evaluate`、`explain`、`recommend` 接受 `--device`（`prepare` / `inspect` 为 CPU 侧数据与 SURD 构建，无该参数）。合法取值与 `__main__.py` 一致：`auto`（默认）、`cpu`、`cuda`、`cuda:0`、`mps`。

```bash
conda activate tsmoe

# 默认 --device auto：有 CUDA 则用 GPU
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model
python -m tsmoe evaluate --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt --split test

# 显式指定 GPU（多卡时常用 cuda:0）
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model --device cuda
python -m tsmoe evaluate --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt --device cuda:0
```

显存紧张时可减小 `--eval-batch-size` / `--chunk`（评估）或 `--batch-size` / `--negatives`（训练）；这些标志已存在于 CLI。

### CPU / 无 GPU 回退

无 GPU 或只需 CPU 时：仍可用同一环境，对训练与评估等命令传 `--device cpu`。若希望 conda 直接装 CPU 版 PyTorch，将 `environment.yml` 中的 `pytorch-cuda=12.4` 换成 `cpuonly`，并去掉 `nvidia` channel，再重建环境。

### 备选：venv + pip（CUDA）

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
# 与 environment.yml 的 CUDA 12.4 对应；其它版本见 pytorch.org
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

`requirements.txt` 仅声明版本范围 `torch>=2.4,<3`，不锁定 CUDA/CPU 轮子；GPU 服务器请用上面的 cu124 索引（或 conda）。

---

## 2. 目录约定

所有命令均在**仓库根目录**执行。典型布局：

```text
TS_MOE/
  environment.yml
  requirements.txt
  tsmoe/                 # 包；入口 python -m tsmoe
  examples/              # 下载城市数据、生成 demo CSV
  tests/
  data/<city>/           # 下载或自备的签到 CSV（可按城市分子目录）
  runs/<city>/cache/     # prepare 产出的样本与 SURD 缓存
  runs/<city>/model/     # train 产出的 best.pt、history.json
```

`prepare --data-root` 会先找 `<data-root>/<city>_checkins_train.csv`；找不到则再试 `<data-root>/<city>/<city>_checkins_train.csv`。因此下载脚本写到 `data/tokyo/` 时，可直接用 `--data-root data --city tokyo`。

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

采用官方预处理生成的**逐签到 CSV**。带时区的时间戳转成 UTC，无时区按 UTC 解释。等时事件按原文件行序稳定排序。以 `(user_id, trail_id, timestamp, venue_id)` 去重（保留第一条）。不可解析时间戳整行隔离；缺失类别或坐标仍保留事件并编码。审计写入缓存的 `manifest.json`。

---

## 3. 最快跑通：合成 demo（无需下载）

```bash
conda activate tsmoe

python examples/make_demo.py --data-root data
python -m tsmoe prepare --data-root data --city demo --out runs/demo/cache
# train / evaluate 默认 --device auto（GPU 可用则走 CUDA）
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model --epochs 1 --dim 16 --heads 2 --layers 1
python -m tsmoe evaluate --cache runs/demo/cache --checkpoint runs/demo/model/best.pt
# 等价显式写法：在上述两条命令末尾加 --device cuda 或 --device cuda:0
```

---

## 4. 东京（或其它官方城市）全流程

```bash
conda activate tsmoe

python examples/download_city.py --city tokyo --data-root data
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model --device cuda
python -m tsmoe evaluate --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt \
  --split test --output runs/tokyo/test.json --device cuda
```

省略 `--device` 时默认为 `auto`，在已正确安装 CUDA PyTorch 的 GPU 服务器上效果与 `--device cuda` 相同。

已有官方仓库数据时可跳过下载，把 `--data-root` 指到其 `data/` 或 `data/tokyo/`。`download_city.py` 支持的城市前缀包括：`bandung`、`beijing`、`istanbul`、`jakarta`、`kuwait_city`、`melbourne`、`moscow`、`new_york`、`palembang`、`petaling_jaya`、`sao_paulo`、`shanghai`、`sydney`、`tangerang`、`tokyo`。复现特定数据版本可传 `--revision <commit-or-branch>`。

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
| `train` | 用缓存训练；按验证集 NDCG@10 保存 `best.pt` |
| `evaluate` | 对训练目录内全部 POI 精确打分，输出 Recall / NDCG / MRR @1/5/10 |
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
| `--device`（仅 train / evaluate / explain / recommend） | 默认 `auto`：`cuda.is_available()` 则为 CUDA，否则 CPU；也可 `cpu`、`cuda`、`cuda:0`、`mps` |
| `--eval-batch-size`、`--chunk` | 评估批大小 16、每块候选 128（可按显存下调） |
| `--threads` | PyTorch CPU 线程数，默认 2 |

训练使用 AdamW；每轮损失、专家派发与验证指标写入 `history.json`。评估对训练出现过的全部 POI 分块打分；JSON 含 `all_targets` 与 `known_only` 两种分母及覆盖率。并列得分按目录行升序排名。一次预测一个正确目标，故此处 Recall@K = Hit@K。

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
| `environment.yml` | Conda 环境（Python 3.12 + CUDA 12.4 PyTorch） |
| `requirements.txt` | pip 版本约束（仅 `torch`；CUDA 轮子见上文索引 URL） |
| `tsmoe/data.py` | CSV 读取、类别对齐、时间与地理特征、窗口 |
| `tsmoe/quantizer.py` | 四源摘要、标准化、K-means |
| `tsmoe/surd.py` | 联合频数表、26 项 SURD、冻结先验 |
| `tsmoe/prepare.py`、`tsmoe/inspect.py` | 折外估计、缓存、逐样本追踪 |
| `tsmoe/model.py` | Transformer、稀疏专家、Router、损失 |
| `tsmoe/engine.py`、`tsmoe/__main__.py` | 训练、评估、推荐、解释与 CLI |

数据格式与来源：[Massive-STEPS](https://github.com/CRUISEResearchGroup/Massive-STEPS) · [next-POI 划分脚本](https://github.com/CRUISEResearchGroup/Massive-STEPS/blob/main/src/create_next_poi_dataset.py) · [Tokyo 数据集](https://huggingface.co/datasets/CRUISEResearchGroup/Massive-STEPS-Tokyo)。

---

## 9. 当前局限（与代码一致）

- **未打包安装**：无 `pip install -e .`；必须在仓库根目录用 `python -m tsmoe`。
- **无独立配置文件**：超参均通过 CLI 传入；无 YAML/JSON 训练配置入口。
- **无服务化推理 API**：推理入口是 `recommend` / `explain` CLI，不是 HTTP 服务。
- **`environment.yml` 锁定 `pytorch-cuda=12.4`**：更旧驱动需改 pin（12.1 / 11.8）；本仓库不自动探测本机 CUDA。
- **无多进程 / 多卡数据并行开关**：`--device cuda:0` 等可选单卡；没有 `--gpus`、DDP 或 DataParallel 专用 CLI。
- **大数据内存**：大型城市 prepare 的峰值内存随训练前缀增长；本仓库未提供分片 / 外存统计路径。
