# TS-MoE · Massive-STEPS 下一 POI 推荐

根据所附《TS-MoE 基础版》实现的 PyTorch 项目。从 Massive-STEPS 城市签到 CSV 出发，构造同一 trail 内的下一跳预测样本，用训练数据估计逐 POI 的 26 维 SURD，再训练候选感知的 15 专家 Transformer。程序支持完整目录评估、输入新轨迹进行推荐，以及从联合计数追踪到单个候选的最终得分。

## 在东京数据上运行

在解压后的 `TS-MoE/` 目录执行。使用 Python 3.11 或 3.12 和 PyTorch 2.4+（2.x）。已在 Python 3.12、PyTorch 2.8.0 CPU 上完成端到端验证。有 CUDA 版 PyTorch 时会自动使用 CUDA，也可显式传入 `--device cpu`。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python examples/download_city.py --city tokyo --data-root data
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model
python -m tsmoe evaluate --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt --split test --output runs/tokyo/test.json
```

已经有官方仓库的数据时，可以跳过下载，把 `--data-root` 指向其 `data/` 或 `data/tokyo/`。下载工具支持官方城市文件名前缀，例如 `beijing`、`new_york`、`sao_paulo`。要复现特定数据版本，可传入 `--revision` 和数据集 commit SHA。

```text
data/tokyo/
  tokyo_checkins_train.csv
  tokyo_checkins_validation.csv
  tokyo_checkins_test.csv
```

| CSV 字段 | 程序中的用途 |
| --- | --- |
| `trail_id`、`user_id` | 检查轨迹归属、组织事件、隔离划分，以及定位案例 |
| `venue_id` | 历史 P、下一 POI 标签与候选目录中的原始整数 ID，保留有效值 0 |
| `venue_category_id_code`、`venue_category_id` | 历史 C。优先使用训练集中一致的 code，通过稳定类别 ID 对齐重新编号的文件。支持只含其中一个字段的 CSV |
| `timestamp` | 轨迹排序、历史周期、相邻间隔和时间折边界 |
| `latitude`、`longitude` | 历史 L 的球面坐标、位移、方向及有效位 |

采用官方预处理代码生成的**逐签到 CSV**。一个输入行对应一次访问。带时区的时间戳转成 UTC，无时区的时间戳按 UTC 解释。等时事件按原文件行序稳定排序。以 `(user_id, trail_id, timestamp, venue_id)` 识别重复事件，保留第一条。若同一场地在同一时间戳下实际存在两次独立签到，这一口径会将其合并。不可解析的时间戳整行隔离，缺失的类别或坐标保留事件并编码。历史裁剪后，第一步的时间间隔和位移重新置为无前序。程序在 `manifest.json` 中记录数据审计结果。

### 选择预测目标

默认 `--train-targets all --eval-targets all`，符合附件。一条 `A → B → C → D` 产生三个样本，目标分别为 B、C、D。每个样本的输入止于前一个事件。训练、验证、测试沿用官方 CSV 的轨迹成员关系。

| 想运行的协议 | `prepare` 参数 |
| --- | --- |
| 附件主模型，每个前缀均评分 | `--train-targets all --eval-targets all` |
| 训练和评估均使用每条轨迹的末站目标 | `--train-targets last --eval-targets last` |
| 用全部前缀训练，仅评估每条轨迹的末站 | `--train-targets all --eval-targets last` |

官方生成 next-POI 提示的脚本取轨迹最后一条访问作为目标。报告对比结果时应同时列出前缀协议、目录覆盖率及评估分母。官方发布的三个集合按轨迹划分，跨集合的日期可以交叠。下面的严格时间限制适用于**训练内部的 SURD 折外估计**。

## 离线 SURD 如何产生

1. **构造四种历史摘要。** P/C 分别使用独立 salt 的 SHAKE-256 符号哈希。T 使用 UTC 小时和星期的周期编码、小时差与有效标志。L 使用球面三维坐标、公里距离、方向和有效标志。每源拼接最后一步、位置衰减均值、普通均值及窗口长度比例，默认维数依次为 193、193、19、25。
2. **量化为四个状态。** 每折只在合法参考历史上拟合四组标准化器与 K-means，然后在同一批历史上做最近中心分配。默认每源请求 4 簇，可传 `--clusters 2` 或 `--clusters 8`，实际簇数受不同摘要数量和中心占用情况限制。
3. **从同一张表分解。** 将 `(Y,Q_P,Q_C,Q_T,Q_L)` 计入一个稀疏联合频数表，对它边缘化取得 15 个子集，计算每个具体 POI 的事件级 specific information，分配为 4 Unique、11 Redundancy 和 11 Synergy。每折检查非负性、子集单调性、单源重建、总量守恒及两种条件熵计算的一致性。
4. **保存三套 26 维结果。** `raw_surd` 以 bit 为单位，`normalized_surd` 按逐 POI 总信息量归一化，`routing_prior` 结合目标支持度与全局结构做收缩。另存样本数、有效标志、置信权重、15 个 specific information、联合频数及量化器文件摘要。
5. **防止折外泄漏。** 默认将最早 20% 的训练预测样本作为暖启动评分块，其后分成 3 个时间块。对每个块，参考目标时间严格早于该块最早的预测时刻，即历史末事件时间。相同时间戳的参考目标也会排除。空参考表的置信度为零。验证、测试和新轨迹推断都使用全训练集生成的 `final` 表。

用下面的命令查看东京数据中候选 `venue_id=3` 的 26 项分解。指定 `--sample-index` 后，还能看到该训练前缀对应的四个簇编号、每个子集对应的联合计数、边缘计数及点互信息。行级的 specific information 是对该候选的状态分布取条件平均。

```bash
python -m tsmoe inspect --cache runs/tokyo/cache --table final --poi-id 3 --sample-index 0 --output runs/tokyo/offline_trace.json
```

## 在线训练与评估

四个独立 Transformer 分别编码 P/C/T/L，只对当前 batch 的历史编码一次。每个候选用自己的 POI embedding 查询四路历史，并读取自己的 SURD 行。Router 同时读取候选、当前上下文和按置信度缩放的完整 26 维先验，再向对应的 U/S logits 添加显式偏置。4 个单源专家和 11 个组合专家按候选做 Top-K 派发。组合专家只接收对应来源的乘法交互与残差项。最终分数为 Base 分数加上门控后的 MoE 残差。Redundancy 参与路由、门控和监督。

| 参数 | 默认值与作用 |
| --- | --- |
| `prepare --max-length`、`--hash-dim`、`--tau-pos` | 50 次签到、64 维 ID 哈希、位置衰减常数 10 |
| `prepare --clusters`、`--folds`、`--warmup` | 每源 4 簇、暖启动后 3 个评分块、暖启动比例 0.2 |
| `train --dim`、`--heads`、`--layers` | 隐藏维 128、4 个注意力头、每源 2 层 |
| `train --top-k`、`--temperature` | 15 个专家中选 4 个，温度 1.0，同分时使用固定专家顺序 |
| `train --lambda-prior`、`--lambda-surd`、`--lambda-balance` | 0.1、0.05、0.001，分别控制显式先验、门控与路由监督、软负载约束 |
| `train --negatives` | 默认每例均匀无放回抽 63 个负例，在正例与这些负例上做交叉熵。样本数足以覆盖目录时自动用全目录，`-1` 强制全目录。采样训练目标不含重要性校正 |
| `--eval-batch-size`、`--chunk` | 评估时默认每批 16 条历史、每块 128 个候选，可按显存调整 |
| `train --no-surd` | 保留同一组 U/S 专家和门控结构，把先验输入、偏置和两项统计监督的有效置信度置零，用于普通 MoE 对照 |

训练使用 AdamW，默认 20 轮，按验证集**所有目标**的 NDCG@10 保存 `best.pt`。每轮三个损失项、15 个专家的实际派发次数和验证指标在 `history.json`。可通过 `--lambda-prior 0` 或 `--lambda-surd 0` 单独关闭相应机制。

评估逐块精确打分训练集中出现过的所有 POI。输出 Recall、NDCG、MRR 的 @1、@5、@10，以及两种分母下的结果。`all_targets` 将目录外目标计作未命中；`known_only` 只统计目录内目标，同时报告覆盖率。得分并列时按目录行升序排名。JSON 还按目标训练频次、历史长度和最后历史类别的训练频次分组。一次预测有一个正确目标，所以此处 Recall@K 等于 Hit@K。

### 推荐新轨迹与解释候选

`--history` 接受与官方签到 CSV 字段相同的文件。每条 trail 可只含一条已观测签到，程序用截至该轨迹末次签到的历史返回下一站推荐。输出使用原始 `venue_id`。

```bash
python -m tsmoe recommend --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt --history my_history.csv --top-n 10 --output recommendations.json
python -m tsmoe explain --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt --split test --sample-index 0 --output case.json
```

`explain` 默认解释真实目标。若目标在训练目录外，选择最高分候选并同时记录真实目标。可用 `--poi-id` 指定任一目录内候选。案例包括三种先验、置信度、learned logits、显式 prior bias、dense 路由概率、选中专家、Top-K 权重、门值、逐专家有符号分数贡献及全目录名次。`base_score + moe_bias_contribution + sum(expert_score_contributions)` 可还原最终得分。

## 验证与复现

```bash
python -m unittest discover -s tests -v
python -m tsmoe.surd
```

自测覆盖全复制、单源独有、二源与四源 XOR、独立噪声、独立擦除、确定性类别/位置七种解析分布，以及折外边界、窗口重置、来源隔离、稀疏派发梯度、零置信度、候选分块、目录外目标和完整命令行流程。

**已执行的东京数据验证。** 训练、验证、测试各有 5,855、808、1,694 个全部前缀样本，训练目录含 2,494 个 POI。以 `--epochs 1 --dim 16 --heads 2 --layers 1 --negatives 15 --batch-size 64 --eval-batch-size 32 --chunk 256 --threads 3 --device cpu` 做连通性验证。测试集 1,453 / 1,694 个目标在训练目录内，以全体目标计的 Recall@10 为 0.1582，NDCG@10 为 0.0920。这些数值对应单轮小模型。原始文件摘要、运行参数和完整指标见 `examples/tokyo_smoke.json`。

无需下载真实数据即可运行一个小例子。

```bash
python examples/make_demo.py --data-root data
python -m tsmoe prepare --data-root data --city demo --out runs/demo/cache
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model --epochs 1 --dim 16 --heads 2 --layers 1
python -m tsmoe evaluate --cache runs/demo/cache --checkpoint runs/demo/model/best.pt
```

## 文件索引

| 文件 | 内容 |
| --- | --- |
| `tsmoe/data.py` | CSV 读取、类别对齐、事件时间与地理特征、共享窗口 |
| `tsmoe/quantizer.py` | 四源固定摘要、标准化、K-means、保存与加载 |
| `tsmoe/surd.py` | 单一联合频数表、15 个边缘分布、26 项 SURD、冻结先验查表与解析自测 |
| `tsmoe/prepare.py`、`tsmoe/inspect.py` | 折外估计、缓存校验和逐样本量化追踪 |
| `tsmoe/model.py` | 四源 Transformer、候选注意力、稀疏专家、Router、残差门与损失 |
| `tsmoe/engine.py`、`tsmoe/__main__.py` | 命令行、训练、精确排名、推荐与案例解释 |

缓存的 `manifest.json` 记录字段、UTC 和单位约定、目录、分折和文件摘要。改变簇数、窗口或目标选取方式时，为新配置指定新的 `--out`，并用配套缓存训练和评估。离线摘要和标准化矩阵保存在主存中，较大城市的峰值内存随训练前缀数增长。

数据格式与发布来源：[Massive-STEPS 官方代码](https://github.com/CRUISEResearchGroup/Massive-STEPS) · [官方的目标生成与 CSV 划分脚本](https://github.com/CRUISEResearchGroup/Massive-STEPS/blob/main/src/create_next_poi_dataset.py) · [Tokyo 数据与字段](https://huggingface.co/datasets/CRUISEResearchGroup/Massive-STEPS-Tokyo)。量化器和离线 SURD 核心沿用附件附录 C/D 的实现及分量顺序。
