# TS-MoE 基础版：基于 POI、Category、Time、Location 序列的逐 POI SURD 引导混合专家模型

> 修订日期：2026-11-18  
> 本版输入：POI（P）、Category（C）、Time（T）、Location（L）四条逐 check-in 对齐的历史序列；User 不作为输入。  
> 保留约束：仅保留 Unique 与 Synergy Experts；Redundancy 作为辅助信息；离线估计仅使用路径 A“同一个联合概率表，再边缘化”；完整保存 26 维 SURD 分解及先验。  
> 记号约定：分量前缀 U 表示 Unique，不表示 User；user_id、trail_id 仅用于分组、切分、去重和评估，不参与离线来源编码或在线预测。  
> 本次修订：简要补充 trail 如何拆成多个预测样本、193/19/25 维摘要的来源，以及簇数的选择流程。保留 Quantizer 与联合计数示例、无图基础方案、完整 26 维 SURD 和 15 个 U/S Experts；教学示例不代表真实数据训练结果。  
> 文档中的数学公式均采用独立的双美元块，以便 Markdown 数学预览。

## 1. 本版设计与研究目标

本文提出 Target-conditioned SURD-guided Mixture-of-Experts Transformer，简称 TS-MoE，预测用户下一次访问的**具体 POI**。

核心任务是：先从训练历史中，逐 POI 识别四类来源所提供的独有、冗余与协同预测信息；再将完整信息结构作为可学习 Router 的先验，让不同候选 POI、不同历史轨迹上下文激活不同的交互 Experts。

本版固定采用以下设计。

| 项目 | 本版定义 |
| --- | --- |
| 信息来源 | POI、Category、Time、Location，固定来源顺序为 P、C、T、L |
| 序列对齐 | 四个来源来自同一 trail 的同一批历史 check-in，长度、顺序、截断位置和 padding mask 一致 |
| 标识字段 | user_id、trail_id 只用于分组、切分、去重和评估 |
| 离线表示 | P/C 独立固定哈希，T/L 固定数值特征；各源分别做最后一步、位置衰减均值、均值和长度摘要 |
| 拟合与计数 | 在合法 D_ref 的历史摘要上拟合标准化和 K-means，再用同一 D_ref 的状态与标签计数；外层评分数据留出 |
| 个性化依据 | 当前可见的四条历史序列；不依赖静态用户身份参数 |
| 预测目标 | 下一次访问的具体 venue_id，保留离散 POI 身份，不将 POI ID 按数值区间合并 |
| 离线概率估计 | 一个五维离散联合频数表，15 个子集分布全部由它边缘化得到 |
| 分解理论 | 原始 SURD 的目标事件级 specific information 排序分配规则 |
| 完整分解 | 4 个 Unique、11 个 Redundancy、11 个 Synergy，共 26 项 |
| 实际 Experts | 4 个 Unique Experts 与 11 个 Synergy Experts，共 15 个 |
| Redundancy 的作用 | 完整输入 Router，辅助 Base/MoE 残差门和路由监督；不设置 Redundancy Expert |
| 缓存 | 原始 26 维分解、完整归一化 26 维分解、稳健化后的 26 维路由先验 |
| 在线计算 | 查询候选 POI 的 26 维先验，再结合当前上下文产生 15 个 Expert logits |
| 稀疏执行 | 默认 Top-4，仅执行被选中的 U/S Experts |
| 预测结构 | Base 推荐分支加上可学习门控制的 MoE 残差 |
| 训练目标 | 推荐交叉熵、SURD 辅助损失、与先验一致的软负载约束 |

维度必须区分：

$$
D_{\mathrm{SURD}}=4+11+11=26,
\qquad
E_{\mathrm{MoE}}=4+11=15.
$$

**26 是统计先验的维度，15 是实际 Experts 的数量。**二者不需要相等。本版没有 21 维中间表，也不把高阶冗余分摊给两两 Experts。

本次修改是用 Category 序列替换 User 来源，同时把来源顺序固定为 P、C、T、L。仍然是四来源问题，因此 26/15 两个维度保持不变；但统计变量和列语义已经变化，旧版 User/POI/Time/Location 的量化器、联合表、SURD 缓存和相关模型参数不能直接按列改名后复用。

原始 SURD 定义与排序规则来自参考文献 [1]、[2]。将其用于推荐路由、设置残差门、设计辅助损失，均属于本文的模型设计，不能表述为 SURD 定理必然要求的网络结构。


## 2. 原始字段、四条对齐序列与预测样本

### 2.1 数据字段到模型输入的映射

本版直接对应给定的 check-in 数据结构。每条原始记录是一场访问事件；一条 trail 包含多个这样的事件。字段职责如下。

| 原始字段 | 本版用途 | 编码或处理原则 |
| --- | --- | --- |
| `trail_id` | 轨迹分组、边界检查、去重与切分 | 不作为离线来源特征或在线预测输入 |
| `user_id` | 用户核验、轨迹分组、切分和评估 | 不设置用户 embedding、用户画像来源或用户打分偏置 |
| `venue_id` | POI 历史序列 P；下一事件的该字段作为监督标签 | 离线固定哈希，在线独立 embedding；ID 数值大小不具有距离意义 |
| `venue_category_id_code` | Category 历史序列 C 的首选类别键 | 离线固定哈希，在线独立类别 embedding；不按连续数值处理 |
| `venue_category_id` | 核验、建立稳定的类别映射 | 与类别 code 表示同一来源，不额外增加一个 SURD 来源 |
| `venue_category` | 类别名称、映射核验及结果解释 | 默认不引入文本 Encoder，不与类别编号重复计为独立来源 |
| `timestamp` | Time 历史序列 T、trail 内事件排序 | 从历史时间生成周期特征与历史时间间隔 |
| `latitude`、`longitude` | Location 历史序列 L | 使用同一次历史 check-in 的 POI 经纬度；可派生历史距离、方向 |
| `name`、`address` | 数据检查及结果展示 | 第一版不作为额外模型输入 |
| `venue_city`、`venue_country` | 数据检查、空间分组及结果分析 | 第一版不单独进入模型；若后续纳入，应明确归入哪一来源并重建统计 |
| `venue_city_latitude`、`venue_city_longitude` | 行政区域位置元数据 | 不默认为 POI 精确坐标；若用作缺失位置回退，必须标记其粗粒度性质 |

类别 code 仅在映射一致时直接使用。如果它在不同文件、城市或数据划分中被重新编号，应以稳定的 `venue_category_id` 为依据，在相应训练协议内重新建立一致映射。缺失类别使用 UNK；PAD 与 UNK 必须区分。在线 embedding 词表外的类别也映射到其 UNK；离线基础哈希对已知有效但参考集中未出现的非负类别 ID 仍能按固定规则编码，具体边界见第 4 节。

### 2.2 先组织 trail，再同时构造四条序列

默认按 `(user_id, trail_id)` 分组，并检查一个 trail 是否对应唯一用户。若数据规范保证 trail_id 全局唯一，可以单独用它分组；不能仅按 user_id 把所有独立 trail 无声明地串成一条轨迹。

每个组内按 timestamp 排序。相同时间戳使用已记录的原始事件顺序或稳定事件索引打破并列，并保留该规则；没有可靠顺序的并列事件应按数据协议标记或处理，不能用下一 POI 标签推断先后。

四条序列必须从同一个排序后的事件列表同时生成，使用相同的裁剪索引和 padding mask。Category 或坐标缺失时使用各自的未知/缺失标记保留该事件；若因数据质量删除事件，应在组装四条序列之前共同删掉整条记录。不得分别 dropna 后拼接四个长度相同但事件已经错位的数组。

时间预处理也需要固定：若时间戳包含时区，先正确解析；小时和星期使用有可靠依据的本地时区，或统一使用 UTC 并记录该选择。不能只凭国家代码猜测唯一时区。时间间隔由相邻历史时间计算，窗口第一个事件使用“无前序间隔”标记。第一版将缺失或不可解析 timestamp 的记录在组装四条序列之前统一隔离，不用未来时间补齐；若业务必须保留且存在可信事件顺序，则要增加时间缺失标记、明确排序及间隔失效规则，并重建对应统计。

### 2.3 一个样本是历史前缀与下一个 POI

**一个样本是同一条 trail 内的“已观测历史前缀 → 紧接着的目标 POI”。一条 trail 通常生成多个样本。** 例如，按时间排序的 trail 为 `[A,B,C,D,E]`，在未超过窗口上限时：

| 样本 | P 历史 | 目标 Y |
| --- | --- | --- |
| 1 | `[A]` | B |
| 2 | `[A,B]` | C |
| 3 | `[A,B,C]` | D |
| 4 | `[A,B,C,D]` | E |

长度为 M 的 trail，在事件均合法且允许一步历史时，可提供 M−1 个候选预测样本；第一个事件没有历史，不生成当前版本的样本。两条长度为 5 和 4 的 trail 共提供 7 个候选样本。**n 是经过参考范围、去重和切分筛选后的预测样本数，不是 trail 数。**

对一条按时间排序的 trail，在第 t 个事件之后预测第 t+1 个事件。历史窗口最多保留最近 N_max 次访问，默认 N_max=50；P/C/T/L 使用相同的历史位置与截断范围，目标事件及其未来属性不进入历史。实际长度为：

$$
N_i=\min(t,N_{\max}).
$$

例如，预测第 80 次访问时，只使用同一 trail 的第 30～79 次访问，共 50 次；不会跨独立 trail 补足历史。

以下下标 j 表示裁剪后窗口内的位置。四个原始来源为：

$$
\begin{aligned}
\mathbf P_i&=(p_{i,1},\ldots,p_{i,N_i}),\\
\mathbf C_i&=(c_{i,1},\ldots,c_{i,N_i}),\\
\mathbf T_i&=(\tau_{i,1},\ldots,\tau_{i,N_i}),\\
\mathbf L_i&=(\ell_{i,1},\ldots,\ell_{i,N_i}),\\
\ell_{i,j}&=(\mathrm{latitude}_{i,j},\mathrm{longitude}_{i,j}).
\end{aligned}
$$

同一个 j 上的 POI、类别、时间、位置必须来自同一 check-in。四个来源经过各自特征化或 embedding 后记为：

$$
\mathcal X_i=(X_{P,i},X_{C,i},X_{T,i},X_{L,i}),
\qquad
X_{m,i}\in\mathbb R^{N_i\times d_m},
\qquad
m\in\mathcal M=\{P,C,T,L\}.
$$

训练目标仍为下一次访问的具体 POI：

$$
Y_i=p_{\mathrm{trail}(i),t+1},
\qquad
Y_i\in\mathcal V_P.
$$

推荐模型的输入条件明确写为：

$$
s_\theta(y\mid\mathbf P_i,\mathbf C_i,\mathbf T_i,\mathbf L_i).
$$

user_id 和 trail_id 不出现在这个条件中。对当前预测样本，它们不能以重复常量序列、额外 token、用户向量、用户特定 Router、用户打分偏置或 SURD 的第五来源进入模型。第 4 节的离线基础特征同样只读取 P/C/T/L 历史，不读取 user/trail 身份。相同四条历史序列及相同候选，在固定模型、确定性推理和相同配置下应得到相同分数，不因 user_id 的数值不同而变化。

这里没有取消基于行为的个性化：不同历史前缀仍然可以产生不同的兴趣表示和候选排序；只是模型不再通过静态用户身份直接区分样本。

历史长度与统计样本数是不同概念：本文件用大写 N 表示序列长度，用小写 n 表示离线参考集中的预测样本数。每个真实目标 check-in 在一张离线统计表中只计为一个预测事件。若推荐训练对同一目标构造多个增强窗口，离线统计必须去重，或明确使用总权重为 1 的加权方案。同一次访问可以先作为某个样本的目标，再成为后续样本的已观测历史；同一 POI 在不同时间的真实访问则是不同目标事件，可以分别计数。

### 2.4 一个对齐样本的例子

下面是说明数据布局的虚构 trail 前缀，不表示真实访问记录。

| 窗口位置 | P：venue_id | C：类别 code | T：历史 timestamp | L：历史经纬度 |
| --- | --- | --- | --- | --- |
| 1 | poi_17 | cat_3 | 已观测时刻 1 | 坐标 1 |
| 2 | poi_42 | cat_8 | 已观测时刻 2 | 坐标 2 |
| 3 | poi_19 | cat_3 | 已观测时刻 3 | 坐标 3 |

这个样本的四条序列长度均为 3，标签为第 4 次访问的 venue_id。真实第 4 次访问的 Category、Time 和 Location 不进入这三步历史，也不用于生成该样本的离线来源摘要。训练标签是 next-POI，不改成 next-Category。

### 2.5 历史信息与候选属性的边界

| 来源 | 默认允许使用的信息 | 来源隔离要求 |
| --- | --- | --- |
| POI | 预测时已经访问的 POI ID 序列 | P 摘要和 P Encoder 不读取当前样本的类别、时间、位置或用户画像 |
| Category | 与历史 POI 对齐的类别序列 | C 摘要和 C Encoder 不读取当前样本的 POI ID；类别映射版本固定 |
| Time | 历史小时、星期、历史间隔等 | 不读取未知的真实下一次访问时间 |
| Location | 历史经纬度、由历史坐标得到的位移与方向 | 不读取真实未来位置；默认不使用需要 Time 的速度特征 |

第一版只使用 trail 历史事件，不额外假设有实时 GPS 或已知下一次查询时间。若任务确实提供预测时已知的查询时间或当前位置，应作为明确的数据协议扩展，并同步修改来源定义和离线统计。

候选 POI 自身可查询的静态类别和坐标可以放入候选表示。它们对每一个候选都以相同规则查询，与把真实下一次访问的属性提前加入历史序列不同。静态元数据也应遵守实验规定的可用时间，不能只给正例补充来自未来记录的属性。

### 2.6 候选打分与训练推理一致性

对每一个候选独立计算分数：

$$
s_\theta(y\mid\mathcal X_i),
\qquad
y\in\mathcal C_i.
$$

候选可以来自完整词表，也可以来自独立召回模块。训练时正例和负例都必须使用自己的候选表示与自己的 SURD 行；同一训练样本的所有候选使用同一张折外统计表。

闭集实验的候选 POI 词表应由训练协议确定；未见候选、缺失类别与分布外历史需要明确处理。基础量化器将新历史分配到已有的最近中心，不临时创建新的簇。离线与在线均不需要用户词表或未知用户 embedding。零历史样本仍然缺少行为上下文。若没有新 POI 的可用表示或召回支持，应单独报告冷启动覆盖率，不能把 global prior 当作已解决整个冷启动预测问题。

## 3. 整体架构与离线、在线职责

离线部分构造固定统计先验：

$$
\mathcal D_{\mathrm{ref}}
\longrightarrow
(Q_P,Q_C,Q_T,Q_L,Y)
\longrightarrow
\widehat p(Q_P,Q_C,Q_T,Q_L,Y)
\longrightarrow
\{\iota_A(y)\}_{A\ne\varnothing}
\longrightarrow
\mathbf d_y^{\mathrm{raw}}\in\mathbb R_{\ge0}^{26}.
$$

在线部分读取完整先验：

$$
(\mathcal X_i,y,\bar{\mathbf r}_y)
\longrightarrow
\text{Candidate-aware representations}
\longrightarrow
\text{Router and residual gate}
\longrightarrow
\text{Top-}K\text{ U/S Experts}
\longrightarrow
s_\theta(y\mid\mathcal X_i).
$$

其中，离线基础特征/量化器与在线推荐 Encoder 职责不同：

- 离线 P/C 固定哈希与 T/L 数值特征无需预训练；对各来源独立摘要，在合法参考集历史上拟合标准化与 K-means。
- 特征配置、摘要规则、标准化与 codebook 固定后定义四个离散随机变量，不随推荐训练的 batch 更新。
- 在线 Encoder、Experts、Router、残差门通过推荐目标联合优化；在线的 P/C embedding 与离线固定哈希分别使用。
- 离线表不参与反向传播。
- 更新特征配置、摘要或 codebook 都会改变统计变量，必须重新量化、计数、分解并保存新版本。

本版的 prior 是 **POI-conditioned**：同一个候选 POI 的离线行固定。当前四条历史序列与候选之间的动态差异由在线 Router 与残差门表达；本版不宣称每条历史都重新获得一次严格统计 SURD。


## 4. Quantizer 与聚类：从四条历史得到四个离散变量

### 4.1 Quantizer 的作用：把一段来源历史变成一个可计数的状态

**本版需要聚类，而且聚类已经包含在 Quantizer 内部：`fit()` 用 K-means 拟合各来源的聚类中心，`transform()` 使用固定中心，把每条来源历史的摘要转换成一个整数簇编号。** 因此，“使用 Quantizer”与“使用聚类”并不是两种互斥方案，也不需要在 Quantizer 后面再做一遍聚类。

Quantizer 在这里译为“量化器”。它的作用是把复杂的历史摘要归并到有限个状态，使第 5 节能够统计“某种历史状态与某个下一 POI 同时出现了多少次”。本版选择 K-means 向量量化：中心构成 codebook（中心表），最接近哪个中心，就输出那个中心的编号。官方 K-means / 向量量化文档也使用这一解释 [6][7]。

先区分以下处理阶段：

| 阶段 | 对一个来源的一条预测样本做什么 | 得到什么 | 是否在此步聚类 |
| --- | --- | --- | --- |
| 固定特征化 | 把每次历史事件变成数值特征；P/C 用固定哈希，T/L 用固定数值规则 | 一串事件特征向量 | 否 |
| 历史摘要 | 汇总最后一步、位置衰减均值、普通均值和长度比例 | 一条历史对应一个摘要向量 | 否 |
| 标准化 | 用参考历史的逐坐标均值和尺度调整摘要 | 同维数的标准化摘要 | 否 |
| `fit()` 中的 K-means | 对参考集内该来源的所有标准化摘要聚类 | 该来源的一张中心表 | **是** |
| `transform()` 中的最近中心分配 | 对每条摘要查找最近的已保存中心 | **一个整数状态，例如 2** | 不重新拟合，只分配已有簇号 |
| 配对与计数 | 把四个来源的状态与实际下一 POI 标签配对 | 五维联合频数表 | 否 |

严格说，量化映射是“摘要到有限状态编号”的部分；附录 D 的 `BasicSourceQuantizer` 为方便调用，还封装了前面的固定特征、摘要和标准化步骤。单事件固定哈希不是 K-means，簇编号也不是哈希向量。

**为什么 P/C 原本已经是离散 ID，还要量化？** 因为当前要描述的是一整段历史，而不是一个单次 POI ID。例如 `[17, 42, 19]` 和 `[17, 42, 33]` 是两条不同的 POI 历史；若把每种完整序列都单独当作一个状态，状态组合可能很多、重复计数很少。当前方案先做固定摘要，再把相近摘要归到少量簇中，得到便于基础频数估计的有限状态。这是有损压缩，会丢失部分历史差异；它并不保证所保留的信息最适合预测，也不保证已经消除有限样本误差。

本版用最基础的固定特征实现：P/C 的 ID 分别转成可复现的哈希向量，T/L 分别转成固定数值特征；各来源独立汇总自己的历史，再进行标准化和 K-means。这里不训练离线 embedding 或序列 Encoder，也没有图构建、图重构或 User/Trail 表示。**“不需要神经表示预训练”仍然包含“需要拟合标准化参数与 K-means 中心”。**

对参考样本 i，四条历史只包含预测时已经发生的事件。设截断后的共同有效长度为 N_i：

$$
a_{m,i,j}=\psi_m(x_{m,i,j}),
\qquad
h^{\mathrm{off}}_{m,i}=g_m(a_{m,i,1:N_i}),
\qquad
Q_{m,i}=\mathcal Q_m(h^{\mathrm{off}}_{m,i}),
\qquad
m\in\{P,C,T,L\}.
$$

一个预测事件生成一组四来源状态和一个下一 POI 标签：

$$
(Q_{P,i},Q_{C,i},Q_{T,i},Q_{L,i},Y_i).
$$

Y 保持具体 venue_id 对应的目标身份。不能把同一下一 POI 标签沿历史长度重复计数；N 个历史事件用于构造一次预测的摘要，不是 N 个独立的 next-POI 观测。user_id、trail_id 只服务于分组、切分、去重和评估，不进入固定特征、摘要、量化器或 SURD 的来源坐标。

### 4.2 P/C：独立的 SHAKE-256 固定符号哈希

第一版对 P 和 C 都使用 64 维固定符号哈希，不对 ID 的整数大小直接聚类。给定来源、固定种子和身份 ID，用 SHAKE-256 生成 64 个字节；每个字节的奇偶性决定该维的正负号，并除以维数平方根：

$$
\begin{aligned}
b_{m,v}
&=\operatorname{SHAKE256}
\left(\operatorname{encode}(\mathrm{seed},\mathrm{salt}_m,v)\right),\\
\psi_m(v)_r
&=\frac{2(b_{m,v,r}\bmod 2)-1}{\sqrt{64}},
\qquad r=1,\ldots,64,
\qquad m\in\{P,C\}.
\end{aligned}
$$

实现使用两个不同的来源 salt：`P/basic_fixed_features_v1` 和 `C/basic_fixed_features_v1`。因此即使某个 POI ID 和某个 Category ID 的整数值相同，也不会有意共享同一个编码。这里的“独立”指特征化规则分离，不表示数据中的 P 与 C 在统计上独立。

哈希规则不依赖标签或训练频次，也不使用 Python 默认的 `hash()`。每个有效的非负 ID 都能被确定性编码，包括参考集中未出现过的 ID；真实事件上的缺失身份统一使用 `UNK=-2`，padding 专用 `PAD=-1`，padding 不参与池化。离线哈希与第 11 节在线可学习 embedding 的词表和 UNK 规则是两套机制，应分别记录。

**64 维哈希没有学习 POI 或类别语义。** 两个向量接近不代表两个 POI 在地理或业务上相近；有限维编码、向量均值和后续量化都可能造成碰撞或信息损失。它提供的是计算简单、可重复的身份数值化基线。哈希维度与 seed 应进入配置，并通过不同 seed 的结果检查统计稳定性。

### 4.3 T/L：6 维时间特征与 8 维位置特征

T 的默认单事件向量为 6 维，与第 11.1 节一致。历史小时和星期使用周期编码，历史相邻时间间隔使用固定单位后的 log1p：

$$
\psi_T(\tau_j)
=
\left[
\sin(2\pi h_j/24),\cos(2\pi h_j/24),
\sin(2\pi w_j/7),\cos(2\pi w_j/7),
\log(1+\Delta\tau_j),
\mathrm{gap\_valid}_j
\right].
$$

统一时区和时间间隔单位，例如以小时计间隔。窗口第一步将间隔与 gap_valid 都置为 0；缺失 timestamp 按第 2 节的数据规则处理，不能以预测目标的时间替代。间隔只使用当前窗口内相邻的历史时间，不读取下一事件。

L 的默认单事件向量为 8 维。经纬度先转换成弧度，记经度为 lambda、纬度为 phi；距离和方向只由相邻历史坐标得到：

$$
\psi_L(\ell_j)
=
\left[
\cos\phi_j\cos\lambda_j,
\cos\phi_j\sin\lambda_j,
\sin\phi_j,
\log(1+\Delta d_j),
\sin b_j,\cos b_j,
\mathrm{coord\_valid}_j,
\mathrm{step\_valid}_j
\right].
$$

距离统一成公里等预先声明的单位。无有效坐标时，位置数值用有限值填充并设置 coord_valid=0；第一步或缺少有效前序坐标时，位移和方向填 0、step_valid=0。零位移时方向未定义，使用零方向向量。缺失坐标不能被当作真实的经纬度原点。L 不读取时间间隔，也不计算需要 T 的速度。

附录 D 的模块接收已经生成的数值张量，不在内部解析 timestamp 或经纬度；默认输入分别为 `T: [B,N,6]`、`L: [B,N,8]`。模块允许预先声明的其他固定维数，但一旦拟合，该统计版本的维数、字段顺序和数值单位就必须保持不变。

### 4.4 四个来源的摘要结构与维数来源

每个来源分别拼接最后一步、按位置衰减的均值、普通均值和历史长度比例：

$$
\begin{aligned}
w_{i,j}
&=
\frac{\exp(-(N_i-j)/\tau_{\mathrm{pos}})}
{\sum_{k=1}^{N_i}\exp(-(N_i-k)/\tau_{\mathrm{pos}})},\\
h^{\mathrm{off}}_{m,i}
&=
\left[
a_{m,i,N_i};
\sum_{j=1}^{N_i}w_{i,j}a_{m,i,j};
\frac{1}{N_i}\sum_{j=1}^{N_i}a_{m,i,j};
\frac{N_i}{N_{\max}}
\right].
\end{aligned}
$$

默认 N_max=50、tau_pos=10。四条序列共同截断，采用右侧 padding，并共享 `valid_mask`；True 表示真实历史事件。权重只依赖位置，所以 P/C 不会通过衰减系数额外读入 Time。

若一个事件的特征为 d_m 维，则最后一步、位置衰减均值、普通均值各占 d_m 维，历史长度比例占 1 维，因此：

$$
\boxed{D_m=d_m+d_m+d_m+1=3d_m+1.}
$$

| 来源 | 单事件维数 d_m 及组成 | 历史摘要维数 D_m | n 条样本的摘要矩阵 |
| --- | --- | --- | --- |
| P | 固定哈希：64 | 3×64+1=193 | n×193 |
| C | 固定哈希：64 | 3×64+1=193 | n×193 |
| T | 小时周期 2＋星期周期 2＋间隔 1＋有效标记 1＝6 | 3×6+1=19 | n×19 |
| L | 球面位置 3＋距离 1＋方向周期 2＋有效标记 2＝8 | 3×8+1=25 | n×25 |

**3 来自三组摘要，不是历史有三个事件；n 是预测样本数，后面的数字才是一条历史的摘要维数。** 历史长度和簇数改变，都不会改变已固定的摘要维数；标准化也保持形状不变。P/C 的 64 是可调整的 `hash_dim`，例如改为 32 时，其摘要维数变为 97。T/L 的 6/8 来自第 4.3 节声明的特征组成。每个样本独立摘要和量化，四路最终仍形成一行四个整数；不同目标前缀不能共用整条 trail 的同一个摘要。

长度是四条对齐历史共同可见的结构信息，不是用户标识。但四路都包含 N_i/N_max，可能使量化后的来源共同携带与历史长度相关的信息，从而形成冗余。解释 SURD 时应把这项摘要设计算在内；需要检查时，可在相同切分下消融长度特征，而不能把相关冗余全部解释成 POI、类别、时间、位置之间的业务重叠。

附录 D 要求每条历史非空。零历史样本不进入本估计器，需要调用方另行提供无历史推荐策略，不伪造四个来源状态后加入联合计数；本文在线序列示意默认历史非空。该摘要保留部分近期顺序信息，但不保留完整顺序，也不等同于语义序列编码。

### 4.5 Quantizer 内部的聚类：四个来源各拟合一个 K-means

#### 4.5.1 聚类对象是一整段历史的摘要

假设参考集有 n 条预测样本，就有 n 条 P 历史、n 条 C 历史、n 条 T 历史和 n 条 L 历史。先逐条摘要，得到下表。**矩阵的一行是一条预测样本的整段来源历史，不是一次单独的 check-in，也不是 POI 目录中的一个 POI。**

| 来源 | 默认聚类输入矩阵 | 一行的含义 | 请求 4 簇时的中心表 | 最终状态列 |
| --- | --- | --- | --- | --- |
| P | n × 193 | 一整段 POI 历史的标准化摘要 | 最多 4 × 193 | n 个整数 |
| C | n × 193 | 一整段 Category 历史的标准化摘要 | 最多 4 × 193 | n 个整数 |
| T | n × 19 | 一整段 Time 历史的标准化摘要 | 最多 4 × 19 | n 个整数 |
| L | n × 25 | 一整段 Location 历史的标准化摘要 | 最多 4 × 25 | n 个整数 |

四路各自聚类，得到四张中心表；不能先把 P/C/T/L 摘要拼接成一个大向量、只聚成一列状态，否则就不再有本文要分解的四个来源变量。

#### 4.5.2 `fit()`：标准化并拟合中心

在当前合法参考集 D_ref 的历史摘要上，分别计算每个来源的坐标均值和总体标准差：

$$
\begin{aligned}
\mu_{m,r}
&=\frac{1}{|\mathcal D_{\mathrm{ref}}|}
\sum_{i\in\mathcal D_{\mathrm{ref}}}h^{\mathrm{off}}_{m,i,r},\\
\sigma_{m,r}
&=\sqrt{\frac{1}{|\mathcal D_{\mathrm{ref}}|}
\sum_{i\in\mathcal D_{\mathrm{ref}}}
(h^{\mathrm{off}}_{m,i,r}-\mu_{m,r})^2},\\
\widetilde\sigma_{m,r}
&=\begin{cases}
\sigma_{m,r},&\sigma_{m,r}>10^{-12},\\
1,&\text{otherwise},
\end{cases}\\
\widetilde h_{m,i,r}
&=\frac{h^{\mathrm{off}}_{m,i,r}-\mu_{m,r}}
{\widetilde\sigma_{m,r}}.
\end{aligned}
$$

之后，对每个来源分别拟合 K-means。其优化目标是让同一来源的摘要尽量接近被分配的中心 [7]：

$$
\min_{c_{m,0},\ldots,c_{m,K_m-1}}
\sum_{i\in\mathcal D_{\mathrm{ref}}}
\min_{0\le k<K_m}
\left\|\widetilde h_{m,i}-c_{m,k}\right\|_2^2.
$$

这里 K_m 是请求的簇数；中心和摘要有相同的特征维数。附录 D 使用固定 seed 初始化中心，重复“分配到最近中心—按已分配摘要更新中心”，达到停止条件后保存中心。这个目标中没有 Y，也没有 SURD 或推荐损失；它拟合的是来源摘要的几何分组，不保证找到了语义最优或预测最优的分组。

本版统一请求每源 4 簇。若参考集中不同摘要不足 4 个，或优化后有未占用中心，附录 D 会减少到实际有效簇数 K_m^eff。因此应保存 `effective_clusters`，不要假设每列一定出现 0、1、2、3 四种值；常量来源可以只有一个状态。

**每张 SURD 表只配一套四来源量化器，所有候选 POI 共用。** 不能先按下一 POI 标签分组、为每个 y 分别拟合中心。per-POI 的含义是后续在同一个联合分布中固定目标 Y=y 计算信息量，不是每个目标重新定义一套 Q。

#### 4.5.3 `transform()`：把摘要变成一个整数编号

拟合完成后，均值、尺度和中心保持冻结。对每条来源历史，重新按相同规则摘要和标准化，再输出最近中心的编号。实现使用从 0 开始的编号：

$$
Q_{m,i}
=\underset{0\leq k<K_m^{\mathrm{eff}}}{\operatorname{arg\,min}}
\left\|\widetilde h_{m,i}-c_{m,k}\right\|_2^2.
$$

**输出的是簇的编号，不是中心向量、距离向量、概率向量或下一 POI 预测。** 例如 P 的摘要为 193 维，中心表为 4×193，量化后这一条 P 历史仍然只得到一个整数。

`fit()` 只接收四源历史与有效位 mask，不接收 Y；`transform()` 不更新中心。缺失身份经 UNK 特征和相同规则映射到已有簇，模块不会额外创造一个名为 unknown 的量化簇。新历史也分配到已保存的最近中心，不因一次新输入临时创建新簇。

#### 4.5.4 手算例：为什么一个向量最后只得到 2

以下使用**假设的已拟合中心**解释最近中心分配，不是用户数据的拟合结果。为了能够手算，只展示二维标准化摘要；实际 P 摘要仍使用第 4.4 节的 193 维。

假设当前 P 历史的二维标准化摘要为：

$$
\widetilde h_{P,i}=(0.2,\,0.9).
$$

已有四个中心，平方距离如下：

| 中心编号 k | 假设中心 c_P,k | 当前摘要到该中心的平方距离 |
| --- | --- | ---: |
| 0 | (0, 0) | 0.2² + 0.9² = 0.85 |
| 1 | (1, 0) | (0.2−1)² + 0.9² = 1.45 |
| 2 | (0, 1) | 0.2² + (0.9−1)² = 0.05 |
| 3 | (1, 1) | (0.2−1)² + (0.9−1)² = 0.65 |

最小值对应中心 2，因此：

$$
q_{P,i}=\operatorname{arg\,min}(0.85,1.45,0.05,0.65)=2.
$$

这个 2 表示“该样本的 P 历史被分到 P 的第 2 号簇”，不表示 POI ID 为 2，也不表示 2 bit 或第二名。与它不同的另一条历史，只要摘要也最接近这个中心，同样可以得到 2。中心编号是类别标签，数字大小没有顺序或距离含义。

C/T/L 分别重复同样的最近中心分配，使用各自的摘要、标准化参数与中心表。P 的第 2 号簇与 C 的第 2 号簇没有共享语义；如果重新拟合另一版本的中心，同一来源的第 2 号簇也未必仍代表原来的模式。

#### 4.5.5 64 维、193 维、4 个簇与 26 维分别是什么

| 数字或配置 | 表示什么 | 不应混淆为 |
| --- | --- | --- |
| `hash_dim=64` | P/C 的单事件固定特征有 64 个坐标 | 64 个来源或 64 个状态 |
| P/C 的摘要维数 193 | 3×64 个摘要坐标加 1 个长度比例 | 193 个簇 |
| `n_clusters=4` | 每个来源请求最多 4 个历史状态 | 四个 POI、四个历史事件或四个 Experts |
| `q_state` 的 4 列 | P、C、T、L 各一个状态编号 | 该样本的 SURD 分解 |
| SURD 的 26 维 | 一个目标 POI 的 U/R/S 分量 | 量化器输出的簇数 |
| MoE 的 15 个 Experts | 4 个 Unique 与 11 个 Synergy Experts | 离散状态数 |
| Router 的 Top-4 | 一次候选打分选用的 Expert 数量 | K-means 的簇数；两者默认都是 4 只是配置上的巧合 |

用 S_m 表示来源 m 的状态集合，区别于前文表示量化映射的花体 Q_m。若四个来源实际都恰好有 4 个状态，则来源联合状态最多有：

$$
\mathcal S_m=\{0,1,2,3\},
\qquad
\left|\mathcal S_P\times\mathcal S_C\times\mathcal S_T\times\mathcal S_L\right|
=4^4=256.
$$

这仍然是四个离散变量的组合，不是 256 个来源。加上 V 个可能的目标 POI，联合表理论上最多有 V×256 个格子，但第 5 节只保存实际出现的格子。少量簇可以降低状态组合数，却不保证每个目标都有充分支持。

4 是基础起始配置，尚未通过用户数据验证其最优性，也不是 SURD 要求的固定值。可以选择 2、3、4、6、8、16 等簇数；簇太少会合并历史差异，簇太多可能使计数更稀疏。改变 K 只改变中心表的行数和可输出状态数，不改变 n×193、n×19、n×25 的摘要形状，四路量化输出仍为 n×4。

#### 4.5.6 簇数如何选择：先比较少量候选，再看统计与推荐效果

四路统一使用 K 个簇时，来源联合状态数的理论上界为 K⁴；实际占用数可能更少：

| 每源 K | 联合状态上界 |
| ---: | ---: |
| 2 | 16 |
| 4 | 256 |
| 8 | 4,096 |
| 16 | 65,536 |

建议采用以下简化流程：

1. **固定条件。** 保持特征、摘要及第 10 节的时间切分不变，先比较 K=2/4/8，必要时扩展。每个 K 只在合法 D_ref 上拟合量化器并计数。
2. **检查支持。** 记录实际簇大小、出现过的联合状态和每个 POI 的目标样本数 n_y，尤其检查低频目标是否大量状态只出现一次；不能仅凭总样本量选 K。
3. **检查稳定性。** 轮廓系数只作几何分组的辅助指标 [8]；结合不同合法时间块或按 trail/用户等分组重采样，观察主要 SURD 分量是否稳定。训练信息量可能受有限样本偏差影响 [9]，不以其最大为唯一目标。
4. **验证后选择。** 比较验证集 Recall@10、NDCG@10 等预先确定的指标；表现相近时优先较小且统计更稳定的 K。每组均需重新量化、计数、分解并生成缓存，测试集不参与选择。

不同来源也可以采用不同 K，例如 P/C/T/L 分别为 8/4/4/8；但当前 `n_clusters` 接口只接受四路共用的一个正整数，分别配置需先扩展接口。现有代码将 `n_clusters=4` 改为 8，即可让四个来源分别请求 8 簇，实际簇数仍以 `effective_clusters` 为准。

### 4.6 在同一个合法参考集上转换并配对标签计数

本基础版不再额外拆分表示拟合集与计数集。先只用 D_ref 的来源历史拟合标准化和 codebook，再用固定变换处理同一批参考历史，并把结果与对应 next-POI 标签配对：

$$
\begin{aligned}
\theta_{\mathrm{quant}}
&=\operatorname{Fit}
\left(\{X_{P,i},X_{C,i},X_{T,i},X_{L,i}\}_{i\in\mathcal D_{\mathrm{ref}}}\right),\\
\mathbf q_i
&=\operatorname{Transform}_{\theta_{\mathrm{quant}}}
\left(X_{P,i},X_{C,i},X_{T,i},X_{L,i}\right),\\
C(y,q_P,q_C,q_T,q_L)
&=\sum_{i\in\mathcal D_{\mathrm{ref}}}
\mathbf 1\{Y_i=y,\ \mathbf q_i=(q_P,q_C,q_T,q_L)\}.
\end{aligned}
$$

这就得到第 5–7 节及附录 C 需要的联合计数输入：`q_state: [n,4]`，列顺序固定为 P、C、T、L；`y_index: [n]`，对应统一候选 POI 目录的行编号。目标标签只参与配对计数，不参与历史摘要、标准化或 K-means。

外层评分折、验证集、测试集仍然排除在该参考集之外；严格时间实验还要求参考目标已在评分时刻之前发生。具体构造见第 10 节。在同一参考集上拟合量化并计数是本版的基础估计协议，不意味着频数估计或 SURD 分量无偏。

#### 4.6.1 最终获得的四个离散变量是什么样的

用大写 Q_P、Q_C、Q_T、Q_L 表示参考分布中的四个随机变量，用小写 q_P,i、q_C,i、q_T,i、q_L,i 表示第 i 条样本上实际得到的四个值。若对应来源实际有 4 个状态，则该变量的取值集合为 `{0, 1, 2, 3}`；每条样本只取其中一个值。

沿用第 2.4 节三步历史的结构，下面展示**假设已拟合中心下的一次赋值**。表中的簇号只用于说明最终数据形态，不能从原始 ID 或时间数字直接推断出来；具体编号必须由实际保存的中心计算。

| 来源 | 一条样本中的原始历史示意 | 先得到的默认摘要 | 假设最终状态 |
| --- | --- | --- | ---: |
| P | POI `[17, 42, 19]` | 193 维向量 | 2 |
| C | 对齐的类别 `[3, 8, 3]` | 193 维向量 | 0 |
| T | 三次已观测时间，例如同日 08:00、08:20、08:50 | 19 维向量 | 1 |
| L | 与以上三次访问对齐的三个历史坐标 | 25 维向量 | 3 |

这条样本最终得到：

$$
\mathbf q_i=(q_{P,i},q_{C,i},q_{T,i},q_{L,i})=(2,0,1,3).
$$

它的历史长度是 3，但输出不是 3×4 个状态，而是**四个来源各一个状态，共 4 个整数**。历史事件先参与摘要，然后整条摘要才被分配簇号。如果这次预测对应的真实下一 POI 是 `poi_42`，则额外配上 Y=`poi_42`。这个目标没有参与刚才的四次来源变换。

#### 4.6.2 多条样本放在一起：n×4 的整数表

下表是 **8 个不同预测事件的合成离散样本**，用于演示计数。状态值为教学构造，不声称是上一表或第 4.8 节代码的实际拟合输出。

| 样本 i | q_P | q_C | q_T | q_L | 实际下一 POI：y_i |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 2 | 0 | 1 | 3 | poi_42 |
| 2 | 2 | 0 | 1 | 3 | poi_42 |
| 3 | 2 | 1 | 1 | 3 | poi_42 |
| 4 | 0 | 1 | 2 | 1 | poi_17 |
| 5 | 0 | 1 | 2 | 1 | poi_17 |
| 6 | 1 | 2 | 0 | 2 | poi_8 |
| 7 | 2 | 0 | 1 | 3 | poi_17 |
| 8 | 3 | 3 | 3 | 0 | poi_8 |

对应的程序数据可以写成：

```python
# Synthetic post-quantization states; columns are P, C, T, L.
q_state = [
    [2, 0, 1, 3],
    [2, 0, 1, 3],
    [2, 1, 1, 3],
    [0, 1, 2, 1],
    [0, 1, 2, 1],
    [1, 2, 0, 2],
    [2, 0, 1, 3],
    [3, 3, 3, 0],
]  # shape: [8, 4]

target_catalog = ["poi_42", "poi_17", "poi_8"]
y_index = [0, 0, 0, 1, 1, 2, 1, 2]  # shape: [8]
```

`y_index=0` 只是 `poi_42` 在统一候选目录中的行号，属于一一对应的身份编码，不是把多个目标 POI 聚成“目标簇 0”。不同目标 POI 必须保持不同目录行。

样本 1、2、7 的四个来源状态完全相同，但实际下一 POI 可以不同。这正是联合概率表需要记录的现象：一个量化状态描述一类历史，并不唯一确定未来。样本 1、2 是两个不同预测事件，不是将同一个真实目标事件复制计数。

下一步只做“按相同五元组累加次数”，无需再次聚类。第 5.4 节继续使用这 8 行，展示联合频数与边缘化。

### 4.7 这套分解能够解释什么

本版分解的对象是固定版本的四个离散摘要：

$$
I(Y;Q_P,Q_C,Q_T,Q_L).
$$

P 的特征只读取历史 POI 身份，C 只读取历史类别，T 只读取历史时间，L 只读取历史坐标；共同的位置、有效位和长度只描述对齐的历史结构。来源隔离不等于来源之间统计独立，更不保证某个 Unique 或 Synergy 分量一定为正。

固定哈希、池化和少量簇会改变保留下来的信息。因此这些结果描述“当前摘要和量化规则下，四个来源对目标 POI 的独有、重叠和组合信息”，不能直接当作完整原始序列或在线神经表示的分解。若 Category/Location 是 POI 的固定元数据，量化后出现非零 Category Unique 或组合增量，仍应按第 21.1 节解释为摘要之间的信息保留差异。

来源摘要与候选无关；不能用每行真实 Y 作为 query 先生成表示，再混合计数。对于未见的有效 POI ID，能生成哈希向量也不代表已有针对该目标的 SURD 统计支持；目标行是否存在、可靠性及回退仍由第 9、15 节决定。

### 4.8 最简可运行接口：拟合、转换与保存量化器

将附录 D 保存为 `basic_source_quantizer.py` 后，下面的玩具数据可在具备 PyTorch 的环境中运行，用于核对接口和保存/加载一致性。真实实验应将 `ref_history` 与 `ref_valid_mask` 替换成第 2 节构造的合法参考历史。这里只用 4 条样本演示接口；请求 4 簇时，某些来源可能每条摘要独占一簇，不能据此判断统计可靠性或推荐效果。

示例的基础时间和位置固定，用于保持 6/8 维特征约定，但历史长度与 gap_valid/step_valid 有差别，因此 T/L 的摘要仍可能不同，不能预先断言它们都属于同一簇。第 4.6 节的 8 行教学状态是独立例子，不是下面代码的打印结果。

```python
from pathlib import Path
import torch
from basic_source_quantizer import BasicSourceQuantizer

ref_history = {
    "P": torch.tensor([[11, 22, 33], [22, 11, -1],
                       [33, 11, 22], [11, 33, -1]]),
    "C": torch.tensor([[1, 2, 1], [2, 1, -1],
                       [1, 1, 2], [1, 1, -1]]),
    "T": torch.zeros(4, 3, 6, dtype=torch.float64),
    "L": torch.zeros(4, 3, 8, dtype=torch.float64),
}
ref_valid_mask = ref_history["P"].ne(-1)  # True = real history event
# Toy events: Monday 00:00, zero gaps, valid identical coordinates (0, 0).
ref_history["T"][..., 1] = 1.0  # cos(hour)
ref_history["T"][..., 3] = 1.0  # cos(weekday)
ref_history["T"][:, 1:, 5] = ref_valid_mask[:, 1:]
ref_history["L"][..., 0] = 1.0  # spherical x
ref_history["L"][..., 6] = ref_valid_mask
ref_history["L"][:, 1:, 7] = ref_valid_mask[:, 1:]

qz = BasicSourceQuantizer(
    n_clusters=4, hash_dim=64, max_length=50, seed=0, tau_pos=10.0
)
qz.fit(ref_history, ref_valid_mask)  # Fits four K-means codebooks; no Y.
q_state = qz.transform(ref_history, ref_valid_mask)  # Assigns cluster IDs; CPU long [n, 4]
summaries = qz.summarize(ref_history, ref_valid_mask)
print("summary shapes:", {s: tuple(v.shape) for s, v in summaries.items()})
print("q_state shape/dtype:", tuple(q_state.shape), q_state.dtype)

fold_id = 0
quantizer_path = Path("offline_cache") / f"basic_quantizer_fold_{fold_id}.pt"
quantizer_path.parent.mkdir(parents=True, exist_ok=True)
qz.save(quantizer_path)
qz_loaded = BasicSourceQuantizer.load(quantizer_path)
assert torch.equal(q_state, qz_loaded.transform(ref_history, ref_valid_mask))
print("q_state (P, C, T, L):", q_state.tolist())
print("effective clusters:", qz.effective_clusters)
```

在默认配置下，摘要形状分别为 P/C 的 `(4,193)`、T 的 `(4,19)`、L 的 `(4,25)`；`q_state` 的形状为 `(4,4)`、类型为 `torch.int64`。具体簇编号以实际输出为准，不能手工要求某条历史必然属于编号 0 或 2。第一列是各样本的 P 状态，第二列是 C 状态，依次类推。

接下来只需把 `q_state` 与对齐的 `y_index` 交给第 16.1 节的 `build_offline_surd`。标准化/codebook 检查点与 SURD 缓存应保存为匹配版本，第 16–17 节记录配置及检查点摘要。

**当前架构中，四个离散状态用于离线计数；在线推荐仍将四条历史序列交给各自的在线 Encoder，并读取候选 POI 的 26 维 SURD 缓存行。** 不需要先把在线历史压缩成这 4 个整数再替代原有序列输入，也不需要每次打分重拟合 K-means、重新计数或重新计算 SURD。只有复现离散状态或做状态诊断时，才需要加载冻结量化器并调用 `transform()`。

### 4.9 如果不使用聚类，是否也能获得离散变量

**可以，但那将是另一种离散来源定义；本文当前基础实现仍采用 K-means。** 本文的联合频数实现需要有限、可计数的来源状态，并不要求状态一定由聚类产生。规则分箱、固定空间网格、直接保留有限类别等，也能构造离散变量。

例如，可另设一个更易解释、但只保留最后一步信息的对照方案：

| 来源 | 一个不聚类的对照定义 | 一次取值的例子 |
| --- | --- | --- |
| Q_P | 最后一次历史 POI 的身份编号 | 19，对应某个具体历史 POI |
| Q_C | 最后一次历史类别的身份编号 | 3，对应某个具体类别 |
| Q_T | 最后一次历史小时按 6 小时分箱 | 08:50 属于 `[06:00,12:00)`，编号 1 |
| Q_L | 最后一次历史坐标所在的预定义网格编号 | 网格 205 |

这样也可形成 `(19,3,1,205,Y)` 并统计联合频数。但是它与当前“整段历史摘要聚类”的语义不同：P/C 的编号在该对照中表示原始身份，而不是历史簇；过去多步访问、时间间隔和移动模式也没有按当前摘要保留。直接保留大量 POI 或网格状态还可能使频数更稀疏。

因此，是否采用聚类是离散化方案的选择。对当前 193/193/19/25 维历史摘要，K-means 已经承担了“归并为少量状态”的工作，后面不需要另加聚类。若换成上述规则方案，必须重新定义 Q、重新计数、重新分解，不能沿用当前量化版本的 SURD 缓存。

## 5. 路径 A：从一个联合概率表得到全部子集概率

### 5.1 建立五维稀疏联合频数表

先在第 4 节允许使用的参考集历史上拟合标准化与量化器，冻结后变换这些历史，再与各自的下一 POI 标签配对。量化器拟合不读取标签；本基础版允许同一合法参考集用于历史量化拟合和频数统计，外层评分块、验证及测试数据仍按第 10 节隔离。离散化后的参考集为：

$$
\mathcal D_{\mathrm{ref}}
=
\left\{
(q_{P,i},q_{C,i},q_{T,i},q_{L,i},y_i)
\right\}_{i=1}^{n}.
$$

统计：

$$
C(y,q_P,q_C,q_T,q_L)
=
\sum_{i=1}^{n}
\mathbf1
\left[
(y_i,q_{P,i},q_{C,i},q_{T,i},q_{L,i})
=
(y,q_P,q_C,q_T,q_L)
\right].
$$

定义同一个经验联合分布：

$$
\widehat p(y,q_P,q_C,q_T,q_L)
=
\frac{C(y,q_P,q_C,q_T,q_L)}{n}.
$$

主版本采用**无额外平滑的经验分布**，使用字典或哈希表保存出现过的状态。无需分配完整的稠密五维张量。

实现中固定使用键 `(y, q_P, q_C, q_T, q_L)`。五个维度是“一个预测目标 + 四个来源”，没有 user_id 维度；Category 已经是四来源之一，不额外增加第六维。各来源状态由完整合法历史窗口各生成一个。

第 4.6 节的 `q_state` 只保存右侧四个来源坐标；`y_index` 提供目标坐标。代码中可用目标目录行号作为键的第一项，展示时再映射回具体 POI 名称。建立联合表只是在同一套冻结状态定义下计数，不重新拟合任何来源量化器。

### 5.2 子集索引与固定顺序

非空子集集合为：

$$
\mathcal A
=
\{
P,C,T,L,
PC,PT,PL,CT,CL,TL,
PCT,PCL,PTL,CTL,
PCTL
\}.
$$

这是先按阶数、再按固定来源顺序 P、C、T、L 枚举的顺序。代码、文件列与 Router 必须使用同一个索引约定。C 表示 Category，联合频数 C(·) 是另一个带参数的记号，二者不要混淆。

### 5.3 从联合表边缘化

对任意子集：

$$
C_A(y,q_A)
=
\sum_{q_{\mathcal M\setminus A}}
C(y,q_P,q_C,q_T,q_L).
$$

继续得到：

$$
C_A(q_A)
=
\sum_y C_A(y,q_A),
\qquad
n_y
=
\sum_{q_A}C_A(y,q_A).
$$

因此：

$$
\widehat p(y)=\frac{n_y}{n},
\qquad
\widehat p(q_A\mid y)
=
\frac{C_A(y,q_A)}{n_y},
\qquad
\widehat p(y\mid q_A)
=
\frac{C_A(y,q_A)}{C_A(q_A)}.
$$

所有 15 个子集必须来自这一个联合表。禁止对每张子集表独立加入不同的 Laplace 伪计数；这可能使它们不再对应同一个联合分布。

对正文公式中真正出现的联合非零状态，相关分母必为正，因此计算 specific information 时不需要给每项概率随意加 epsilon。联合零项按零贡献处理。

当某个目标 POI 的参考样本数为零时，事件级分解不可估计。缓存中记录 unknown 标志，数值占位为零；不能将其解释为“已证明该 POI 没有预测信息”。

### 5.4 接着第 4.6 节的例子：8 行状态怎样变成一张概率表

对第 4.6.2 节的 8 个合成预测事件，按目标与四个来源状态一起聚合，得到：

| 目标 y | q_P | q_C | q_T | q_L | 联合频数 C | 联合概率 C/8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| poi_42 | 2 | 0 | 1 | 3 | 2 | 2/8 |
| poi_42 | 2 | 1 | 1 | 3 | 1 | 1/8 |
| poi_17 | 0 | 1 | 2 | 1 | 2 | 2/8 |
| poi_17 | 2 | 0 | 1 | 3 | 1 | 1/8 |
| poi_8 | 1 | 2 | 0 | 2 | 1 | 1/8 |
| poi_8 | 3 | 3 | 3 | 0 | 1 | 1/8 |

8 个样本合并成 6 个实际出现的联合状态；频数总和仍为 8。比如：

$$
C(\mathrm{poi\_42},2,0,1,3)=2,
\qquad
\widehat p(\mathrm{poi\_42},2,0,1,3)=\frac{2}{8}.
$$

相同四来源状态 `(2,0,1,3)` 总共出现 3 次，其中 2 次的目标是 poi_42，1 次是 poi_17，因此：

$$
\widehat p\bigl(Y=\mathrm{poi\_42}\mid Q_P=2,Q_C=0,Q_T=1,Q_L=3\bigr)
=\frac{2}{3}.
$$

这说明“量化状态”与“目标概率”是两个不同层次：Quantizer 只输出状态；概率要由许多带标签的样本共同计数得到。

如果现在只看 P，不再关心 C/T/L，就将同一目标、同一 P 状态下的频数相加，而不是再次聚类：

$$
C_P(\mathrm{poi\_42},2)
=\sum_{q_C,q_T,q_L}C(\mathrm{poi\_42},2,q_C,q_T,q_L)
=2+1=3.
$$

包括其他目标后，P 状态 2 的总次数是 4：

$$
C_P(2)=3+1=4,
\qquad
n_{\mathrm{poi\_42}}=3,
\qquad n=8.
$$

于是，观察 P 状态 2 后，poi_42 的概率为 3/4；不看来源时，其基准概率为 3/8。这些计数就是第 6 节点互信息与 specific information 的输入。例如：

$$
\widehat i(\mathrm{poi\_42};q_P=2)
=\log_2\frac{3/4}{3/8}
=1\ \mathrm{bit}.
$$

这里特意保留了其他目标的数据来计算分母 4；不能为了做 per-POI 分解，先删除所有目标不是 poi_42 的样本。其他 14 个来源子集也从同一个联合表投影并累加，不改变已有簇号。最终第 7 节才将每个目标的 15 个 specific information 分配为 26 个 U/R/S 分量；这 26 个分量不是 Quantizer 直接输出的。

下列代码可用 Python 标准库复核上述频数与边缘化。它接续第 4.6.2 节的合成 `q_state`、`y_index`、`target_catalog`，不调用聚类，也不需要 PyTorch：

```python
from collections import Counter
from math import log2

joint = Counter(
    (target_catalog[row], *q)
    for row, q in zip(y_index, q_state)
)
marginal_p = Counter()
for (y, q_p, q_c, q_t, q_l), count in joint.items():
    marginal_p[y, q_p] += count

n = sum(joint.values())
n_y = sum(count for (y, _), count in marginal_p.items() if y == "poi_42")
count_p2 = sum(count for (_, q_p), count in marginal_p.items() if q_p == 2)

assert n == 8 and len(joint) == 6
assert joint["poi_42", 2, 0, 1, 3] == 2
assert marginal_p["poi_42", 2] == 3
assert n_y == 3 and count_p2 == 4
print(log2((3 / count_p2) / (n_y / n)))  # 1.0 bit
```

这些数值仅用于验证定义与计算流程，样本量不足以支持关于真实推荐规律的结论。

## 6. Per-POI Specific Information：从点值到目标事件平均

本节的事件级 specific information 定义与 SURD 目标状态分解一致 [1][2]。

### 6.1 点互信息：一个来源状态如何改变对一个 POI 的判断

Q_A 是来源子集 A 的离散状态变量，q_A 是它的一次取值，y 是固定目标 POI。点互信息为：

$$
i(y;q_A)
=\log_2\frac{p(y\mid q_A)}{p(y)}
=\log_2\frac{p(q_A\mid y)}{p(q_A)}.
$$

它回答：观察到 q_A 后，事件 Y=y 的概率相对基准概率增加还是减少？正值表示提高，零表示不变，负值表示降低。1 bit 表示概率比为 2；−1 bit 表示概率比为 1/2。

点互信息不是推荐模型的候选分数，也不是已经分解好的 Unique 或 Synergy。它同时取决于 y 和本次状态 q_A，同一个 POI 对不同状态可以得到不同符号。

从同一个联合表边缘化后：

$$
\widehat i(y;q_A)
=\log_2\frac{n\,C_A(y,q_A)}{n_y\,C_A(q_A)}.
$$

n 是参考集事件总数，n_y 是目标为 y 的事件数，C_A(y,q_A) 是同时满足该目标和该来源状态的事件数，C_A(q_A) 是不区分目标的来源状态事件数。该表达式用于联合计数非零的项。

### 6.2 事件级 specific information：固定 POI 后的条件平均

SURD 本版使用的量是固定目标 Y=y 后，对来源状态取条件平均：

$$
\begin{aligned}
\iota_A(y)
&=\mathbb E_{Q_A\mid Y=y}[i(y;Q_A)]\\
&=\sum_{q_A}p(q_A\mid y)
\log_2\frac{p(y\mid q_A)}{p(y)}\\
&=D_{\mathrm{KL}}
\bigl(p(Q_A\mid Y=y)\,\Vert\,p(Q_A)\bigr)\ge0.
\end{aligned}
$$

它回答：在所有最终访问 y 的历史事件中，来源子集 A 总体提供多少关于这个目标事件的可预测信息？

| 量 | 固定或平均的对象 | 本文用途 |
| --- | --- | --- |
| i(y;q_A) | 固定一个 y 和一个 q_A | 解释特定状态相对基准的支持或反支持 |
| iota_A(y) | 固定 y，按 p(q_A∣y) 对状态平均 | 逐 POI 的 SURD 输入 |
| I(Y;Q_A) | 再按 p(y) 对目标平均 | 全局信息量和熵分析 |

计数公式就是上述条件平均的经验估计：

$$
\boxed{
\widehat\iota_A(y)
=
\sum_{q_A:C_A(y,q_A)>0}
\underbrace{\frac{C_A(y,q_A)}{n_y}}_{\text{目标 }y\text{ 下的状态权重}}
\underbrace{
\log_2\frac{n\,C_A(y,q_A)}{n_y\,C_A(q_A)}
}_{\widehat i(y;q_A)}
}
$$

因此两者不是不同方法：一个是单个状态的 log 概率比，一个是这些 log 概率比的条件平均。平均前允许负贡献；平均后因等价于 KL 散度而非负。不能把负点互信息先置零，否则改变了被估计的量。

“固定 y”不等于删掉其他目标后重新估计全部概率。p(q_A) 和 p(y) 仍来自完整参考分布。如果只保留 y，再计算普通互信息，Y 会成为常数，结果退化为零。

### 6.3 合法计数例：负点互信息如何参与正的平均

设参考集只有两个目标 POI，分别为 y 和 y′，有 100 个预测事件；来源子集 A 有状态 a、b：

| 来源状态 | 目标 y | 目标 y′ | 状态总数 |
| --- | ---: | ---: | ---: |
| a | 12 | 18 | 30 |
| b | 8 | 62 | 70 |
| 合计 | 20 | 80 | 100 |

目标 y 的基准概率为 0.2。于是：

$$
\widehat i(y;a)
=\log_2\frac{12/30}{20/100}
=1\ \mathrm{bit},
$$

$$
\widehat i(y;b)
=\log_2\frac{8/70}{20/100}
=-0.807354922\ \mathrm{bit}.
$$

在 20 个目标为 y 的事件中，a 占 12/20，b 占 8/20：

$$
\widehat\iota_A(y)
=\frac{12}{20}\times1
+\frac{8}{20}\times(-0.807354922)
=0.277058031\ \mathrm{bit}.
$$

若错误地把负项置零，会得到 0.6 bit，夸大信息量。0.277 bit 描述目标 y 的平均信息结构，不表示每条最终访问 y 的轨迹都得到这个数值，也不能直接设为当前 Router 权重。

另一目标的事件级信息与全局平均为：

$$
\widehat\iota_A(y')=0.020418639\ \mathrm{bit},
$$

$$
\widehat I(Y;Q_A)
=0.2\times0.277058031
+0.8\times0.020418639
=0.071746517\ \mathrm{bit}.
$$

### 6.4 逐 POI 与全局量的关系

每个 POI 获得 15 个事件级信息量：

$$
\boldsymbol\iota_y
=
[\iota_A(y)]_{A\in\mathcal A}
\in\mathbb R_{\ge0}^{15}.
$$

对目标分布加权才得到普通互信息：

$$
I(Y;Q_A)
=
\sum_y p(y)\iota_A(y).
$$

必须**先逐目标事件分解，再根据需要加权平均**。因为排序、最小值和最大值是非线性操作，不能先平均 15 个信息量，再声称得到了同一组事件级分解的平均。

“按目标事件分析”也不等于只保留单个目标类别后计算普通互信息。若在数据中把 Y 固定为常数，再计算普通的来源与 Y 的互信息，该量会退化为零。

### 6.5 同一分布必须满足的约束

对有观测的目标：

$$
0
\le
\iota_A(y)
\le
\iota_B(y)
\le
-\log_2 p(y),
\qquad
A\subseteq B.
$$

这些是验证条件，不是用来掩盖估计错误的随意截断规则。经验联合表在精确算术下满足它们；浮点误差只允许在声明的数值容差内修正。

## 7. Per-POI SURD：统一获取公式与逐项计算

以下采用原始 SURD 的事件级分配规则 [1][2]；统一 U/R 公式是该排序规则的等价写法。它不等同于一般多源 PID 格的所有原子定义。

### 7.1 Unique 与 Redundancy 的统一公式

固定目标 y，记四个单来源信息为：

$$
a_m(y)=\iota_{\{m\}}(y),
\qquad
m\in\mathcal M=\{P,C,T,L\},
\qquad
[x]_+=\max(x,0).
$$

每个单来源 Unique 为：

$$
\boxed{
u_m(y)
=
\left[
a_m(y)
-\max_{j\in\mathcal M\setminus\{m\}}a_j(y)
\right]_+.
}
$$

它是该单来源超过其他最强单来源的增量。固定 y 时，至多一个来源获得正 Unique；最大值并列时，它们的 Unique 都为零。这不是给定其余全部来源的条件互信息定义。

对任意至少包含两个来源的集合 A：

$$
\boxed{
r_A(y)
=
\left[
\min_{m\in A}a_m(y)
-\max_{j\in\mathcal M\setminus A}a_j(y)
\right]_+,
\qquad |A|\ge2.
}
$$

空补集的最大值约定为零：

$$
\max_{j\in\varnothing}a_j(y)=0,
\qquad
r_{\mathcal M}(y)=\min_{m\in\mathcal M}a_m(y).
$$

冗余公式分配“集合内每个来源都达到，而集合外来源尚未达到”的单来源信息区间。这是当前 SURD 的数值分配规则，不能根据“冗余”一词将其替换成另一种 PID 的重叠定义。

统一公式与原文链式排序完全等价。若：

$$
a_{m_1}\le a_{m_2}\le a_{m_3}\le a_{m_4},
\qquad a_{m_0}=0,
$$

则：

$$
r_{\{m_j,\ldots,m_4\}}
=a_{m_j}-a_{m_{j-1}},
\quad j=1,2,3,
\qquad
u_{m_4}=a_{m_4}-a_{m_3}.
$$

其余槽位为零。并列值间的增量为零，因而直接公式与链式计算在精确并列时也一致。

### 7.2 Synergy 的统一公式与同阶分配

先定义每一阶的最大信息：

$$
B_k(y)
=\max_{\substack{A\subseteq\mathcal M\\|A|=k}}\iota_A(y),
\qquad k=1,2,3,4.
$$

将当前阶全部子集按信息量升序排序：

$$
\iota_{A_{k,1}}(y)\le\cdots\le\iota_{A_{k,n_k}}(y),
\qquad n_k=\binom4k.
$$

精确并列按固定子集索引排序，首项的前置值设为 0。统一协同公式为：

$$
\boxed{
s_{A_{k,j}}(y)
=
\left[
\iota_{A_{k,j}}(y)
-\max\left\{
B_{k-1}(y),\iota_{A_{k,j-1}}(y)
\right\}
\right]_+,
\quad k=2,3,4.
}
$$

两个比较对象有不同职责：上一阶全体组合的最大值决定哪些信息属于更高阶增量；当前阶前一个值避免同阶重复分配。不能对每个组合独立减去它自己的某个子集信息，再把结果拼起来。

精确并列时，排序较前的组合获得尚未分配的区间，其余并列组合为零。近似并列要做稳定性分析，不能从一次细小排序差异推出强语义结论。按阶求和满足：

$$
\sum_{|A|=k}s_A(y)=B_k(y)-B_{k-1}(y).
$$

### 7.3 从同一个分布逐项计算 26 个分量

使用第 8.1 节的合法擦除分布：公平二值 Y，P/C/T/L 分别以 0.4/0.3/0.1/0.2 的概率透露 Y，其余时擦除。任意子集有：

$$
\iota_A(y)=1-\prod_{m\in A}(1-\rho_m),
\qquad
(a_P,a_C,a_T,a_L)=(0.4,0.3,0.1,0.2).
$$

先代入单来源与冗余公式：

$$
u_P=[0.4-\max(0.3,0.1,0.2)]_+=0.1,
$$

$$
r_{PC}=[\min(0.4,0.3)-\max(0.1,0.2)]_+=0.1,
$$

$$
r_{PCL}=[\min(0.4,0.3,0.2)-0.1]_+=0.1,
\qquad
r_{PCTL}=\min(0.4,0.3,0.1,0.2)=0.1.
$$

其他不形成正信息区间的项为零，例如：

$$
r_{PT}=[\min(0.4,0.1)-\max(0.3,0.2)]_+=0.
$$

二阶下界是 0.4；按升序逐项分配：

| 顺序 | 组合 | 组合信息 | 比较阈值 | Synergy |
| ---: | --- | ---: | ---: | ---: |
| 1 | TL | 0.28 | 0.40 | 0 |
| 2 | CT | 0.37 | 0.40 | 0 |
| 3 | CL | 0.44 | 0.40 | 0.04 |
| 4 | PT | 0.46 | 0.44 | 0.02 |
| 5 | PL | 0.52 | 0.46 | 0.06 |
| 6 | PC | 0.58 | 0.52 | 0.06 |

三阶下界是二阶最大值 0.58：

| 顺序 | 组合 | 组合信息 | 比较阈值 | Synergy |
| ---: | --- | ---: | ---: | ---: |
| 1 | CTL | 0.496 | 0.580 | 0 |
| 2 | PTL | 0.568 | 0.580 | 0 |
| 3 | PCT | 0.622 | 0.580 | 0.042 |
| 4 | PCL | 0.664 | 0.622 | 0.042 |

四阶仅一项：

$$
s_{PCTL}=0.6976-0.664=0.0336.
$$

按缓存顺序列出全部 26 项：

| 类型 | 分量 | bit |
| --- | --- | ---: |
| Unique | U_P | 0.1 |
| Unique | U_C | 0 |
| Unique | U_T | 0 |
| Unique | U_L | 0 |
| Redundancy | R_PC | 0.1 |
| Redundancy | R_PT | 0 |
| Redundancy | R_PL | 0 |
| Redundancy | R_CT | 0 |
| Redundancy | R_CL | 0 |
| Redundancy | R_TL | 0 |
| Redundancy | R_PCT | 0 |
| Redundancy | R_PCL | 0.1 |
| Redundancy | R_PTL | 0 |
| Redundancy | R_CTL | 0 |
| Redundancy | R_PCTL | 0.1 |
| Synergy | S_PC | 0.06 |
| Synergy | S_PT | 0.02 |
| Synergy | S_PL | 0.06 |
| Synergy | S_CT | 0 |
| Synergy | S_CL | 0.04 |
| Synergy | S_TL | 0 |
| Synergy | S_PCT | 0.042 |
| Synergy | S_PCL | 0.042 |
| Synergy | S_PTL | 0 |
| Synergy | S_CTL | 0 |
| Synergy | S_PCTL | 0.0336 |

守恒和单来源重建可以直接核对：

$$
0.1+0.3+0.2976=0.6976=\iota_{PCTL}(y),
$$

$$
\iota_C(y)
=u_C+r_{PC}+r_{PCL}+r_{PCTL}
=0+0.1+0.1+0.1=0.3.
$$

两个目标在这个分布中对称，全局互信息也为0.6976 bit。26维归一化后的 U/S 总质量为0.569954128，冗余总质量为0.430045872；它们仍不是当前 Router 的 softmax 概率。

### 7.4 26 项的固定存储顺序

$$
\mathbf d_y^{\mathrm{raw}}
=
[
\mathbf u_y;
\mathbf r_y;
\mathbf s_y
]
\in\mathbb R_{\ge0}^{26}.
$$

| 分组 | 列序列 | 列数 |
| --- | --- | --- |
| Unique | U_P、U_C、U_T、U_L | 4 |
| Redundancy | R_PC、R_PT、R_PL、R_CT、R_CL、R_TL、R_PCT、R_PCL、R_PTL、R_CTL、R_PCTL | 11 |
| Synergy | S_PC、S_PT、S_PL、S_CT、S_CL、S_TL、S_PCT、S_PCL、S_PTL、S_CTL、S_PCTL | 11 |

这些是 26 个可能的分量槽位，不意味着对某个 POI 全部同时非零。零基索引中，Unique 占 0–3，Redundancy 占 4–14，Synergy 占 15–25。U_P 中的 U 是 Unique 类型、P 是 POI 来源；U_C 是 Category 的 Unique，文档不再有 User 来源的 U_U。

### 7.5 分解守恒与单来源重建

记完整来源的事件级信息量为：

$$
T_y=\iota_{PCTL}(y).
$$

需要验证：

$$
\boxed{
T_y
=
\sum_m u_m(y)
+
\sum_{|A|\ge2}r_A(y)
+
\sum_{|A|\ge2}s_A(y)
}
$$

还应验证每个单来源：

$$
\iota_m(y)
=
u_m(y)
+
\sum_{A:m\in A}r_A(y).
$$

对各阶协同：

$$
\sum_{|A|=k}s_A(y)
=
B_k(y)-B_{k-1}(y).
$$

对目标分布加权：

$$
I(Y;Q_P,Q_C,Q_T,Q_L)
=
\sum_y p(y)T_y.
$$

剩余不确定性为：

$$
H(Y\mid Q_P,Q_C,Q_T,Q_L)
=
H(Y)-\sum_y p(y)T_y.
$$

它单独记录，不成为第 27 个分量，也不成为 Expert。它与数据使用上的“标签泄漏”是不同概念。

不要额外假设任意多来源子集的互信息，都能按“名字包含该子集”的若干分量简单相加重建；本节检查的是原始定义支持的整体与单来源关系。

#### 两种熵怎样从同一张表计算

令完整四来源状态为 q：

$$
C(y,q)=C(y,q_P,q_C,q_T,q_L),
\quad
C(q)=\sum_yC(y,q),
\quad
n_y=\sum_qC(y,q),
\quad
n=\sum_{y,q}C(y,q).
$$

目标熵由 POI 目标频率得到：

$$
\boxed{
\widehat H(Y)
=-\sum_{y:n_y>0}
\frac{n_y}{n}\log_2\frac{n_y}{n}.
}
$$

它衡量不观察四来源时，下一 POI 的平均不确定性。零概率项按零贡献处理，不直接计算 log(0)。

条件熵由同一联合表直接得到：

$$
\boxed{
\widehat H(Y\mid Q_P,Q_C,Q_T,Q_L)
=-\sum_{y,q:C(y,q)>0}
\frac{C(y,q)}{n}\log_2\frac{C(y,q)}{C(q)}.
}
$$

它衡量观察完整来源状态后仍然剩余的平均不确定性。先对每个 q 计算目标条件分布的熵，再按 p(q) 平均，与上式等价。

也可以用 SURD 总量得到：

$$
\widehat H(Y\mid Q_P,Q_C,Q_T,Q_L)
=
\widehat H(Y)-\sum_y\frac{n_y}{n}T_y,
\qquad
T_y=\widehat\iota_{PCTL}(y).
$$

代码应同时实现“联合计数直接计算”和“目标熵减互信息”，并检查两者相等。这里的熵是参考分布上的统计量，不是在线推荐 softmax 的训练交叉熵，也不是单个目标 POI 的神经分类损失。

擦除例中：

$$
H(Y)=1,\quad I(Y;Q_P,Q_C,Q_T,Q_L)=0.6976,
\quad H(Y\mid Q_P,Q_C,Q_T,Q_L)=0.3024
\quad\mathrm{bit}.
$$

第6.3节的计数例若视为一个来源变化、其余来源为常数，则：

$$
\widehat H(Y)=0.721928095,\quad
\widehat H(Y\mid Q)=0.650181578,\quad
\widehat I(Y;Q)=0.071746517
\quad\mathrm{bit}.
$$

经验条件熵小不保证测试集容易预测；过细且稀疏的状态可能让经验条件熵过小，仍须结合覆盖率、折外表现和重采样稳定性解释。

### 7.6 相同值与近似相同值

本文件的参考实现采用：

1. 数值从小到大排序。
2. 精确相同的值，按固定子集索引的字典序排序。
3. 同阶精确并列时，排序较前的项取得相应区间增量，后续并列项增量为零。
4. 近似并列只记录稳定性标志，不重新分摊质量。

作者不同版本的代码可能对近似并列设置额外容差 [3]。复现实验时必须固定实现版本与组合顺序。近似并列的单项协同归属可能不稳定，应同时报告按阶汇总的协同和重采样稳定性，不把某一次微小排序差异当作强结论。

## 8. 四来源可复算示例

### 8.1 独立擦除通道：验证 26 项排序与守恒

使用一个合成概率分布验证计算，不把它当作真实推荐结果。这里的 P/C/T/L 是抽象来源标签，独立擦除假设不代表真实数据中的 Category 与 POI 相互独立。

目标 Y 是公平二值变量。P、C、T、L 分别以 0.4、0.3、0.1、0.2 的概率独立透露 Y，否则显示缺失状态。记透露概率为：

$$
(\rho_P,\rho_C,\rho_T,\rho_L)
=
(0.4,0.3,0.1,0.2).
$$

任一子集的信息量为：

$$
\iota_A(y)
=
1-\prod_{m\in A}(1-\rho_m).
$$

| 阶数 | 各子集 specific information，单位 bit |
| --- | --- |
| 一阶 | P：0.4；C：0.3；T：0.1；L：0.2 |
| 二阶 | PC：0.58；PT：0.46；PL：0.52；CT：0.37；CL：0.44；TL：0.28 |
| 三阶 | PCT：0.622；PCL：0.664；PTL：0.568；CTL：0.496 |
| 四阶 | PCTL：0.6976 |

单来源排序为 T、L、C、P，得到：

$$
u_P=0.1,\qquad
r_{PC}=0.1,\qquad
r_{PCL}=0.1,\qquad
r_{PCTL}=0.1.
$$

二阶以 0.4 为基准，并按同阶信息从小到大分配：

$$
s_{CL}=0.04,\quad
s_{PT}=0.02,\quad
s_{PL}=0.06,\quad
s_{PC}=0.06.
$$

三阶以 0.58 为基准：

$$
s_{PCT}=0.042,\qquad
s_{PCL}=0.042.
$$

四阶以 0.664 为基准：

$$
s_{PCTL}=0.0336.
$$

其余分量为零。总量：

$$
0.1+0.3+0.18+0.084+0.0336
=
0.6976.
$$

本版将这 26 项全部保存。三个非零冗余项仍然占据自己的原始槽位，不分给 PC、PL 或其他 Expert；只有 U/S 对应可执行 Experts。

### 8.2 确定性类别和位置：允许相应分量为零

再构造一个解析检查：P 均匀取四个两比特状态，目标 Y=P；Category 是 P 的低位，Location 是 P 的高位，Time 是独立公平比特。这是验证元数据关系的抽象例子，不是实际 next-POI 数据。

$$
C=P\bmod 2,
\qquad
L=\lfloor P/2\rfloor,
\qquad
T\perp P.
$$

每个目标事件的单来源信息量分别为：

$$
(\iota_P,\iota_C,\iota_T,\iota_L)=(2,1,0,1)\ \mathrm{bit}.
$$

根据第 7 节的 SURD 排序分配，仅有：

$$
u_P=1\ \mathrm{bit},
\qquad
r_{PCL}=1\ \mathrm{bit},
\qquad
T_y=2\ \mathrm{bit}.
$$

其余 24 项为零，包含 Category 的 Unique 和全部 Synergy。虽然 C 与 L 合在一起可以恢复 P，但本 SURD 的二阶增量要超过最强单来源 P 已经提供的 2 bit，因此 S_CL 仍为零。这个例子同时提醒：SURD 的冗余/协同分配不能直接替换为另一套 PID 定义。附录 A 同时验证这组结果。

## 9. 完整 26 维先验、可靠性与长尾处理

### 9.1 原始分解与完整归一化

原始分解保持 bit 单位，绝不覆盖：

$$
\mathbf d_y^{\mathrm{raw}}\in\mathbb R_{\ge0}^{26}.
$$

使用全部 26 项的总和归一化：

$$
\mathbf r_y^{(26)}
=
\begin{cases}
\mathbf d_y^{\mathrm{raw}}/T_y,
&
T_y>\epsilon_I,\\
\mathbf0,
&
T_y\le\epsilon_I.
\end{cases}
$$

有效行满足：

$$
\mathbf1^\top\mathbf r_y^{(26)}=1.
$$

阈值用于识别数值零，不能借此删除实质性的负数或掩盖守恒失败。

### 9.2 全局先验

先平均原始分解，再归一化：

$$
\mathbf d_{\mathrm{global}}
=
\sum_y\widehat p(y)\mathbf d_y^{\mathrm{raw}}.
$$

$$
\mathbf r_{\mathrm{global}}
=
\frac{
\mathbf d_{\mathrm{global}}
}{
\mathbf1^\top\mathbf d_{\mathrm{global}}
},
$$

只在分母大于数值阈值时计算；否则记录零全局先验。

先平均每行已归一化的分布通常会得到不同结果，不能将它与上述全局互信息分解混用。

### 9.3 长尾收缩仍然保留 26 维

对有足够数值信息的行：

$$
c_y=\frac{n_y}{n_y+\kappa_n},
$$

$$
\bar{\mathbf r}_y
=
c_y\mathbf r_y^{(26)}
+
(1-c_y)\mathbf r_{\mathrm{global}}
\in\mathbb R_{\ge0}^{26}.
$$

若某行没有观测，或总信息不超过数值阈值，使用 global 行作为可解释的回退占位；该行的个体 SURD 引导强度设为零。若 global 也为零，回退行为退化为不使用 SURD 的上下文 Router。

若本地行有效，但 global 的总量低于数值阈值而无法归一化，则直接保留本地归一化行，不与零 global 向量混合。这样用于路由的有效行仍然和为 1；本地与 global 均无效时才使用全零行。

这一步改变的是用于模型的先验，不是原始分解。必须分别保存 raw、normalized、routing 三个 26 维版本，并保留 observed 与 has_information 标志。

### 9.4 绝对信息量与置信权重

只看相对比例会放大很小的信息量。因此定义：

$$
\omega_y
=
\mathbf1[\text{observed}_y]
\mathbf1[T_y>\epsilon_I]
\frac{n_y}{n_y+\kappa_n}
\frac{T_y}{T_y+\kappa_I},
\qquad
0\le\omega_y\le1.
$$

其中，信息强度参数使用 bit 单位。它和样本数收缩都是工程可靠性规则，不是统计置信区间。

本版将该权重同时用于 Router 的 SURD 输入、显式 prior bias 与辅助损失。这样可以降低低样本、低绝对信息量行的显式先验偏置和辅助监督强度。新 POI 或零信息行的推荐损失仍可训练上下文 Router 和残差门；不向其伪造均匀的 SURD teacher。

## 10. 离线完整流程、时间扩展与 Cross-fitting

### 10.1 单张基础 SURD 表的构建

本版只有一个合法参考集 D_ref：固定特征不需要预训练；标准化与 K-means 只读取该参考集的来源历史，然后对同一参考集配对标签计数。不再增加内部的表示拟合/计数两段划分。

1. 根据外层切分确定 D_ref，构造预测时可用的 P/C/T/L 历史、next-POI 标签和真实目标事件 ID。
2. 按预先声明的规则完成对齐、截断、缺失处理与目标事件去重；记录字段、时区、数值单位和候选目录。
3. 只对 D_ref 的四源历史调用 `BasicSourceQuantizer.fit()`，拟合四套标准化参数与 codebook；不传入标签。
4. 冻结量化器，用 `transform()` 将 D_ref 转为 `q_state: [n,4]`，与 `y_index: [n]` 一一配对。
5. 构建同一张五维联合频数表，从它边缘化全部 15 个非空来源子集。
6. 逐目标 POI 计算 specific information，再按第 7 节生成 26 个原始分量。
7. 检查非负性、子集单调性、单来源重建、分解守恒，以及两种条件熵计算的一致性。
8. 以全部 26 项总和归一化，再用真实计数计算全局先验、收缩先验与可靠性。
9. 成对保存量化器检查点与 SURD 缓存，包括 raw/normalized/routing 三套分量、熵、计数范围、配置与版本摘要。

这里不训练 15 个条件概率网络；第 5 节路径 A 的全部概率仍来自同一张联合表。同一参考集拟合量化并计数避免了额外划分带来的样本损耗，但有限样本、数据自适应分箱和稀疏支持仍可能导致估计偏差。外层留出、目标置换、重采样及量化敏感性检查分别检验不同问题，不能把任何单个检查解释为分量无偏的证明。

### 10.2 外层折外先验：每折一套量化器与统计

将推荐模型需要训练的样本划分为外层评分折：

$$
\mathcal D_{\mathrm{train}}
=\bigcup_{k=1}^{K_{\mathrm{fold}}}F_k.
$$

对于普通块折外协议，从排除当前折后的训练范围构造参考集，并按下文的事件级规则 purge：

$$
\begin{aligned}
\mathcal D_{\mathrm{ref}}^{(-k)}
&=\operatorname{Purge}
\left(\mathcal D_{\mathrm{train}}\setminus F_k,\ F_k\right),\\
\left(\mathcal Q^{(-k)},\mathcal R^{(-k)}\right)
&=\operatorname{BuildBasicSURD}
\left(\mathcal D_{\mathrm{ref}}^{(-k)}\right).
\end{aligned}
$$

BuildBasicSURD 是第 10.1 节流程的记号，包含在同一参考集上拟合量化、转换和计数，并非额外要求一个同名 API。当前评分折的目标事件不能作为参考目标进入联合表，也不能以另一个重复窗口的形式进入拟合样本。普通折外协议可能包含评分时点之后的其他训练数据；若实验声称每次只利用过去信息，就应采用下一小节的时间扩展协议。

同一样本的全部候选使用同一张折外统计表，分别读取对应候选 POI 的行：

$$
\bar{\mathbf r}_{i,y}
=\mathcal R^{(-k(i))}[y].
$$

标准化、codebook、联合计数、target_count、global fallback 和 confidence 必须来自同一个参考范围。只将最终计数改为折外、却在全数据上拟合 codebook，不能称为这里规定的折外流程。

每折可独立拟合 K-means，其簇编号只对该折的 codebook 有意义。即使两折都出现状态 0，也不代表同一种历史模式；不能把不同 codebook 下的 q_state 拼接成一张频数表。已经分解完成的 26 个分量则可以按样本选择，因为各表统一采用 P/C/T/L 的分量语义和顺序。候选 POI 目录的行顺序也必须固定；没有支持的目标行按第 9、15 节回退。

### 10.3 默认采用时间扩展，并按真实事件进行 purge

对按时间排序的评分块 F_k，使用其开始预测时点以前已经完成的训练目标事件构造参考集：

$$
\mathcal D_{\mathrm{ref}}^{(k)}
=
\left\{
i\in\mathcal D_{\mathrm{train}}:
t_i^{\mathrm{target}}
<\min_{j\in F_k}t_j^{\mathrm{pred}}
\right\},
$$

$$
\left(\mathcal Q^{(k)},\mathcal R^{(k)}\right)
=\operatorname{BuildBasicSURD}
\left(\mathcal D_{\mathrm{ref}}^{(k)}\right).
$$

同时要求每个参考样本自己的历史只到其预测时刻，并且目标发生在历史之后。若时间戳并列，按第 2.2 节固定的稳定事件次序判断先后；下式中的时间比较按时间戳与稳定次序构成的排序键理解：

$$
\max\operatorname{time}(X_{P,C,T,L,i})
\leq t_i^{\mathrm{pred}}
<t_i^{\mathrm{target}}.
$$

上述时间条件代替内部两段拆分：同一 D_ref 拟合量化并计数，整个参考范围在当前评分块之前。较早事件可以合法地出现在后续样本的历史中；这种已经可见的过去信息不需要从历史里删除。

但切分检查必须基于真实事件 ID 与预测窗口，不能只比较 DataFrame 行号。相同目标 check-in 的重复记录、同一目标对应的多个重叠窗口，不能跨入评分折与参考计数；若采用固定 embargo 或禁止相交窗口的实验协议，应在构造 D_ref 时一并实施。一般块折外还需防止评分折目标事件被较晚参考窗口当成历史特征使用；可删除这些窗口，或按预先声明的轨迹块/事件块整体分组。时间扩展默认协议已经通过目标截止时间排除了这种未来回流，仍需核验去重与边界。

最早的暖启动块可能没有足够参考历史，此时使用零置信度，关闭对应 SURD 辅助约束并训练推荐任务，不用后面的评分数据反向补齐它的统计。可以先预留暖启动历史，再从 3 个评分时间块起步。评分块越多不一定越好：应检查早期表的目标支持度、完整联合状态覆盖，以及折外 prior 与最终表之间的分布变化。

### 10.4 验证、测试与最终训练表

在模型选择规定的训练边界内，使用全部合法训练参考样本拟合最终量化器并计数：

$$
\mathcal D_{\mathrm{ref}}^{\mathrm{final}}
=\operatorname{PrepareReference}(\mathcal D_{\mathrm{train}}),
$$

$$
\left(\mathcal Q^{\mathrm{final}},\mathcal R^{\mathrm{final}}\right)
=\operatorname{BuildBasicSURD}
\left(\mathcal D_{\mathrm{ref}}^{\mathrm{final}}\right).
$$

PrepareReference 表示既定的历史可用性、去重和字段处理，不额外丢弃一段训练样本来拟合离线表示。最终表可以使用整个训练范围；验证和测试固定使用该训练范围产生的量化/统计版本，验证/测试的历史分布不用于重新拟合 scaler 或中心，其目标也不进入计数与可靠性。

验证集可以按实验协议用于选择超参数，测试标签不得参与拟合或调参。线上部署使用当前允许训练边界内的固定版本；如果开展滚动更新实验，应另行声明更新时间、可用数据和重建规则，不能在一次固定表评估中临时吸收验证或测试事件。

每个统计版本至少成对记录量化器检查点、该文件的摘要、固定特征配置、实际簇数、来源顺序、参考范围、候选目录和 SURD 缓存。改变 hash seed、摘要、数值单位、标准化参数或中心，都可能改变随机变量 Q 的含义，必须用对应版本重新转换参考历史并重建 SURD。不能让新 codebook 与旧计数表混用；第 16–17 节给出配对保存与加载方式。

## 11. 在线四来源 Encoder 与 Candidate-aware 表示


### 11.1 四路特征化与独立序列编码

模型接收以下历史张量；B 为 batch 大小，N 为 padding 后的历史长度。

| 张量 | 形状 | 内容 |
| --- | --- | --- |
| `poi_ids` | B × N | 历史 venue_id，POI ID embedding 的查表键 |
| `category_ids` | B × N | 对齐的历史类别编号，C 的类别查表键 |
| `time_features` | B × N × d_T | 历史 timestamp 的来源专属数值特征 |
| `location_features` | B × N × d_L | 历史坐标的来源专属数值特征与缺失标记 |
| `padding_mask` | B × N，bool | 四路共用；True 表示补齐位置，与离线 valid_mask 互补 |

在线 `padding_mask=True` 表示需要屏蔽的位置；离线接口则约定 `valid_mask=True` 表示真实事件，二者通过 `valid_mask = ~padding_mask` 转换。直接调用 PyTorch TransformerEncoder 时，将在线 mask 以 `src_key_padding_mask` 参数传入；候选注意力也按 True=屏蔽处理。

POI 与 Category 使用独立 embedding。Time 可使用以下特征，其中 h 为历史小时、w 为历史星期序号，Delta tau 为历史相邻间隔：

$$
\phi_T(\tau_j)
=
[
\sin(2\pi h_j/24),\cos(2\pi h_j/24),
\sin(2\pi w_j/7),\cos(2\pi w_j/7),
\log(1+\Delta\tau_j),
\mathrm{gap\_valid}_j
].
$$

时间间隔统一成事先声明的单位，例如小时。窗口第一个间隔设为 0，并将 gap_valid 置为 0，以区别于真实零间隔。缺失时间按第 2 节的数据协议处理。

Location 可以用经纬度弧度 lambda、phi 的球面三维编码，拼接仅由相邻历史坐标产生的距离、方向及有效标记：

$$
\phi_L(\ell_j)
=
[
\cos\phi_j\cos\lambda_j,
\cos\phi_j\sin\lambda_j,
\sin\phi_j,
\log(1+\Delta d_j),
\sin b_j,\cos b_j,
\mathrm{coord\_valid}_j,
\mathrm{step\_valid}_j
].
$$

距离使用统一单位，例如公里；第一步或无有效前序坐标时使用 step_valid=0，零位移导致方向未定义时使用约定的零方向向量。缺失坐标不能被当作真实的经纬度 (0,0)。Location 不拼入时间间隔或速度。

四路初始表示为：

$$
\begin{aligned}
H_P^{(0)}&=E_P(\mathbf P)+E_{\mathrm{pos}},\\
H_C^{(0)}&=E_C(\mathbf C)+E_{\mathrm{pos}},\\
H_T^{(0)}&=\mathrm{MLP}_T(\Phi_T)+E_{\mathrm{pos}},\\
H_L^{(0)}&=\mathrm{MLP}_L(\Phi_L)+E_{\mathrm{pos}}.
\end{aligned}
$$

四路均投影到共同维度 d，并使用同一事件位置索引。来源 Encoder 相互独立：

$$
H_m
=
\operatorname{TransformerEncoder}_m
\left(H_m^{(0)},\mathrm{padding\ mask}\right),
\qquad
H_m\in\mathbb R^{N\times d},
\qquad
m\in\{P,C,T,L\}.
$$

在到达组合 Experts、Router 和 Base 之前，不先把四个来源融合成一个 token 再复制为四路。Unique Expert 的历史输入因此只来自对应来源；共享的位置索引与合法的 padding mask 不携带用户 ID。

一次预测的 Encoder 只能读取给定历史前缀。若批量实现同时对多个前缀位置预测，应使用相应因果 mask。历史间隔与位移使用向后差分，不使用下一事件生成中心差分或未来平滑特征。

### 11.2 候选表示

基础版本：

$$
q_y=E_{\mathrm{POI}}(y)\in\mathbb R^d.
$$

若静态属性可用，可扩展为：

$$
q_y
=
W_q
\left[
E_{\mathrm{POI}}(y);
E_{\mathrm{cat}}(c_y^{\mathrm{cat}});
\phi_{\mathrm{geo}}(\operatorname{coord}(y))
\right].
$$

类别记号与可靠性系数是不同变量。候选的类别信息与历史 Category 序列职责不同：前者描述正在评分的 y，后者描述已经发生的访问序列。静态属性可用性必须在所有候选上一致处理，不能只给真实正例添加属性。候选表示中同样不加入用户 ID 或用户偏置。

### 11.3 候选查询四个来源

对每个来源：

$$
\begin{aligned}
Q_m^{\mathrm{attn}}(y)&=q_yW_m^Q,\\
K_m^{\mathrm{attn}}&=H_mW_m^K,\\
V_m^{\mathrm{attn}}&=H_mW_m^V,\\
z_m(y)
&=
\operatorname{Softmax}
\left(
\frac{Q_m^{\mathrm{attn}}(y)(K_m^{\mathrm{attn}})^\top}{\sqrt{d_k}}
+
M_{\mathrm{pad}}
\right)V_m^{\mathrm{attn}}.
\end{aligned}
$$

实际可使用多头注意力。每个候选最终获得四个 d 维向量。

注意力矩阵使用 attn 上标，区别于离线离散来源变量 Q 与来源状态数 K。多头结果经过输出投影回到统一的 d 维。

整个 batch 可批量计算：

$$
H_m:\ B\times N\times d,
\qquad
Q_{\mathrm{cand}}:\ B\times M\times d,
\qquad
Z_m:\ B\times M\times d.
$$

Encoder 输出只计算一次并复用；不能为每个候选重新跑四个历史 Encoder。候选查询是在固定 y 下访问某一路历史，允许 q_y 含候选的合法静态属性；它不授权单来源 Expert 读取其他历史来源。

## 12. 仅包含 Unique 与 Synergy 的 15 个 Experts

### 12.1 Unique Experts

四个单来源 Experts 分别处理 POI、Category、Time、Location 历史：

$$
\mathcal E_U
=
\{E_P^U,E_C^U,E_T^U,E_L^U\}.
$$

每个 Expert 只接收对应来源的 candidate-aware 向量：

$$
o_m^U(y)
=
W_{m,2}^U
\operatorname{GELU}
\left(
W_{m,1}^U\operatorname{LN}(z_m(y))
+
b_{m,1}^U
\right)
+
b_{m,2}^U.
$$

初始隐藏层宽度取 2d，输出 d 维。由于候选信息已经通过注意力进入表示，无需给单来源 Expert 拼接其他来源。

“Unique Expert”描述其单来源结构和路由职责，不保证输出向量中只包含严格统计意义上的 Unique 信息。

### 12.2 Synergy Experts

二阶及以上组合为：

$$
\mathcal A_{\ge2}
=
\{
PC,PT,PL,CT,CL,TL,
PCT,PCL,PTL,CTL,
PCTL
\}.
$$

每种组合一个 Expert，共 11 个。PC 表示 POI–Category，CT 表示 Category–Time，CL 表示 Category–Location，其余按同一来源命名规则解释：

$$
\mathcal E_S
=
\{E_A^S:A\in\mathcal A_{\ge2}\}.
$$

先生成共享的来源投影：

$$
v_m(y)=\operatorname{LN}(W_m^S z_m(y)).
$$

对被选中的组合只读取该组合内的历史来源，不将全来源上下文 h^c 拼回 Expert：

$$
p_A(y)
=
\bigodot_{m\in A}v_m(y).
$$

结合乘法交互与稳定的残差项：

$$
c_A^S(y)
=
\operatorname{LN}
\left(
W_A^S p_A(y)
+
\frac{1}{\sqrt{|A|}}
\sum_{m\in A}V_{A,m}z_m(y)
\right).
$$

输出：

$$
o_A^S(y)
=
\operatorname{FFN}_A^S(c_A^S(y))
\in\mathbb R^d.
$$

乘法项表达多来源交互，归一化与残差缓解高阶乘积的数值衰减。残差也意味着 Expert 可以利用主效应；不能仅凭乘法结构就声称其输出是纯统计 Synergy。

### 12.3 Expert 固定顺序

| Expert 索引 | 类型与组合 |
| --- | --- |
| 0–3 | U_P、U_C、U_T、U_L |
| 4–9 | S_PC、S_PT、S_PL、S_CT、S_CL、S_TL |
| 10–13 | S_PCT、S_PCL、S_PTL、S_CTL |
| 14 | S_PCTL |

本版没有 Redundancy Expert，没有 shared-private reconstruction head，也不保留原方案中为冗余 Expert 设置的 reconstruction loss。

## 13. 26 维 SURD-guided Router

### 13.1 保留全向量，读取直接对应 Expert 的坐标

对当前候选，读取完整 26 维先验：

$$
\bar{\mathbf r}_y
=
[
\bar{\mathbf r}_y^U;
\bar{\mathbf r}_y^R;
\bar{\mathbf r}_y^S
].
$$

其中三个分组分别有 4、11、11 个坐标。

为计算对 15 个实际 Experts 的直接支持度，选择 U/S 坐标：

$$
\mathbf v_y
=
[
\bar{\mathbf r}_y^U;
\bar{\mathbf r}_y^S
]
\in\mathbb R_{\ge0}^{15}.
$$

定义：

$$
\alpha_y=\mathbf1^\top\mathbf v_y,
\qquad
\rho_y^R=\mathbf1^\top\bar{\mathbf r}_y^R.
$$

对有效归一化行：

$$
\alpha_y+\rho_y^R=1.
$$

这里的 15 维切片只是实际 Expert 的对应分数，**不替代 26 维缓存，不把冗余项映射进去，也不重新归一化到和为 1**。

### 13.2 完整 26 维辅助上下文

构造当前上下文：

$$
h_{i,y}^c
=
\operatorname{MLP}_c
\left(
[z_P(y);z_C(y);z_T(y);z_L(y);q_y]
\right)
\in\mathbb R^d.
$$

对完整先验编码：

$$
a_y^{\mathrm{SURD}}
=
\operatorname{MLP}_{26}
\left(
\omega_y\operatorname{sg}(\bar{\mathbf r}_y)
\right)
\in\mathbb R^{d_r}.
$$

sg 表示停止梯度，其前向/反向含义与 PyTorch 对应写法见第 13.5 节。初始取辅助维度 64，MLP 结构为 26 → 64 → 64。

初始上下文 MLP 采用 5d → 2d → d，Base MLP 也采用 5d → 2d → d。隐藏层使用 GELU 与 Dropout，最终特征层输出 d 维。

Router 输入：

$$
h_{i,y}^{\mathrm{route}}
=
[
h_{i,y}^c;
q_y;
a_y^{\mathrm{SURD}};
\omega_y\rho_y^R;
\omega_y\alpha_y
].
$$

因此，全部 11 个冗余分量保留各自身份，直接影响 Router；冗余总质量另外作为一个清晰的辅助特征。

在初始配置下，Router 输入宽度为 2d + 64 + 2。路由 MLP 采用“输入宽度 → d → 15”，门控 MLP 采用“输入宽度 → d/2 → 1”；这些宽度只决定网络容量，不改变 26 维统计先验。

### 13.3 显式 U/S prior bias

为避免极小 U/S 质量被除法放大，使用单调、有限的偏置：

$$
b_{y,e}^{US}
=
\log(1+15v_{y,e}).
$$

最终 15 个 logits 为：

$$
\boxed{
\boldsymbol\ell_{i,y}
=
\operatorname{MLP}_{\mathrm{route}}
\left(
h_{i,y}^{\mathrm{route}}
\right)
+
\lambda_{\mathrm{prior}}\omega_y
\log(\mathbf1+15\mathbf v_y)
}
$$

这里 log 使用自然对数，只是神经 logits 的变换，不改变前面以 bit 保存的信息量。

归一化：

$$
\boldsymbol\pi_{i,y}
=
\operatorname{Softmax}
\left(
\frac{\boldsymbol\ell_{i,y}}{\tau}
\right)
\in\mathbb R^{15}.
$$

这一步是对**实际 Expert logits**进行 softmax，不是把 SURD 26 维表改成 15 维统计分解。

Router 的可学习 MLP 本身已读入完整 prior 编码，因此其输出不能简单称为“完全不含 SURD 的上下文分数”。第 23 节分别记录 learned logits 和最后显式相加的 prior bias，两者共同决定 dense 路由概率。

### 13.4 Top-K 稀疏执行

选择：

$$
\mathcal K_{i,y}
=
\operatorname{TopK}
(\boldsymbol\pi_{i,y},K).
$$

归一化被选中的模型权重：

$$
\widetilde\pi_{i,y,e}
=
\frac{
\pi_{i,y,e}
}{
\sum_{j\in\mathcal K_{i,y}}\pi_{i,y,j}
},
\qquad
e\in\mathcal K_{i,y}.
$$

只执行被选中的 Experts：

$$
h_{i,y}^{\mathrm{MoE}}
=
\sum_{e\in\mathcal K_{i,y}}
\widetilde\pi_{i,y,e}\,o_e(y).
$$

Top-K 索引选择本身离散；主任务梯度通过被选中的权重与 Expert 输出传播。辅助路由损失在截断前的 dense softmax 上计算，因此所有 15 个 logits 都能获得相关监督。

### 13.5 sg 的定义与梯度边界

sg 是 stop-gradient，即保留前向数值、停止向该输入传播梯度：

$$
\operatorname{sg}(x)=x\quad\text{（前向）},
\qquad
\frac{\partial\,\operatorname{sg}(x)}{\partial x}=0
\quad\text{（反向）}.
$$

PyTorch 中对应 `Tensor.detach()` [4]：

```python
fixed_prior = prior26.detach()
prior_embedding = prior_mlp(confidence[..., None].detach() * fixed_prior)
```

离线 SURD 表不会被推荐损失修改；prior_mlp 的参数仍参与计算图，因此仍可训练。不能把 prior_mlp 的输出再无条件 detach，否则这个 MLP 就无法从推荐目标学习。

当前统计版本的固定特征配置、摘要规则、标准化、codebook 与计数保持冻结；在线推荐 Encoder、Experts、Router、prior_mlp 和 gate 联合训练。若更新离线摘要或量化器，需重新生成离散状态和整张 SURD 表，不能靠取消 detach 对离散计数过程在线反传。

若表通过不需要梯度的 registered buffer 读取，detach 可能在数值上是多余的；公式仍保留 sg，以明确固定统计 teacher 和可学习模型的边界。

当 confidence=0 时，候选特定的统计输入被关闭；prior_mlp(0) 仍可能因为可学习偏置输出一个共享常量，这不携带该候选的 SURD 结构。因此“关闭个体先验指导”不要求这个 MLP 的输出数值严格为零。

不要用 `torch.no_grad()` 包围在线 prior_mlp 的整个前向；它应只应用于离线冻结统计/特征提取。否则会连可学习 MLP 参数的梯度一起关闭。

## 14. Redundancy 辅助的 Base/MoE 残差门与预测层

### 14.1 Base 分支

Base 分支接收全部四个来源与候选表示：

$$
h_{i,y}^{\mathrm{base}}
=
\operatorname{MLP}_{\mathrm{base}}
\left(
[z_P(y);z_C(y);z_T(y);z_L(y);q_y]
\right).
$$

$$
s_{\mathrm{base}}(i,y)
=
w_{\mathrm{base}}^\top h_{i,y}^{\mathrm{base}}
+
b_{\mathrm{base}}.
$$

Base 可以利用来源之间共享或重复的预测证据，因此没有 Redundancy Expert 并不意味着删除了这些有用信息。

### 14.2 可学习残差门

门控 logit 与实际门为：

$$
a_{i,y}^{\mathrm{gate}}
=
\operatorname{MLP}_{\mathrm{gate}}
\left(
h_{i,y}^{\mathrm{route}}
\right),
\qquad
g_{i,y}
=
\sigma(a_{i,y}^{\mathrm{gate}}).
$$

它读取当前 P/C/T/L 历史上下文及完整先验，因此同一个候选 POI 对不同历史前缀可以有不同的 MoE 残差强度；这种差异不依赖 user_id。

MoE 分数：

$$
s_{\mathrm{MoE}}(i,y)
=
w_{\mathrm{MoE}}^\top h_{i,y}^{\mathrm{MoE}}
+
b_{\mathrm{MoE}}.
$$

最终预测：

$$
\boxed{
s_\theta(y\mid\mathcal X_i)
=
s_{\mathrm{base}}(i,y)
+
g_{i,y}s_{\mathrm{MoE}}(i,y)
}
$$

### 14.3 Redundancy 在本版中的三项职责

1. 11 个冗余坐标完整进入辅助上下文，帮助 Router 识别不同来源之间的信息重复结构。
2. 冗余总质量通过辅助目标影响 Base/MoE 残差门。
3. 完整归一化后，U/S 支持度自然保留未被冗余占据的质量，避免人为把很小的 U/S 放大。

将 U/S 信息质量用作残差门目标，是本文的建模假设。信息量不在数学上等于神经权重，也不能保证 Base 分支只携带冗余信息。门是可学习的，推荐损失允许它偏离先验，不强制设置为某个离线数值。

## 15. 联合训练目标

### 15.1 推荐交叉熵

对每条样本：

$$
\mathcal L_{\mathrm{rec},i}
=
-\log
\frac{
\exp(s_\theta(y_i^+\mid\mathcal X_i))
}{
\sum_{y\in\mathcal C_i}
\exp(s_\theta(y\mid\mathcal X_i))
}.
$$

小词表可使用全量 softmax。大词表可以使用声明清楚的采样推荐目标；这只影响在线推荐训练，不改变离线概率表，也不能把采样候选 softmax 用作路径 A 的概率估计。

### 15.2 辅助损失为何同时控制门与路由

本节解决两个问题：MoE 残差应获得多强的支持，以及其内部应偏向哪些 Experts。记完整先验的 U/S 切片为 v、总质量为 alpha、置信权重为 omega：

$$
\mathcal L_{\mathrm{SURD},i,y}
=
\omega_y\left[
\operatorname{BCEWithLogits}
\bigl(a_{i,y}^{\mathrm{gate}},\operatorname{sg}(\alpha_y)\bigr)
-\sum_e\operatorname{sg}(v_{y,e})\log\pi_{i,y,e}
\right].
$$

第一项用 alpha 作门的软目标：冗余较多时更倾向 Base，Unique/Synergy 较多时更支持 MoE。这是本文的建模假设，不是信息论保证最优神经门必等于信息比例。

第二项提供 Expert 偏好，同时保留原 U/S 质量：

$$
-\sum_e v_{y,e}\log\pi_{i,y,e}
=
\alpha_y H\left(\frac{\mathbf v_y}{\alpha_y},
\boldsymbol\pi_{i,y}\right),
\qquad\alpha_y>0.
$$

因此 alpha=0.001 的目标只得到质量为1时千分之一的路由监督；实现无须计算除法。若只有0.001的U_P和0.999的全来源冗余，强行把U/S切片变成U_P=1，会抹掉几乎全为冗余的信息结构。

这个例子有合法分布：P完全透露公平二值Y；C/T/L共用同一个擦除通道，以0.999概率透露Y，其余时共同擦除。其SURD为U_P=0.001、R_PCTL=0.999，其余为零。

对 gate logit 和 Router logit 的梯度分别为：

$$
\frac{\partial\mathcal L_{\mathrm{SURD}}}
{\partial a^{\mathrm{gate}}}
=
\omega_y(g_{i,y}-\alpha_y),
$$

$$
\frac{\partial\mathcal L_{\mathrm{SURD}}}
{\partial\ell_{i,y,e}}
=
\frac{\omega_y}{\tau}
\left(\alpha_y\pi_{i,y,e}-v_{y,e}\right).
$$

因此路由概率对齐的是条件目标v/alpha，监督强度由alpha和omega控制；不能把未归一化v直接称为pi的概率目标。

| 情况 | 门的辅助目标 | 路由辅助作用 | 设计目的 |
| --- | --- | --- | --- |
| 较强Unique/Synergy | alpha较大 | 偏向对应Experts | 给稀疏路由提供统计起点 |
| 纯冗余 | alpha=0 | 严格为零 | 不人为制造U/S teacher |
| U/S很小 | alpha很小 | 梯度相应减小 | 不放大微弱信息质量 |
| 低样本或低绝对信息 | 保持比例，降低omega | 显式指导减弱 | 降低不稳定统计的控制力 |
| 未观测或零信息 | omega=0 | 两项关闭 | 让推荐目标学习上下文路由 |

还可构造一个解释辅助损失的分支概率向量：

$$
\mathbf t_y=[1-\alpha_y;\mathbf v_y],
\qquad
\mathbf p_{i,y}=[1-g_{i,y};g_{i,y}\boldsymbol\pi_{i,y}].
$$

有效先验行上两者各自和为1，交叉熵恰等于门BCE加路由项。这个“Base加15Experts”的16分支解释不改变SURD维度，**不替换26维缓存**。

推荐损失允许模型偏离软目标；统计prior描述参考分布，当前历史、表示能力和推荐目标仍会影响门与路由。

### 15.3 纯冗余、弱信息与未知目标的行为

| 情况 | 处理 |
| --- | --- |
| 用于路由的有效先验全部是 Redundancy | U/S 总质量为零，路由监督严格为零；门控监督推动模型更多使用 Base |
| U/S 占比很小 | 显式 prior bias 与路由辅助梯度都较弱，不将其重新放大 |
| 全部绝对信息量很小 | 可靠性权重降低，避免只看相对比例 |
| 已观测但事件信息为数值零 | 记录 has_information=false，个体 SURD 指导关闭；推荐目标仍正常训练 |
| 未观测 POI | 记录 observed=false，使用 global 回退占位但关闭个体 SURD 指导 |
| global 也没有信息 | 使用零先验输入，prior_mlp 可保留共享偏置；由上下文驱动 U/S MoE 与 Base |
| 局部纯冗余但长尾收缩引入 global U/S | 这部分 U/S 来自回退，是稳健化设计，不属于该 POI 原始分解 |

纯冗余时门控目标为零不等于模型输出门严格为零。除非另外明确启用硬回退，本版保留推荐损失修正离线先验的能力。

### 15.4 软负载约束为何比较批次整体质量

15.2逐样本提供指导；15.4关注一个batch的整体趋势。它试图避免模型长期集中到少数Experts，而当前batch的可靠先验支持其他组合。

对J个有效样本—候选对：

$$
f_e=\frac1J\sum_{(i,y)}\omega_y g_{i,y}\pi_{i,y,e},
\qquad
f_e^\star=\frac1J\sum_{(i,y)}\omega_y v_{y,e},
$$

$$
\mathcal L_{\mathrm{balance}}
=\sum_{e=1}^{15}(f_e-f_e^\star)^2.
$$

f是置信度加权后的MoE软责任质量，f-star是同一批样本与候选的先验目标。它们保留总质量差异：

$$
\sum_e f_e
=\frac1J\sum_{(i,y)}\omega_y g_{i,y},
\qquad
\sum_e f_e^\star
=\frac1J\sum_{(i,y)}\omega_y\alpha_y.
$$

例如 batch 先验目标为 U_P=0.40、S_CT=0.30，其余项为零。为便于展示，忽略模型其他 13 路的极小软质量，假设模型在 U_P、S_CT 上的质量约为 0.65、0.05，则：

$$
\mathcal L_{\mathrm{balance}}
\approx(0.65-0.40)^2+(0.05-0.30)^2
=0.125.
$$

该项鼓励减少过量U_P、增加不足S_CT；不要求两者各占0.5，更不要求其他13个Experts分到流量。先验确实支持某个Expert时，也不为均匀化而反对集中。

它是软矩量匹配。若每个样本的15.2都达到teacher最优点，整体约束也自然满足；作用是在有限训练、不同损失竞争和batch波动下提供弱正则，不是额外的因果识别步骤。建议保持较小权重并做消融。

必须说明：

1. g×pi是截断前软质量，**不是真实派发次数**。当前模型即使gate很小，仍计算Top-K Experts；低gate缩小残差，不自动省计算。
2. 候选采样改变batch中的POI频率。f和f-star须用同一候选集合、mask和权重；也不能直接当作按全局p(y)加权的分解。
3. omega是固定权重，f-star应detach；g和pi保留梯度。未知或不可靠候选不应因为负载正则被强行赋予统计意义。
4. 真正的系统吞吐需另外检查派发次数、容量溢出、等待时间；如需改变执行负载，应单独设计capacity/dispatch策略。

### 15.5 最终目标与优化范围

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{rec}}
+
\lambda_{\mathrm{SURD}}\mathcal L_{\mathrm{SURD}}
+
\lambda_{\mathrm{balance}}\mathcal L_{\mathrm{balance}}
}
$$

三个强度参数分别负责不同功能：

| 参数 | 作用 |
| --- | --- |
| lambda_prior | 在线 logits 中显式 SURD prior bias 的强度 |
| lambda_SURD | 门控与路由辅助监督的权重 |
| lambda_balance | 软负载约束的权重 |

不要将 prior bias 系数与辅助损失系数混用为同一个含义不清的参数。

在线 Encoder、candidate attention、Base、15 个 Experts、Router 与 gate 联合训练；离线固定哈希、数值特征规则、摘要、标准化、codebook 和全部统计表保持冻结。

## 16. PyTorch：离线分解保存，再作为在线 Router 输入

### 16.1 离线模块的输入与保存结果

附录 D 的 `basic_source_quantizer.py` 完成固定特征摘要、四个独立 K-means 的拟合及最近中心分配；附录 C 的 `surd_offline_torch.py` 接收它输出的四来源整数状态并完成 SURD。两者仅依赖 PyTorch 与标准库。统计函数内部不重新拟合特征或量化器。

接口中的 `target_rows` 就是第 4 节的 `y_index`。`q_states` 的每一行只含四个来源簇编号，目标 POI 行号由 `target_rows` 单独传入；26 维结果在统计分解之后才产生，不能把 193 维摘要或中心向量直接当作 `q_states`。具体数据形态见第 4.6.2 节。

| 输入 | 形状/类型 | 含义 |
| --- | --- | --- |
| `q_states` | n × 4，整数 tensor | 每个真实预测事件的 P/C/T/L 离散摘要状态 |
| `target_rows` | n，整数 tensor | 目标 POI 在统一 poi_ids 词表中的行号 |
| `poi_ids` | 长度 V 的唯一 int/string 列表 | 外部 POI ID 到缓存行号的共同映射 |
| `manifest` | JSON 可表示的字典 | 基础特征配置、配套量化器摘要、参考时间范围、去重与字段处理说明 |

实现先用 torch.unique 建立唯一的五维稀疏联合表，再从这张表投影、汇总 15 个边缘表。统计计算使用 CPU float64，不对 SURD 分解反向传播。这是清晰的统计参考实现，不声称已针对大规模 GPU 计数优化。

正常数据中每个真实目标事件只出现一次。默认每行权重为 1，也支持整数频数压缩；增强窗口先按真实目标去重。附录的 probability_mass 模式仅用于解析示例，会关闭样本量置信度，避免将概率质量误当作样本数。

```python
import hashlib
from pathlib import Path
from surd_offline_torch import build_offline_surd, save_surd_table

# qz, q_state and quantizer_path are produced in section 4.8.
# y_index: one next-POI catalog row per reference event, in the same row order.
# target_catalog: the same ordered POI ID list used in every fold.
stats = build_offline_surd(
    q_states=q_state,               # [n_ref, 4], P/C/T/L
    target_rows=y_index,            # [n_ref], catalog row index
    poi_ids=target_catalog,
    manifest={
        "feature_method": "basic_fixed_features_v1",
        "feature_manifest": qz.config,
        "quantizer_manifest": {
            "file": quantizer_path.name,
            "sha256": hashlib.sha256(quantizer_path.read_bytes()).hexdigest(),
            "config": qz.config,
        },
        "split_manifest": {
            "reference_interval": reference_interval_iso,
            "outer_fold": fold_id,
        },
        "event_weight": "one_per_real_target_event",
    },
)
cache_path = Path("offline_cache") / f"surd_basic_fold_{fold_id}.pt"
cache_path.parent.mkdir(parents=True, exist_ok=True)
save_surd_table(stats, cache_path)

assert stats["raw_surd"].shape == (len(target_catalog), 26)
assert stats["routing_prior"].shape == (len(target_catalog), 26)
# H_Y and H_cond are in bits, from the same joint count table.
print(stats["H_Y"], stats["H_cond"])
```


保存的字典包含完整 26 维 raw/normalized/routing 三套量、15 维 specific information、target_count、confidence、global prior、未知/零信息标记、联合计数及 H_Y/H_cond。`raw_surd` 用于解释统计结果；在线默认读取经过可靠性处理的 `routing_prior`，两者不能混称。

### 16.2 载入冻结表并按候选查询

```python
import torch
from torch import nn
from surd_offline_torch import FrozenSURDTable, gather_fold_priors

# Initialize once. These are buffers, not trainable parameters.
fold_tables = nn.ModuleList([
    FrozenSURDTable.from_file(path) for path in fold_cache_paths
]).to(device)
assert all(t.manifest.get("feature_method") == "basic_fixed_features_v1"
           for t in fold_tables)

# External POI IDs must first be mapped to catalog row indices.
# This illustration assumes padded rectangular rows; use a separate validity mask.
row_of = fold_tables[0].row_of
candidate_rows = torch.tensor([
    [row_of.get(poi_id, -1) for poi_id in candidates]
    for candidates in candidate_poi_id_rows
], dtype=torch.long, device=device)

prior26, confidence = gather_fold_priors(
    fold_tables, batch.fold_id, candidate_rows
)
# prior26: [B, M, 26]; confidence: [B, M].
# Pass candidate_rows (catalog indices) as batch.candidate_rows.
# Section 16.3 looks up prior26/confidence separately from those indices.

# Validation/test: use the final strictly constructed training-scope table.
final_table = FrozenSURDTable.from_file(final_cache_path).to(device)
assert final_table.manifest.get("feature_method") == "basic_fixed_features_v1"
assert final_table.poi_ids == fold_tables[0].poi_ids
prior26_eval, confidence_eval = final_table(candidate_rows)
```

所有折的 target_catalog 顺序必须一致，混合折查表函数会检查它。fold_id 只选择该样本允许使用的统计表，不作为可学习 Router 特征。每个候选都读取自己的行，不能用正例行覆盖整个候选集合。

行号 -1 明确表示该 POI 不在缓存词表，返回 global prior 占位并令 confidence=0；其他非法索引报错。词表内但计数集未观测的 POI 也关闭个体指导。-1 只用于 SURD 查表，不能直接传给 nn.Embedding；候选 Encoder 使用自己的合法 UNK/PAD 编号。未知 POI 与候选 padding 不同，padding 必须另外用 candidate_mask 排除。是否能给未知 POI 评分仍取决于候选表示和召回支持。

torch.load 使用 weights_only=True，冻结表用 register_buffer 保存参数外张量 [5]。磁盘统计保留 float64，在线默认转 float32；`.to(device)` 会移动 buffer，优化器不会更新它们。统计解释和绘图优先读取磁盘 raw 值，避免把在线浮点格式当作另一个估计版本。

### 16.3 四序列前向、SURD Router 与联合损失

下面保留第 11–15 节的网络接口，用实际载入的 prior26/confidence 贯穿前向和损失。各来源 Encoder、候选注意力和稀疏 dispatch 是架构模块接口，因此这一段是集成示意；可直接运行的统计/保存/加载模块见附录 C。

```python
import torch
import torch.nn.functional as F
from surd_offline_torch import gather_fold_priors

# fold_tables was loaded once at setup, not rebuilt in a training batch.
prior26, confidence = gather_fold_priors(
    fold_tables, batch.fold_id, batch.candidate_rows
)
prior26, confidence = prior26.detach(), confidence.detach()
candidate_mask = batch.candidate_mask.bool()  # [B, M], True = a real candidate
labels = batch.target_candidate_index        # [B], position of the positive POI
assert candidate_mask.any(dim=1).all()
assert candidate_mask.gather(1, labels[:, None]).all()
pair_mask = candidate_mask.to(prior26.dtype)
J = pair_mask.sum().clamp_min(1.0)

# Four aligned histories only; user_id/trail_id are not model features.
SOURCE_ORDER = ("P", "C", "T", "L")
history = {
    "P": batch.poi_ids,              # [B, N]
    "C": batch.category_ids,         # [B, N], same event positions
    "T": batch.time_features,        # [B, N, d_T]
    "L": batch.location_features,    # [B, N, d_L]
}
mask = batch.padding_mask.bool()    # [B, N], True = padding; offline valid_mask = ~mask
assert all(x.shape[:2] == mask.shape for x in history.values())
assert (~mask).any(dim=1).all()       # This online illustration assumes nonempty histories.

H = {
    m: source_encoder[m](source_project[m](history[m]), src_key_padding_mask=mask)
    for m in SOURCE_ORDER
}
candidate = candidate_encoder(batch.candidate_poi_ids, candidate_catalog)
source_representations = {
    m: candidate_attention[m](candidate, H[m], padding_mask=mask)
    for m in SOURCE_ORDER
}
z = source_representations          # each z[m]: [B, M, d]
full_context = torch.cat([z[m] for m in SOURCE_ORDER] + [candidate], dim=-1)
ctx = context_mlp(full_context)     # 5d -> 2d -> d
base_score = base_head(base_mlp(full_context)).squeeze(-1)

# prior26: [B, M, 26], detached, fold-specific during training
# confidence: [B, M]
# ctx: [B, M, d], candidate: [B, M, d]

unique_mass = prior26[..., :4]
redundant_mass = prior26[..., 4:15]
synergy_mass = prior26[..., 15:26]

us_mass = torch.cat([unique_mass, synergy_mass], dim=-1)  # 15
alpha = us_mass.sum(dim=-1)             # no U/S renormalization
rho_r = redundant_mass.sum(dim=-1)

prior_embedding = prior_mlp(
    (confidence[..., None] * prior26).detach()
)

router_input = torch.cat(
    [
        ctx,
        candidate,
        prior_embedding,
        (confidence * rho_r).detach()[..., None],
        (confidence * alpha).detach()[..., None],
    ],
    dim=-1,
)

router_logits = router_mlp(router_input)  # [B, M, 15]
router_logits = (
    router_logits
    + lambda_prior
    * confidence[..., None]
    * torch.log1p(15.0 * us_mass).detach()
)

log_pi = torch.log_softmax(router_logits / temperature, dim=-1)
pi = log_pi.exp()

gate_logit = gate_mlp(router_input).squeeze(-1)  # [B, M]
gate = torch.sigmoid(gate_logit)

# Actual dispatch uses top-k; losses use the dense pre-top-k pi.
# With only 15 experts, stable sorting defines reproducible exact ties.
# This matches the case-study tie convention; no extra jitter is added.
top_index = torch.argsort(pi, dim=-1, descending=True, stable=True)[..., :top_k]
top_prob = pi.gather(-1, top_index)
top_weight = top_prob / top_prob.sum(dim=-1, keepdim=True)
moe_repr = dispatch_selected_us_experts(
    source_representations, top_index, top_weight,
    candidate_mask=candidate_mask,
)

score = base_score + gate * moe_head(moe_repr).squeeze(-1)

gate_loss = F.binary_cross_entropy_with_logits(
    gate_logit, alpha.detach(), reduction="none"
)
route_loss = -(us_mass.detach() * log_pi).sum(dim=-1)
surd_loss = (pair_mask * confidence * (gate_loss + route_loss)).sum() / J

actual_soft_mass = (
    pair_mask[..., None] * confidence[..., None] * gate[..., None] * pi
).sum(dim=(0, 1)) / J
target_soft_mass = (
    pair_mask[..., None] * confidence[..., None] * us_mass.detach()
).sum(dim=(0, 1)) / J
balance_loss = ((actual_soft_mass - target_soft_mass) ** 2).sum()

recommendation_loss = F.cross_entropy(
    score.masked_fill(~candidate_mask, -torch.inf), labels
)

loss = (
    recommendation_loss
    + lambda_surd * surd_loss
    + lambda_balance * balance_loss
)
```


source_project 包含来源 embedding/MLP 与位置编码。dispatch_selected_us_experts 按第 12.3 节索引派发：单源只读本源，组合只读其成员，并跳过无效候选。完整上下文供 Router、Base 和 gate 使用。

上面是给定候选集合上的 softmax 交叉熵；若使用 sampled softmax 等其他目标，需单独说明采样与校正。训练正例必须位于有效候选集合中，labels 是正例在本行候选中的位置。

辅助损失只在有效样本—候选对上求均值。Top-K 用于实际执行，路由监督与 15.4 的软统计使用截断前 dense 概率。低 gate 会减弱 MoE 分数及软责任质量，但当前实现仍执行选中的专家；没有另行实现硬跳过，就不能声称低 gate 已减少计算量。

### 16.4 实现边界与检查

附录 C 自检覆盖七个解析分布、两种条件熵计算、保存/加载、未知候选和混合折 batch。附录 D 的固定特征与量化器已通过合成历史到 SURD 缓存再到冻结查表的连通检查，也检查来源隔离、padding、常量特征和保存/加载。NumPy 参考实现仍保留在附录 A，可独立核对。解析概率质量示例不产生伪造的样本量置信度；这些检查不代表真实推荐效果。

离线表只针对所记录的特征、摘要与 codebook 版本有意义。量化器检查点和 SURD 表成对保存；重拟合 codebook 后旧表必须重算。detach 不会替数据工程完成时间切分，合法参考集与外层评分块的边界必须在进入 build_offline_surd 前建立并记录。在线按候选读取已经保存的 POI 行，不需要每个 batch 重新计算离线摘要或 SURD。

## 17. 保存格式、数据版本与训练接口

### 17.1 必须缓存的内容

下表区分统计函数直接生成的 tensor/顺序字段与调用方提供的工程元数据。source_fields、feature_manifest、quantizer_manifest、preprocessing_manifest、split_manifest 作为 manifest 的子项或检查点引用传入；附录 C 原样保存 manifest，不自动推断预处理与量化版本。

| 字段 | 内容 |
| --- | --- |
| schema_version | PyTorch 模块固定为 PCTL_TORCH_SURD_v2，并检查来源/分量/专家顺序 |
| source_order | 固定为 P、C、T、L |
| source_fields | venue_id、类别映射、历史 timestamp、历史 latitude/longitude 的字段来源 |
| poi_ids | 每一行的真实目标 POI ID |
| subset_order | 15 个非空子集的固定顺序 |
| atom_order | 26 项 U/R/S 的固定顺序 |
| expert_order | 4 个 Unique 与 11 个 Synergy 的固定执行顺序 |
| specific_information | 每个 POI 的 15 个事件级信息量 |
| raw_surd | 每个 POI 的 26 个 bit 数值 |
| normalized_surd | 以全部 26 项总量统一归一化的 26 维结果 |
| routing_prior | 在完整 26 维上做长尾收缩后的先验 |
| total_information | 完整四源事件级信息量 |
| target_count | 参考集中的目标样本数 |
| confidence | 由样本数与绝对信息量构造的引导强度 |
| observed | 是否在参考集中观测过该 POI |
| has_information | 信息量是否超过数值阈值 |
| near_tie | 同阶是否存在近似并列，需要稳定性关注 |
| global_raw / global_prior | 全局 26 维原始与归一化信息 |
| H_Y / H_cond / H_cond_via_mi | 目标熵、直接条件熵及互信息校验结果 |
| joint_states / joint_count | 唯一五维联合状态及其频数，支持分解审计 |
| feature_method / feature_manifest | basic_fixed_features_v1；固定 SHAKE-256 哈希、P/C salt、维度、摘要规则、长度与位置衰减参数 |
| quantizer_manifest | 配套检查点文件名及 SHA-256、配置和实际每源簇数；检查点内保存标准化均值/尺度与 centers |
| preprocessing_manifest | 类别映射、时区/间隔单位、坐标单位、缺失规则、事件排序和共同裁剪规则 |
| split_manifest | 训练截止时间、合法参考范围、外层评分块、purge 规则及参考目标事件 ID 摘要 |

加载缓存时核验 source_order、subset_order、atom_order 与 schema_version，并核验 feature_method=basic_fixed_features_v1；不能仅因为张量仍有 26 列就接受其他来源方案的缓存。重现统计时还要核验配套量化器文件的 SHA-256。类别映射、字段处理、摘要配置或 codebook 改变后，必须重建相关离散状态与整张概率表。

原始联合频数建议保存为稀疏记录；体积过大时可以保存可重建的参考样本状态表与生成配置。第 16 节与附录 C 使用 torch.save 的 .pt 字典，内容为 tensor 与基础元数据；也可用 NumPy NPZ 加 JSON 保存同等信息。

### 17.2 基础版的完整工程调用

先将附录 C、D 分别保存为 `surd_offline_torch.py` 和 `basic_source_quantizer.py`。下面的函数接收已经按第 2、10 节整理好的**合法参考样本**，完成全部基础离线计算并保存一对文件。数据读取、按事件去重、原始时间/坐标特征化与参考集选择由调用方完成；这些步骤依赖实际数据，不在函数内猜测。

```python
import hashlib
from pathlib import Path

from basic_source_quantizer import BasicSourceQuantizer
from surd_offline_torch import build_offline_surd, save_surd_table

def build_basic_cache(
    history, valid_mask, target_rows, target_catalog,
    *, cache_dir, cache_tag, split_manifest, preprocessing_manifest,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Fit sees only reference histories, never next-POI targets.
    qz = BasicSourceQuantizer(
        n_clusters=4, hash_dim=64, max_length=50, seed=0,
        tau_pos=10.0,
    ).fit(history, valid_mask)
    q_state = qz.transform(history, valid_mask)

    quantizer_path = cache_dir / f"basic_quantizer_{cache_tag}.pt"
    qz.save(quantizer_path)
    quantizer_sha = hashlib.sha256(quantizer_path.read_bytes()).hexdigest()

    # Same reference events, in exactly the same order, now paired with Y.
    stats = build_offline_surd(
        q_states=q_state,
        target_rows=target_rows,
        poi_ids=target_catalog,
        manifest={
            "feature_method": "basic_fixed_features_v1",
            "source_fields": {
                "P": ["venue_id"],
                "C": ["venue_category_id_code"],
                "T": ["timestamp"],
                "L": ["latitude", "longitude"],
            },
            "feature_manifest": qz.config,
            "quantizer_manifest": {
                "file": quantizer_path.name,
                "sha256": quantizer_sha,
                "config": qz.config,
            },
            "preprocessing_manifest": preprocessing_manifest,
            "split_manifest": split_manifest,
            "event_weight": "one_per_real_target_event",
        },
    )
    cache_path = cache_dir / f"surd_basic_{cache_tag}.pt"
    save_surd_table(stats, cache_path)
    return quantizer_path, cache_path
```

对每个评分时间块分别调用此函数。`history` 中的 P/C/T/L、`valid_mask` 和 `target_rows` 必须是同一个参考样本集合、相同顺序；参考目标事件必须在该评分块开始前已经发生，具体时间约束见第 10.3 节。每个目标事件在该表中只贡献一次频数。函数不接收 user_id 或 trail_id 特征，也不在内部拆分额外的预训练集。

最终验证/测试表使用实验允许的全部训练参考样本再调用一次，`cache_tag="final"`。验证/测试样本不加入拟合、计数或可靠性统计。没有参考历史的暖启动块单独使用 confidence=0 的在线分支，不把空参考集传给量化器。

量化器检查点用于复现或重新变换历史；在线第 16.2 节直接加载统计表并按候选 POI 查询，不必加载量化器到推荐训练设备。需要复现离散状态时，先检查配对文件：

```python
import hashlib
from basic_source_quantizer import BasicSourceQuantizer
from surd_offline_torch import FrozenSURDTable

table = FrozenSURDTable.from_file(cache_path)
assert table.manifest["feature_method"] == "basic_fixed_features_v1"
pair = table.manifest["quantizer_manifest"]
assert quantizer_path.name == pair["file"]
assert hashlib.sha256(quantizer_path.read_bytes()).hexdigest() == pair["sha256"]
qz = BasicSourceQuantizer.load(quantizer_path)
assert qz.config == pair["config"]
q_state_reloaded = qz.transform(history, valid_mask)
```

不同折的 codebook 可以不同，但各折 q 状态不能合并计数。跨表对齐的是固定命名的 26 个分量槽位，以及相同顺序的目标 POI 词表。

## 18. 复杂度与大词表实现

### 18.1 离线计算

四个来源只有 15 个非空组合。设稀疏联合表中出现过的不同状态数为 S，则从联合表构造全部边缘表的工作量约为：

$$
O(15S).
$$

若直接从 n 条参考样本生成所有计数，其规模约为：

$$
O(15n).
$$

每个 POI 对 15 个值排序和分配的成本很小。实际主要开销来自历史摘要、量化、计数表内存和数据量。

完整稠密表的单元数是：

$$
|\mathcal V_P|K_{P}K_{C}K_{T}K_{L}.
$$

因此本版采用稀疏存储。它缓解内存问题，但不能解决统计稀疏性：没有重复覆盖的状态，仍然无法靠换一种存储方式获得可靠信息。

### 18.2 在线计算

历史 Encoder 每条样本计算一次。对 M 个候选的注意力与 Router 仍然有成本，不能只报告 MoE 执行量。

若每个 Expert 是宽度与 d 同阶的 FFN，Top-K Expert 主计算约为：

$$
O(BMKd^2).
$$

Candidate-aware attention 约包含：

$$
O(BMNd)
$$

的注意力匹配成本，以及相应投影成本。

大词表下采用两阶段：

1. 独立的轻量召回器先生成 Top-M 候选。
2. TS-MoE 对这些候选执行 candidate attention、完整 26 维先验读取、15 路由 logits 与 Top-K Experts。

召回覆盖率必须单独报告，因为真实目标没有进入候选集合时，重排序模型无法恢复它。

## 19. 推荐的第一版配置

以下是待验证的起始配置，不是实测最优值。

| 模块 | 初始配置 |
| --- | --- |
| 输入来源与顺序 | P：POI，C：Category，T：Time，L：Location；没有 User 来源 |
| 历史最大长度 | 50；同时比较 20、100；四条序列共同截断和 padding |
| ID 字段 | user/trail 只用于数据组织、切分、去重和评估 |
| 在线 POI/Category 表示 | 独立可学习 embedding；PAD/UNK 分开，类别映射固定 |
| 在线 Time/Location 表示 | 各自固定数值特征经可学习 MLP 投影 |
| 在线来源 Encoder | 每源 2 层 Transformer，独立参数 |
| 隐藏维度 | 128 |
| 注意力头数 | 4 |
| Expert FFN | 128 → 256 → 128，GELU，Dropout 0.1 |
| Unique Experts | 4 个 |
| Synergy Experts | 11 个 |
| Redundancy Experts | 0 个 |
| SURD 辅助编码器 | 26 → 64 → 64 |
| Router | 输入当前上下文、候选与完整 SURD 辅助信息；输出 15 logits |
| 残差门 | 同类输入，输出 1 个 gate logit |
| Top-K | 4 |
| 温度 | 1.0 起步 |
| 路径 A 平滑 | 主版本不加独立子集平滑 |
| 离线 P/C 特征 | 每源 64 维固定 SHAKE-256 正负哈希；不同 salt；seed=0 |
| 离线 T/L 特征 | 第 11.1 节的 6 维时间、8 维位置数值特征；不加 MLP |
| 离线默认摘要 | 最后一步、位置衰减均值、普通均值、N/N_max；位置衰减 tau_pos=10 |
| 标准化与 K-means | 每源独立；默认 4 簇，首轮比较 2/4/8，按第 4.5.6 节选择并记录实际簇数；最多 100 次迭代 |
| 拟合/计数协议 | 同一合法 D_ref 的历史拟合量化器，再与 Y 配对计数；外层评分块留出 |
| 外层折 | 预留暖启动参考历史后，从 3 个评分时间块起步；每块只用已发生的参考事件 |
| lambda_prior | 0.1 起步 |
| lambda_SURD | 0.05 起步 |
| lambda_balance | 0.001 起步 |
| 样本数收缩参数 | 20 起步 |
| 绝对信息阈值参数 | 0.05 bit 起步 |
| 优化器 | AdamW；学习率与权重衰减在训练/验证协议内确定 |

辅助损失权重可在初期若干 epoch 从零逐渐增加，先让 Base 与 U/S Experts 获得基本预测能力。所有调参只根据训练与验证结果，不访问测试反馈。

## 20. 实验设计与消融

### 20.1 推荐效果

报告 Recall@K、NDCG@K、MRR@K，并按 Head/Mid/Tail POI、历史长度、类别稀疏程度分组。Top-M 召回场景同时报告候选覆盖率和重排序效果。可用 user_id 做分用户评估或用户级 bootstrap，但这只是统计分组，不表示它进入预测输入。

### 20.2 架构与先验消融

| 变体 | 目的 |
| --- | --- |
| Base only | 衡量不使用 MoE 的基础能力 |
| 普通 U/S MoE | 保留相同 15 Experts、参数规模与训练预算，移除 SURD 输入和损失 |
| U/S MoE + global SURD | 检验逐 POI prior 相比全局 prior 的作用 |
| U/S MoE + per-POI 26D SURD | 完整方法 |
| 去掉 R 辅助 | 将冗余上下文置零，并移除冗余驱动的 gate 监督；说明其余控制不变 |
| 去掉显式 prior bias | 保留完整 26 维上下文，检验 bias 的独立作用 |
| 去掉 SURD 辅助损失 | 保留 prior 输入，检验监督是否必要 |
| 去掉残差门 | 固定 MoE 残差系数，检验门控对冗余与噪声的处理 |
| 仅 Unique Experts | 4 Experts，检验高阶交互增益 |
| Unique + 二阶 Synergy | 10 Experts，检验三阶与四阶协同 |
| 随机或打乱 POI prior | 检验提升是否来自正确的 POI—信息结构对应 |

去掉 R 的消融会改变 gate 目标的解释，应在实验记录中明确处理方式，不能悄悄将 U/S 切片重新归一化后把它当作相同模型。

另外增加与本次输入修改直接相关的对照：

| 数据/来源对照 | 控制方式与问题 |
| --- | --- |
| P/C/T/L 与 P/T/L | 从头训练各自架构与离线表，检验显式类别通道的贡献；三来源版本应使用自身的 11 维分解与 7 个 U/S Experts，不混用四来源缓存 |
| 完整模型推理时屏蔽 C | 仅作为缺失通道鲁棒性检查，不能与从头训练的无 C 模型混称 |
| 最后一步摘要与多步历史摘要 | 检验四条序列的历史信息是否被离线摘要有效利用 |
| 不同来源专属量化粒度/种子 | 检查 Category Unique 或高阶 Synergy 是否依赖某次信息压缩 |
| 元数据确定性检查 | 对固定类别/位置映射与可能的时变属性分别分析，不混淆原始信息与摘要互补 |

三来源对照中有 3 个 Unique、4 个 Redundancy、3 个二阶 Synergy 和 1 个三阶 Synergy，因此是 11 个统计分量、7 个 U/S Experts；上表所需缓存应据此单独构建。四来源主模型始终保持 26 维，不因消融改变其交付定义。

### 20.3 分解可靠性

- 比较不同量化粒度与来源摘要。
- 对用户或时间块做 bootstrap，报告主要分量与按阶协同的稳定性。
- 置换目标标签建立有限样本偏差参照；若保留时间依赖，使用与协议一致的块置换。
- 检查 long-tail 收缩前后的路由差异。
- 比较各折参考集 prior 与最终全训练参考集 prior 的分布偏移；记录每源有效簇数和支持度。
- 检查接近并列的同阶组合，避免过度解读单个 Expert 的细小排名差异。

### 20.4 路由分析

分别统计：

- 15 个 Expert 的真实 Top-K 派发频率。
- gate 在 Head/Mid/Tail 与不同上下文中的分布。
- U/S 总质量、冗余总质量与 gate 的关系。
- 不同候选 POI 的路由差异。
- 原始 26 维结构与模型实际选择的偏离程度。

模型偏离 prior 不一定是错误；可能反映当前上下文差异，也可能反映离散摘要的信息损失。需要结合推荐效果与统计稳定性判断。

## 21. 理论边界与必须保留的实现约束


### 21.1 Category 和 Location 可能由 POI 确定

如果每个 POI 对应固定类别和固定坐标，使用第 2.3 节的原始序列记号，完整历史之间存在：

$$
\mathbf C=f_C(\mathbf P),
\qquad
\mathbf L=f_L(\mathbf P).
$$

这是对固定元数据情形的条件分析，不要求所有数据都必然满足。类别版本变化、坐标噪声、多类别标注或额外实时定位会改变关系，应先核验数据定义；缺失规则也可能使观察到的通道不再是同一个确定映射。

在上述关系严格成立时：

$$
I(Y;\mathbf P,\mathbf C,\mathbf T,\mathbf L)
=
I(Y;\mathbf P,\mathbf T).
$$

事件级 specific information 同样满足：

$$
I_{\mathrm{spec}}(y;\mathbf P,\mathbf C,\mathbf T,\mathbf L)
=
I_{\mathrm{spec}}(y;\mathbf P,\mathbf T).
$$

这是由确定性映射和信息的数据处理关系直接推出的边界。完整 P 已经包含查表恢复 C/L 所需的信息，不能再将它们解释为完整 P 之外的新原始观测。

进一步，若直接对上述**未丢失信息的完整来源**应用第 7 节的 SURD 排序规则，则：

$$
u_C(y)=u_L(y)=0,
\qquad
s_{PC}(y)=s_{PL}(y)=s_{CL}(y)=0.
$$

因为含完整 P 的二来源组合 PT 已经覆盖全部四源信息，上一阶全体组合的最大值会使三阶、四阶协同均为零：

$$
s_A(y)=0,
\qquad
|A|\ge3.
$$

这不要求删除 26 个统计槽位，也不要求删除主模型指定的 15 个 Experts；允许相应原始分量及直接先验支持度为零。该结论基于本 SURD 的全局同阶排序规则，不应直接套到另一套 PID 原子定义。

实际离线统计使用的是分别压缩后的 Q_P、Q_C、Q_T、Q_L。通常不能从原始序列的确定关系进一步推出：

$$
Q_C=\widetilde f_C(Q_P),
\qquad
Q_L=\widetilde f_L(Q_P).
$$

原因是 Q_P 可能丢失某些类别/空间信息，而 Q_C/Q_L 保留了它们。因此量化后可以出现 Category Unique 或较高阶 Synergy；这应解释为**所选离散摘要之间的互补**，不能反推原始 Category 在完整 POI 身份之外必然新增信息。第 8.2 节和附录 A 提供了确定性视图的解析检查。

显式 Category/Location 通道还可能帮助参数共享、类别语义与空间结构学习，但这是待实验验证的模型收益，不是“类别一定有正 Unique”的理论保证。

### 21.2 统计分解不等于神经表示的强制语义

- Unique Expert 的输入结构不保证其输出只包含 Unique。
- 乘法交互不自动等于 Synergy。
- Base 分支不保证只包含 Redundancy。
- 不设置 Redundancy Expert 不等于删除冗余预测信息。
- 原始 26 维分解与稳健化路由先验必须分开存储和解释。
- 观测推荐数据上的信息贡献不能直接解释为干预因果效应。

### 21.3 有限样本与零值

经验联合表严格自洽，不代表估计没有偏差。尤其是目标 POI 很多、历史状态很细时，稀疏计数可能产生虚高信息量。

数值非负性、守恒和单调性证明的是计算符合所选分解规则；它们不能单独证明该分解能泛化。需要参考集边界、量化粒度、折外训练与稳定性实验共同支持结论。

## 22. 实现检查清单

### 数据与离线统计

- [ ] 四个输入均符合预测时可用信息。
- [ ] user/trail 只作数据组织、切分、去重和评估，不进入离线来源特征或在线输入。
- [ ] 标准化和量化器只拟合合法 D_ref 的历史，不接收标签；计数与 D_ref 样本逐行对齐。
- [ ] P/C 使用各自稳定哈希，T/L 使用各自固定数值特征；摘要与量化前不跨源融合。
- [ ] 各折不同 codebook 的 q 状态没有被合并到同一张频数表。
- [ ] P/C/T/L 逐事件对齐，裁剪、padding 和缺失处理保持一致。
- [ ] 历史只到预测时刻；真实下一次的类别、位置和时间未混入输入。
- [ ] Category 映射、时区、间隔单位、坐标单位和缺失标记均已记录。
- [ ] P/C 按位置摘要，Location 不引入需要 Time 的速度。
- [ ] 每个真实目标 check-in 的统计权重定义明确。
- [ ] 四个来源量化器在相应参考范围拟合并冻结；每个 K-means 的输入一行对应一条完整来源历史摘要。
- [ ] 每张 SURD 表共用一套四来源中心表，不按目标 POI 重新聚类；不同 codebook 的状态不混在一张频数表中。
- [ ] `q_state` 为 n×4 的整数簇编号；`y_index` 为 n 个独立目标目录行号，目标 POI 身份没有被聚类合并。
- [ ] POI 目标保留具体 venue_id 身份，不把目标改为 Category。
- [ ] 所有 15 个子集来自同一个联合概率表。
- [ ] 未使用独立子集的任意伪计数平滑。
- [ ] per-POI specific information 先条件平均，再分解。
- [ ] 保存完整 26 项，包含三阶和四阶冗余。
- [ ] unknown 与已观测零信息分别标记。
- [ ] 全局先验由原始分解加权后归一化。
- [ ] 外层评分块不进入相应 codebook、频数、global 和可靠性；严格时间协议只使用过去事件。

### 在线模型与损失

- [ ] 四个独立来源 Encoder 为 P/C/T/L，没有 User Encoder 或用户 ID embedding。
- [ ] 仅 4 Unique + 11 Synergy Experts，索引使用 P/C/T/L 的本版约定。
- [ ] 单源 Expert 只读取对应历史来源，组合 Expert 只读取所属来源子集。
- [ ] 没有 Redundancy Expert 或其 reconstruction loss。
- [ ] 完整 26 维先验进入 Router。
- [ ] U/S 15 维切片不作为替代缓存，不强制归一化为强 teacher。
- [ ] 每个候选读取自己的 POI 行。
- [ ] Base 可以访问全部来源与候选表示。
- [ ] 可学习门与当前上下文相关。
- [ ] 辅助损失使用 dense、截断前的路由概率。
- [ ] Top-K 只执行被选中的 Experts。
- [ ] 真实稀疏派发负载单独记录。
- [ ] 推荐训练与推理使用同一候选打分公式。

### 验证与复现

- [ ] 全复制来源得到纯冗余。
- [ ] 单来源复制目标得到对应 Unique。
- [ ] 二源 XOR 得到对应二阶 Synergy。
- [ ] 四源 XOR 得到四阶 Synergy。
- [ ] 独立目标得到零信息。
- [ ] 四源擦除示例得到总量 0.6976 bit。
- [ ] 确定性类别/位置示例得到 U_P=1、R_PCL=1，其余为零，总量 2 bit。
- [ ] 不因增加 Category 通道而强制其 Unique/Synergy 或派发负载非零。
- [ ] 旧 User 来源的缓存未作为本版 P/C/T/L 先验加载。
- [ ] 每个目标满足 26 项总守恒与四个单来源重建。
- [ ] H_cond 的联合计数计算与 H_Y−MI 一致。
- [ ] PyTorch 保存/加载后保持分量顺序、未知候选行为和冻结值。
- [ ] 量化器和 SURD 缓存记录 basic_fixed_features_v1、配对文件摘要与实际簇数；重拟合后同步重算。
- [ ] 来源隔离、共同右侧 padding、常量特征、小样本与基础流水线连通检查通过。
- [ ] case study 同时记录 raw/routing prior、learned logits、bias、dense/Top-K 权重与 gate。
- [ ] 记录 exact tie、near tie、数值容差与版本信息。
- [ ] 测试集标签没有进入离线表或超参数选择。

## 23. Case study与定性分析：从信息结构到路由决策

### 23.1 分析的证据层次

本章回答：SURD描述了什么，Router怎样使用它，哪些偏离来自当前历史或统计不稳定。

| 层次 | 记录内容 | 能解释什么 |
| --- | --- | --- |
| 原始统计 | raw_surd、信息总量、目标计数 | 参考分布上的bit信息结构 |
| 比例及收缩 | normalized、routing、global、confidence | 归一化、长尾回退及可靠性 |
| 显式偏置 | 每路prior bias | 离线先验对logits的直接加项 |
| 动态路由 | MLP logits、最终logits、dense pi | 当前历史与候选的分配结果 |
| 执行及门控 | Top-K、截断后权重、gate | 选中哪些Expert及残差缩放 |
| 分数贡献 | Expert投影分数、Base、最终排名 | Expert怎样改变候选得分 |

raw_surd以bit计，prior是比例，bias是logit加项，pi是模型分配概率，贡献是推荐分数单位；这些数值没有相等的理由。

$$
\ell_{i,y,e}
=a^{\mathrm{route}}_{i,y,e}
+\lambda_{\mathrm{prior}}\omega_y\log(1+15v_{y,e}),
$$

$$
\pi_{i,y,e}
=\frac{\exp(\ell_{i,y,e}/\tau)}
{\sum_j\exp(\ell_{i,y,j}/\tau)}.
$$

这里 a_route 是 Router MLP 的输出，已经可以包含完整 prior 编码；它不等同于完全不含先验的上下文项。下面合成案例直接指定这个可学习项，便于隔离显式 bias 的数值作用。

即使MLP logits全为零，pi通常也不等于v/alpha。log1p只是有限的软偏置，零先验Expert仍有有限logits，可以被上下文激活。

### 23.2 真实案例如何选取

使用既定测试/验证协议下的模型和prior，不能为了“解释”案例把其目标事件重新加入统计表。

按样本量、confidence分层，再抽取Unique主导、Redundancy主导、二阶Synergy、高阶Synergy、低置信度及未知目标的案例。每组包含推荐正确与错误样本，预先记录抽样规则。

建议三类成对对照：

1. 同一POI、不同历史：prior固定，比较MLP logits、pi、Top-K、gate的变化。
2. 同一历史、不同候选：每个候选读自己的SURD行，比较真实目标与高排名负例。
3. 同一案例、不同模型设置：完整模型与无prior bias、无辅助loss或无R辅助的对照；区分重训练消融与推理期干预。

Category/Location由POI确定时，随机换类别而保留POI不应称为真实世界因果反事实。通道masking须保持事件对齐，作为模型依赖性或鲁棒性检查，并结合第21.1节解释。

### 23.3 单案例日志schema

```python
case_record = {
    "sample_id": "...",
    "split_id": "...",
    "prefix_end_time": "...",
    "candidate_poi_id": "...",
    "is_true_next_poi": False,
    "history_poi_ids": [...],
    "history_category_ids": [...],
    "history_time_summary": {...},
    "history_location_summary": {...},
    "prior_version": "...",
    "source_order": ["P", "C", "T", "L"],
    "atom_order": [...],                   # 26
    "expert_order": [...],                 # 15
    "target_count": 0,
    "total_information_bits": 0.0,
    "raw_surd": [...],
    "normalized_surd": [...],
    "routing_prior": [...],
    "confidence": 0.0,
    "us_mass": [...],
    "alpha": 0.0,
    "redundancy_mass": 0.0,
    "route_mlp_logits": [...],
    "explicit_prior_bias": [...],
    "final_route_logits": [...],
    "dense_route_probability": [...],
    "selected_experts": [...],
    "selected_weights": [...],
    "gate": 0.0,
    "base_score": 0.0,
    "expert_projected_scores": {...},
    "expert_score_contributions": {...},
    "moe_bias_contribution": 0.0,
    "final_score": 0.0,
    "candidate_rank": 0,
}
```

这些日志索引应与离线文件一致。用户/轨迹标识只用于检索案例，不因记录而成为模型输入。

### 23.4 五个可复算的路由演示

以下是**合成演示，不是训练结果**。SURD来自合法离散分布；MLP logits和gate为展示机制而指定。统一使用温度1、Top-K=4、lambda_prior=0.5、kappa_n=20、kappa_I=0.05 bit；示例的lambda不替代第19节起始配置。

每个合成数据集两个目标的信息结构相同，因此local与global prior一致，收缩不改变比例。大样本每目标200事件，小样本每目标2事件。

| 合法分布 | 构造 | 非零SURD |
| --- | --- | --- |
| 纯P Unique | 公平二值Y，Q_P=Y，其余来源常数 | U_P=1 bit |
| CT Synergy | 独立公平Q_C/Q_T，Y为二者XOR，Q_P/Q_L常数 | S_CT=1 bit |
| 全来源冗余 | 四来源都复制公平二值Y | R_PCTL=1 bit |

这是量化来源层面的解析分布，不要求真实原始Category与POI满足独立XOR。A/B/E使用纯P Unique，C使用CT Synergy，D使用全冗余。

$$
\omega_{\mathrm{large}}
=\frac{200}{220}\frac1{1.05}
=0.865800866,
\qquad
\Delta_{\mathrm{large}}
=0.5\times0.865800866\times\log16
=1.200254858.
$$

$$
\omega_{\mathrm{small}}
=\frac2{22}\frac1{1.05}
=0.086580087,
\qquad
\Delta_{\mathrm{small}}=0.120025486.
$$

“其他每路”指每个未单列Expert的概率，括号内为它们的总和，完整15路概率因此可以恢复：

| 案例 | SURD | 非零MLP logit | 非零prior bias | dense pi | gate |
| --- | --- | --- | --- | --- | ---: |
| A：可靠先验，无上下文偏好 | U_P=1 | 全零 | U_P：1.200255 | U_P：0.191731；其他14路各0.0577335（和0.808269） | 0.90 |
| B：同POI的不同历史 | 与A相同 | S_PT：2.0 | 与A相同 | S_PT：0.311643；U_P：0.140066；其他13路各0.0421763（和0.548291） | 0.80 |
| C：类别时间协同 | S_CT=1 | 全零 | S_CT：1.200255 | S_CT：0.191731；其他14路各0.0577335（和0.808269） | 0.85 |
| D：纯冗余但路由概率高 | R_PCTL=1 | U_P：3.0 | 全零 | U_P：0.589269；其他14路各0.0293380（和0.410731） | 0.05 |
| E：相同比例但样本很少 | U_P=1 | 全零 | U_P：0.120025 | U_P：0.074535；其他14路各0.0661047（和0.925465） | 0.60 |

为可复算，Router精确并列按固定Expert索引选取；实际实现应记录Top-K并列策略。并列中选中了哪个零先验Expert不代表稳定语义发现。

| 案例 | Top-K归一化权重 |
| --- | --- |
| A | U_P：0.525389；U_C/U_T/U_L：各0.158204 |
| B | S_PT：0.581357；U_P：0.261287；U_C/U_T：各0.078678 |
| C | S_CT：0.525389；U_P/U_C/U_T：各0.158204 |
| D | U_P：0.870049；U_C/U_T/U_L：各0.043317 |
| E | U_P：0.273172；U_C/U_T/U_L：各0.242276 |

A说明全部Unique先验不会强制pi等于1。B说明同POI prior固定，动态历史仍能让S_PT超过U_P；这不证明当前轨迹有严格统计Synergy，组合Expert也能使用P主效应，且在线表示与离线摘要不同。C展示交互先验的偏好。D说明高dense pi不等于大残差贡献：Top-K后的U_P权重约0.870，乘gate后约0.0435，最终影响还取决于输出幅度。E说明“100% Unique”仍可能只有很弱的可靠引导。

### 23.5 高路由概率不等于正的推荐贡献

对线性MoE输出头，定义选中Expert的分数贡献：

$$
\delta_{i,y,e}
=
g_{i,y}\widetilde\pi_{i,y,e}
w_{\mathrm{MoE}}^\top o_e(y).
$$

最终分数可精确分解为：

$$
s_\theta(y\mid\mathcal X_i)
=s_{\mathrm{base}}(i,y)
+g_{i,y}b_{\mathrm{MoE}}
+\sum_{e\in\mathcal K_{i,y}}\delta_{i,y,e}.
$$

SURD与Router权重非负，但Expert投影分数可以负，所以实际贡献也可以负。不能把最常路由的Expert直接称为提高了候选概率。

例如B中，S_PT/U_P/U_C/U_T的投影分数设为−0.4、1.2、0.1、0，Base=1.1，MoE bias=0.2：

$$
\begin{aligned}
\delta_{S_{PT}}&\approx-0.186034,\\
\delta_{U_P}&\approx0.250836,\\
\delta_{U_C}&\approx0.006294,\\
\delta_{U_T}&=0,\\
g\,b_{\mathrm{MoE}}&=0.16,\\
s_\theta&\approx1.331096.
\end{aligned}
$$

这里S_PT路由权重最高，却降低候选分数，并不与非负SURD矛盾。

这是固定前向计算的加性分数贡献。移除来源后重新计算Router、gate和全部Experts，是另一种整体干预效果，不能与固定前向delta混称。

### 23.6 展示方式与结论模板

每个案例可用三联图：左侧26项raw bit及样本量；中间对比normalized、routing prior和15路dense pi；右侧Top-K、gate及实际贡献。不同单位不应共用无标注纵轴。

批量分析建议报告：

- 同POI不同历史的路由差异，与不同POI之间的差异。
- 按confidence分层的gate–alpha关系。
- U/S条件先验与dense pi的距离；alpha或omega为零时不计算除以alpha的指标。
- 先验支持集合的Top-K命中质量、真实派发频率及错误案例。
- raw与routing prior的差异，区分本地证据和global回退。
- prior对齐变化与推荐指标变化：更接近prior不自动意味着更好推荐。

定性表述模板：

> 在规定参考集和量化配置下，该候选POI的主要SURD分量为……，总信息……bit，支持事件数……。当前历史下Router选择……，dense概率、Top-K权重和gate分别为……。与同POI另一条历史相比，prior保持不变，动态路由发生……变化。分数贡献显示……。这些观察与……建模解释一致，但不单独构成来源因果效应或Expert纯信息语义的证明。

没有真实训练结果时，只保留设计与合成演示，不能填写虚构的推荐提升或真实POI行为结论。

## 附录 A. 可运行的路径 A 与逐 POI SURD 参考实现

下面实现仅依赖 Python 标准库和 NumPy。输入是同一个稀疏联合频数表，键顺序为“目标 POI、P 状态、C 状态、T 状态、L 状态”，即 `(y, q_P, q_C, q_T, q_L)`。这些状态由四条对齐历史序列各自摘要和量化而来，函数不接收 user_id。

它返回 15 列 specific information、26 列原始分解和 26 列完整归一化结果。随后可调用 add_26d_shrinkage_and_confidence，得到仍为 26 维的路由先验及可靠性权重；该函数不改写原始分解，模型 Router 另行实现。

可将代码复制为 `surd_reference.py` 后运行；末尾包含七组解析分布检查，包括 Category/Location 是 POI 确定性视图的例子。概率质量输入只用于合成验证；真实训练样本的可靠性系数必须使用真实事件计数，不能把和为 1 的概率质量当作样本数。

```python
"""Four-source SURD from ONE sparse empirical joint distribution.

Sources, in order: P, C, T, L (indices 0, 1, 2, 3).
P=historical POI, C=historical Category, T=Time, L=Location.
U in atom labels means Unique; user_id is never a source.
Input keys are (poi_id, p_state, c_state, t_state, l_state).
Weights may be counts or nonnegative probability masses. No smoothing.
Output atoms: 4 unique, 11 redundant, 11 synergistic, in this order.

Sorting: ascending numerical value, exact ties by subset lexicographic order.
Near ties are reported; they are NOT averaged, reordered, or projected.
The numerical tolerance only permits roundoff and validates identities.
This implements the original SURD allocation equations, not a generic PID.
"""

from collections import defaultdict
from itertools import combinations, product
import math
import numpy as np


SOURCES = ("P", "C", "T", "L")
SUBSETS = tuple(a for k in range(1, 5) for a in combinations(range(4), k))
MULTIS = SUBSETS[4:]
ATOMS = (
    tuple(("U", a) for a in SUBSETS[:4])
    + tuple(("R", a) for a in MULTIS)
    + tuple(("S", a) for a in MULTIS)
)
SUBSET_INDEX = {a: k for k, a in enumerate(SUBSETS)}
ATOM_INDEX = {a: k for k, a in enumerate(ATOMS)}
ATOM_LABELS = tuple(f"{kind}_{''.join(SOURCES[i] for i in a)}" for kind, a in ATOMS)


def surd_event(specific, atol=1e-10, rtol=1e-10):
    """Allocate 15 specific-information values for ONE observed target state.

    ``specific`` is a length-15 vector in SUBSETS order, or a subset-keyed
    mapping. Return a length-26 atom vector in ATOMS order, measured in bits.
    Nonnegative and subset-monotone inputs are required, up to tolerance.
    """
    if hasattr(specific, "keys"):
        specific = [specific[a] for a in SUBSETS]
    values = np.asarray(specific, dtype=float).copy()
    if values.shape != (15,) or not np.isfinite(values).all():
        raise ValueError("specific must contain 15 finite numbers")
    tol = atol + rtol * max(float(np.abs(values).max()), 1.0)
    if values.min() < -tol:
        raise ValueError("specific information is materially negative")
    values = np.maximum(values, 0.0)  # Roundoff only, after the check above.
    info = dict(zip(SUBSETS, values))
    for a in SUBSETS:
        for b in SUBSETS:
            if set(a) < set(b) and info[a] > info[b] + tol:
                raise ValueError("specific information violates subset monotonicity")

    atoms = np.zeros(26, dtype=float)
    order = sorted(range(4), key=lambda i: (info[(i,)], i))
    previous = 0.0
    for k, i in enumerate(order):
        increment = info[(i,)] - previous
        key = ("U", (i,)) if k == 3 else ("R", tuple(sorted(order[k:])))
        atoms[ATOM_INDEX[key]] += increment
        previous = info[(i,)]

    for degree in range(2, 5):
        baseline = max(info[a] for a in SUBSETS if len(a) == degree - 1)
        group = sorted((a for a in SUBSETS if len(a) == degree),
                       key=lambda a: (info[a], a))
        previous = 0.0
        for a in group:
            atoms[ATOM_INDEX[("S", a)]] = max(0.0, info[a] - max(baseline, previous))
            previous = info[a]

    if atoms.min() < -tol:
        raise ArithmeticError("negative SURD atom")
    if abs(math.fsum(atoms) - info[(0, 1, 2, 3)]) > tol:
        raise ArithmeticError("total-information conservation failed")
    for i in range(4):
        reconstructed = atoms[ATOM_INDEX[("U", (i,))]] + math.fsum(
            atoms[ATOM_INDEX[("R", a)]] for a in MULTIS if i in a)
        if abs(reconstructed - info[(i,)]) > tol:
            raise ArithmeticError("single-source reconstruction failed")
    return atoms


def build_surd_prior(joint_counts, poi_ids, atol=1e-10, rtol=1e-10):
    """Marginalize a sparse 5-way table and return raw and normalized arrays.

    ``poi_ids`` fixes the row order and may include unobserved POIs.
    Unknown rows are zero with observed=False; zero-information observed
    rows also remain zero. Their routing fallback must be chosen separately.
    All normalized atoms use the SAME total over all 26 raw atoms.
    """
    poi_ids = tuple(poi_ids)
    if len(set(poi_ids)) != len(poi_ids):
        raise ValueError("poi_ids must be unique")
    row_index = {y: j for j, y in enumerate(poi_ids)}
    rows = []
    for state, weight in joint_counts.items():
        if len(state) != 5:
            raise ValueError("joint state must be (y, p, c, t, l)")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("joint weights must be finite and nonnegative")
        if weight > 0:
            if state[0] not in row_index:
                raise ValueError("poi_ids omits an observed target")
            rows.append((state, weight))
    total_weight = math.fsum(w for _, w in rows)
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("joint table must have positive finite total weight")

    # These 15 marginals all come from the SAME original joint table.
    target_counts = defaultdict(float)
    joint_marginals = {a: defaultdict(float) for a in SUBSETS}
    source_marginals = {a: defaultdict(float) for a in SUBSETS}
    for state, weight in rows:
        y, source = state[0], state[1:]
        target_counts[y] += weight
        for a in SUBSETS:
            q = tuple(source[i] for i in a)
            joint_marginals[a][(y, q)] += weight
            source_marginals[a][q] += weight

    specific = np.zeros((len(poi_ids), 15))
    for column, a in enumerate(SUBSETS):
        terms = defaultdict(list)
        for (y, q), count in joint_marginals[a].items():
            ny, nq = target_counts[y], source_marginals[a][q]
            # log2(count * total_weight / (ny * nq)), without large products.
            log_ratio = math.log2(count) + math.log2(total_weight) - math.log2(ny) - math.log2(nq)
            terms[y].append((count / ny) * log_ratio)
        for y, summands in terms.items():
            specific[row_index[y], column] = math.fsum(summands)

    observed = np.array([target_counts[y] > 0 for y in poi_ids], dtype=bool)
    probability = np.array([target_counts[y] / total_weight for y in poi_ids])
    raw = np.zeros((len(poi_ids), 26))
    near_tie = np.zeros(len(poi_ids), dtype=bool)
    for row in np.flatnonzero(observed):
        raw[row] = surd_event(specific[row], atol=atol, rtol=rtol)
        tol = atol + rtol * max(float(np.abs(specific[row]).max()), 1.0)
        for degree in range(1, 5):
            group = sorted(specific[row, k] for k, a in enumerate(SUBSETS) if len(a) == degree)
            gaps = np.diff(group)
            near_tie[row] |= bool(np.any((gaps > 0) & (gaps <= tol)))

    # A defined numerical-zero threshold avoids normalizing roundoff noise.
    total_information = raw.sum(axis=1)
    zero_tol = atol + rtol * np.maximum(np.max(np.abs(specific), axis=1), 1.0)
    has_information = observed & (total_information > zero_tol)
    normalized = np.zeros_like(raw)
    normalized[has_information] = raw[has_information] / total_information[has_information, None]
    global_atoms = probability @ raw
    global_mi = probability @ specific
    if not np.isclose(global_atoms.sum(), global_mi[-1], atol=atol, rtol=rtol):
        raise ArithmeticError("global-information conservation failed")
    return dict(poi_ids=poi_ids, subsets=SUBSETS, atom_labels=ATOM_LABELS,
                specific=specific, raw=raw, normalized=normalized,
                observed=observed, has_information=has_information,
                near_tie=near_tie, target_probability=probability,
                target_count=np.array([target_counts[y] for y in poi_ids]),
                total_weight=total_weight,
                total_information=total_information,
                global_atoms=global_atoms, global_mi=global_mi)


def add_26d_shrinkage_and_confidence(stats, kappa_n=20.0, kappa_i=0.05,
                                    atol=1e-10, rtol=1e-10):
    """Keep all 26 coordinates; do not renormalize the U/S slice.

    target_count must represent actual event counts for count reliability.
    For synthetic probability-mass input this function's confidence values
    are not statistical sample-size confidence.
    """
    if kappa_n <= 0 or kappa_i <= 0:
        raise ValueError("reliability scales must be positive")
    local = np.asarray(stats["normalized"], dtype=float)
    total = np.asarray(stats["total_information"], dtype=float)
    count = np.asarray(stats["target_count"], dtype=float)
    valid = np.asarray(stats["has_information"], dtype=bool)
    observed = np.asarray(stats["observed"], dtype=bool)
    global_raw = np.asarray(stats["global_atoms"], dtype=float)
    global_total = float(global_raw.sum())
    global_tol = atol + rtol * max(float(np.abs(global_raw).max()), 1.0)
    global_valid = global_total > global_tol
    count_factor = count / (count + kappa_n)

    global_prior = np.zeros(26, dtype=float)
    if global_valid:
        global_prior = global_raw / global_total
        routing = np.broadcast_to(global_prior, local.shape).copy()
        c = count_factor[valid, None]
        routing[valid] = c * local[valid] + (1.0 - c) * global_prior
    else:
        # A rare local state can be informative although global MI is tiny.
        # Retain its normalized row instead of shrinking toward a zero row.
        routing = np.zeros_like(local)
        routing[valid] = local[valid]

    confidence = (
        observed * valid * count_factor * total / (total + kappa_i)
    )
    nonzero = routing.sum(axis=1) > 0.0
    if not np.allclose(routing[nonzero].sum(axis=1), 1.0, atol=atol, rtol=rtol):
        raise ArithmeticError("nonzero routing rows must sum to one")
    if routing.min() < -atol:
        raise ArithmeticError("negative routing coordinate")
    result = dict(stats)
    result.update(routing_prior=routing, confidence=confidence,
                  global_prior=global_prior, count_factor=count_factor)
    return result


def _check_examples():
    def expect(table, expected, label, observed_pois=(0, 1)):
        observed_pois = tuple(observed_pois)
        result = build_surd_prior(table, observed_pois + (99,))
        truth = np.array([expected.get(label, 0.0) for label in ATOM_LABELS])
        np.testing.assert_allclose(
            result["raw"][:-1], np.tile(truth, (len(observed_pois), 1)), atol=1e-12)
        assert not result["observed"][-1] and not result["raw"][-1].any()
        active = {k: round(float(v), 10) for k, v in zip(ATOM_LABELS, result["raw"][0]) if v > 1e-12}
        print(f"{label}: total={result['raw'][0].sum():.10f}; atoms={active}")
        return result

    expect({(y, y, y, y, y): 1.0 for y in [0, 1]},
           {"R_PCTL": 1.0}, "four copies (pure redundancy)")
    expect({(p, p, c, t, l): 1.0 for p, c, t, l in product([0, 1], repeat=4)},
           {"U_P": 1.0}, "copy target from P only")
    expect({(p ^ c, p, c, t, l): 1.0 for p, c, t, l in product([0, 1], repeat=4)},
           {"S_PC": 1.0}, "XOR of P and C")
    expect({(p ^ c ^ t ^ l, p, c, t, l): 1.0 for p, c, t, l in product([0, 1], repeat=4)},
           {"S_PCTL": 1.0}, "four-source XOR")
    noise = expect({(y, p, c, t, l): 1.0 for y, p, c, t, l in product([0, 1], repeat=5)},
                   {}, "independent target noise")
    assert not noise["has_information"].any() and not noise["normalized"].any()

    # Fair Y; each source reveals Y independently, otherwise outputs state 2.
    reveal_probability = (0.4, 0.3, 0.1, 0.2)
    erasure = {}
    for y in [0, 1]:
        for mask in product([0, 1], repeat=4):
            state = (y,) + tuple(y if visible else 2 for visible in mask)
            erasure[state] = 0.5 * math.prod(p if visible else 1 - p
                                            for visible, p in zip(mask, reveal_probability))
    expected = {"U_P": .1, "R_PC": .1, "R_PCL": .1, "R_PCTL": .1,
                "S_PC": .06, "S_PT": .02, "S_PL": .06, "S_CL": .04,
                "S_PCT": .042, "S_PCL": .042, "S_PCTL": .0336}
    result = expect(erasure, expected, "independent erasures")
    for k, a in enumerate(SUBSETS):
        analytical = 1 - math.prod(1 - reveal_probability[i] for i in a)
        np.testing.assert_allclose(result["specific"][:2, k], analytical, atol=1e-12)
    np.testing.assert_allclose(result["total_information"][:2], .6976, atol=1e-12)
    np.testing.assert_allclose(result["normalized"][:2].sum(axis=1), 1.0, atol=1e-12)
    # Category and Location are deterministic metadata views of POI.
    # Let Y=P be a uniform two-bit state; C is its low bit, L its high bit.
    # T is independent. A full P observation already supplies both bits.
    # SURD compares every pair to the strongest singleton, so even S_CL=0.
    metadata = {(p, p, p % 2, t, p // 2): 1.0
                for p in range(4) for t in [0, 1]}
    deterministic = expect(metadata, {"U_P": 1.0, "R_PCL": 1.0},
                           "deterministic Category and Location", range(4))
    info = deterministic["specific"][:4]
    for a in ((0,), (0, 1), (0, 3), (1, 3), (0, 1, 2, 3)):
        np.testing.assert_allclose(info[:, SUBSET_INDEX[a]], 2.0, atol=1e-12)
    for a in ((1,), (3,)):
        np.testing.assert_allclose(info[:, SUBSET_INDEX[a]], 1.0, atol=1e-12)
    np.testing.assert_allclose(info[:, SUBSET_INDEX[(2,)]], 0.0, atol=1e-12)
    for atom in (("U", (1,)), ("U", (2,)), ("U", (3,))):
        np.testing.assert_allclose(deterministic["raw"][:4, ATOM_INDEX[atom]],
                                   0.0, atol=1e-12)
    np.testing.assert_allclose(deterministic["raw"][:4, 15:], 0.0, atol=1e-12)
    print("All seven analytical checks passed; 15 marginals and 26 atoms retained.")


if __name__ == "__main__":
    _check_examples()
```

## 附录 B. 更新后的研究贡献表述

本文研究的是：基于 POI、Category、Time、Location 四条对齐历史序列，利用逐目标 POI 的信息分解，指导序列推荐中不同来源交互的选择。user_id 不作为预测输入，个性化由已观测轨迹提供。

具体贡献可以围绕三个问题组织：

1. **四序列来源下逐 POI 的信息结构。**从同一个训练联合分布得到 P/C/T/L 的目标事件级 U/R/S，保持目标 POI 身份与来源边界。
2. **统计分解与执行结构分离。**完整保留 26 维信息描述，实际只执行 15 个 U/S Experts；冗余提供 Router 与残差门的辅助上下文。
3. **可复现的离线到在线接口。**固定量化、稀疏计数、目标事件分解、折外 prior、可靠性控制与候选独立查表形成完整流程。

本基础版使用固定特征、简单有序摘要与 K-means 定义离散来源，贡献主张聚焦来源约束、分解与路由之间的可检验联系。固定摘要是可复现的起点，不能把摘要本身当作已证实的新表示学习贡献。

这些贡献需要通过等参数量基线、摘要组成与量化粒度对照、哈希维度/种子敏感性、Category 通道消融、逐 POI 与全局 prior 对照、R 辅助消融、分解稳定性及推荐指标共同验证。若 Category/Location 是 POI 的静态元数据，应将非零摘要互补与原始序列层面的确定性边界分开解释。


## 附录 C. 可运行的 PyTorch 离线 SURD、保存与冻结查表

将下面完整代码复制为 `surd_offline_torch.py`。第 16 节从该模块导入函数。代码在 PyTorch 2.6.0 CPU 上验证，直接运行文件会执行解析自检和保存/加载检查；这些检查不包含真实数据的端到端推荐训练。

统计输入必须已遵守第 4/10 节的数据边界。七组解析例用于验证算法，不表示真实 POI 数据遵循 XOR、独立擦除或其他合成分布。

```python
"""Offline four-source SURD, followed by frozen PyTorch lookup.

Input q_states has one (P,C,T,L) quantized history per real prediction event.
target_rows indexes poi_ids, not the numerical magnitude of an external POI ID.
Fixed per-source feature extraction and quantization happen BEFORE this module.
All 15 marginals derive from one sparse five-way joint count table.

This is a CPU/float64 statistical reference, not a differentiable estimator.
Exact ties use fixed source/subset order; near ties are reported, never averaged.
Run this file to execute the seven analytical examples and persistence checks.
"""

from itertools import combinations, product
from pathlib import Path
import json
import math
import tempfile

import torch
from torch import nn


SOURCE_ORDER = ("P", "C", "T", "L")
SUBSETS = tuple(a for k in range(1, 5) for a in combinations(range(4), k))
MULTIS = SUBSETS[4:]
ATOMS = (
    tuple(("U", a) for a in SUBSETS[:4])
    + tuple(("R", a) for a in MULTIS)
    + tuple(("S", a) for a in MULTIS)
)
SUBSET_ORDER = tuple("".join(SOURCE_ORDER[j] for j in a) for a in SUBSETS)
ATOM_ORDER = tuple(f"{kind}_{''.join(SOURCE_ORDER[j] for j in a)}"
                   for kind, a in ATOMS)
EXPERT_ORDER = ATOM_ORDER[:4] + ATOM_ORDER[15:]
ATOM_INDEX = {key: j for j, key in enumerate(ATOMS)}
DTYPE = torch.float64
INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _long_cpu(value, name):
    tensor = torch.as_tensor(value)
    if tensor.dtype not in INTEGER_DTYPES:
        raise TypeError(f"{name} must contain integer IDs, not floats/bools")
    return tensor.detach().to(device="cpu", dtype=torch.long)


def _sum_by_index(index, value, size):
    return torch.zeros(size, dtype=DTYPE).scatter_add_(0, index, value)


@torch.no_grad()
def surd_event_torch(specific, atol=1e-10, rtol=1e-10):
    """Allocate one target event's 15 specific-information values to 26 atoms."""
    values = torch.as_tensor(specific, dtype=DTYPE, device="cpu").clone()
    if values.shape != (15,) or not torch.isfinite(values).all():
        raise ValueError("specific must be 15 finite values")
    tol = atol + rtol * max(float(values.abs().max()), 1.0)
    if float(values.min()) < -tol:
        raise ValueError("materially negative specific information")
    values.clamp_(min=0.0)  # Only roundoff, after validation.
    info = dict(zip(SUBSETS, values.tolist()))
    for a in SUBSETS:
        for b in SUBSETS:
            if set(a) < set(b) and info[a] > info[b] + tol:
                raise ValueError("subset monotonicity failed")

    atom = torch.zeros(26, dtype=DTYPE)
    order = sorted(range(4), key=lambda m: (info[(m,)], m))
    previous = 0.0
    for rank, m in enumerate(order):
        key = ("U", (m,)) if rank == 3 else ("R", tuple(sorted(order[rank:])))
        atom[ATOM_INDEX[key]] += info[(m,)] - previous
        previous = info[(m,)]
    for degree in range(2, 5):
        baseline = max(info[a] for a in SUBSETS if len(a) == degree - 1)
        group = sorted((a for a in SUBSETS if len(a) == degree),
                       key=lambda a: (info[a], a))
        previous = 0.0
        for a in group:
            atom[ATOM_INDEX[("S", a)]] = max(0.0, info[a] - max(baseline, previous))
            previous = info[a]
        total_degree = sum(float(atom[ATOM_INDEX[("S", a)]]) for a in group)
        expected_degree = max(info[a] for a in group) - baseline
        if abs(total_degree - expected_degree) > tol:
            raise ArithmeticError("synergy-by-degree conservation failed")

    if float(atom.min()) < -tol or abs(float(atom.sum()) - info[SUBSETS[-1]]) > tol:
        raise ArithmeticError("atom nonnegativity/total conservation failed")
    for m in range(4):
        rebuilt = float(atom[ATOM_INDEX[("U", (m,))]]) + math.fsum(
            float(atom[ATOM_INDEX[("R", a)]]) for a in MULTIS if m in a)
        if abs(rebuilt - info[(m,)]) > tol:
            raise ArithmeticError("single-source reconstruction failed")
    return atom


@torch.no_grad()
def build_offline_surd(q_states, target_rows, poi_ids, *, sample_weight=None,
                       weight_semantics="event_count", kappa_n=20.0, kappa_i=0.05,
                       atol=1e-10, rtol=1e-10, manifest=None):
    """Build and validate an offline table.

    q_states: [n,4] integer discrete states in P,C,T,L order, each >=0.
    target_rows: [n] integers in [0,V); poi_ids fixes the shared catalog row order.
    Default: one row = one distinct real target event, weight 1.
    Explicit event_count weights may aggregate identical events and must be
    nonnegative integers; duplicated augmented windows must already be deduped.
    probability_mass weights are allowed only for analytic examples: sample-size
    shrinkage is disabled and confidence is ZERO, because mass is not a count.
    """
    q = _long_cpu(q_states, "q_states")
    y = _long_cpu(target_rows, "target_rows")
    poi_ids = list(poi_ids)
    if (not poi_ids or any(type(v) not in (str, int) for v in poi_ids)
            or len(set(poi_ids)) != len(poi_ids)):
        raise ValueError("poi_ids must be a nonempty list of unique int/string IDs")
    V = len(poi_ids)
    if q.ndim != 2 or q.shape[1] != 4 or y.shape != (q.shape[0],) or not len(y):
        raise ValueError("expected q_states[n,4] and target_rows[n], n>0")
    if (q < 0).any() or (y < 0).any() or (y >= V).any():
        raise ValueError("source states must be >=0 and targets must index poi_ids")
    if kappa_n <= 0 or kappa_i <= 0:
        raise ValueError("reliability scales must be positive")
    if weight_semantics not in ("event_count", "probability_mass"):
        raise ValueError("unknown weight semantics")
    w = (torch.ones(len(y), dtype=DTYPE) if sample_weight is None else
         torch.as_tensor(sample_weight, dtype=DTYPE, device="cpu").detach())
    if w.shape != y.shape or not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("sample_weight must be finite, nonnegative and length n")
    if weight_semantics == "event_count" and not torch.equal(w, w.round()):
        raise ValueError("event_count weights must be integer counts")
    keep = w > 0
    q, y, w = q[keep], y[keep], w[keep]
    if not len(y):
        raise ValueError("positive total weight required")
    safe_manifest = json.loads(json.dumps(manifest or {}, allow_nan=False))

    # First construct exactly ONE sparse joint table (Y,Q_P,Q_C,Q_T,Q_L).
    joint, inverse = torch.unique(torch.cat([y[:, None], q], dim=1),
                                  dim=0, sorted=True, return_inverse=True)
    joint_count = _sum_by_index(inverse, w, len(joint))
    total_weight = joint_count.sum()
    if not torch.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("positive finite total weight required")
    target_count = _sum_by_index(joint[:, 0], joint_count, V)
    target_probability = target_count / total_weight
    observed = target_count > 0
    specific = torch.zeros(V, 15, dtype=DTYPE)

    # Every subset marginal is projected/aggregated from that same joint table.
    for column, subset in enumerate(SUBSETS):
        columns = [0] + [1 + j for j in subset]
        marginal, inv = torch.unique(joint[:, columns], dim=0,
                                     sorted=True, return_inverse=True)
        count_yq = _sum_by_index(inv, joint_count, len(marginal))
        states, source_inv = torch.unique(marginal[:, 1:], dim=0,
                                          sorted=True, return_inverse=True)
        count_q = _sum_by_index(source_inv, count_yq, len(states))
        row = marginal[:, 0]
        count_y = target_count[row]
        log_ratio = (count_yq.log2() + total_weight.log2()
                     - count_y.log2() - count_q[source_inv].log2())
        specific[:, column].scatter_add_(0, row, (count_yq / count_y) * log_ratio)

    raw = torch.zeros(V, 26, dtype=DTYPE)
    near_tie = torch.zeros(V, dtype=torch.bool)
    zero_tol = atol + rtol * specific.abs().amax(dim=1).clamp_min(1.0)
    for row in torch.nonzero(observed, as_tuple=False).flatten().tolist():
        raw[row] = surd_event_torch(specific[row], atol, rtol)
        if specific[row].max() > -target_probability[row].log2() + zero_tol[row]:
            raise ArithmeticError("event-information upper bound failed")
        for degree in range(1, 5):
            columns = [j for j, a in enumerate(SUBSETS) if len(a) == degree]
            ordered = specific[row, columns].sort().values
            gap = ordered[1:] - ordered[:-1]
            near_tie[row] |= ((gap > 0) & (gap <= zero_tol[row])).any()
    total_information = raw.sum(dim=1)
    has_information = observed & (total_information > zero_tol)
    normalized = torch.zeros_like(raw)
    normalized[has_information] = (
        raw[has_information] / total_information[has_information, None])
    global_raw = target_probability @ raw
    global_mi = target_probability @ specific
    global_total = global_raw.sum()
    global_tol = atol + rtol * max(float(global_raw.abs().max()), 1.0)
    global_prior = (global_raw / global_total if global_total > global_tol
                    else torch.zeros(26, dtype=DTYPE))

    if weight_semantics == "event_count":
        count_factor = target_count / (target_count + kappa_n)
    else:
        count_factor = observed.to(DTYPE)  # No arbitrary probability-mass shrinkage.
    routing_prior = global_prior[None].expand(V, -1).clone()
    if global_total > global_tol:
        c = count_factor[has_information, None]
        routing_prior[has_information] = (
            c * normalized[has_information] + (1 - c) * global_prior)
    else:
        routing_prior[has_information] = normalized[has_information]
    confidence = (has_information.to(DTYPE) * count_factor
                  * total_information / (total_information + kappa_i))
    if weight_semantics == "probability_mass":
        confidence.zero_()

    # Estimate H(Y) and H(Y|Q_P,Q_C,Q_T,Q_L) directly from the same counts.
    py = target_probability[observed]
    H_Y = -(py * py.log2()).sum()
    full_states, full_inv = torch.unique(joint[:, 1:], dim=0,
                                         sorted=True, return_inverse=True)
    count_full_q = _sum_by_index(full_inv, joint_count, len(full_states))
    H_cond = -((joint_count / total_weight)
               * (joint_count.log2() - count_full_q[full_inv].log2())).sum()
    H_cond_via_mi = H_Y - target_probability @ total_information
    torch.testing.assert_close(H_cond, H_cond_via_mi, atol=atol, rtol=rtol)
    torch.testing.assert_close(global_raw.sum(), global_mi[-1], atol=atol, rtol=rtol)
    nonzero = routing_prior.sum(dim=1) > 0
    torch.testing.assert_close(routing_prior[nonzero].sum(dim=1),
                               torch.ones(int(nonzero.sum()), dtype=DTYPE),
                               atol=atol, rtol=rtol)
    if routing_prior.min() < -atol:
        raise ArithmeticError("negative routing prior")
    return {
        "schema_version": "PCTL_TORCH_SURD_v2",
        "source_order": list(SOURCE_ORDER), "subset_order": list(SUBSET_ORDER),
        "atom_order": list(ATOM_ORDER), "expert_order": list(EXPERT_ORDER),
        "poi_ids": poi_ids, "specific_information": specific, "raw_surd": raw,
        "normalized_surd": normalized, "routing_prior": routing_prior,
        "total_information": total_information, "target_count": target_count,
        "target_probability": target_probability, "observed": observed,
        "has_information": has_information, "near_tie": near_tie,
        "confidence": confidence, "count_factor": count_factor,
        "global_raw": global_raw, "global_prior": global_prior, "global_mi": global_mi,
        "H_Y": H_Y, "H_cond": H_cond, "H_cond_via_mi": H_cond_via_mi,
        "joint_states": joint, "joint_count": joint_count,
        "total_weight": total_weight,
        "config": {"kappa_n": kappa_n, "kappa_i": kappa_i,
                   "atol": atol, "rtol": rtol, "weight_semantics": weight_semantics},
        "manifest": safe_manifest,
    }


def save_surd_table(stats, path):
    """Save primitive metadata and detached CPU tensors; raw remains float64."""
    payload = {key: value.detach().cpu() if torch.is_tensor(value) else value
               for key, value in stats.items()}
    torch.save(payload, Path(path))


class FrozenSURDTable(nn.Module):
    """Runtime candidate-row lookup; -1 is unknown, other invalid IDs raise.

    All fold tables must use the same poi_ids catalog order. Quantizer state
    numbers need NOT match across folds: their counts must never be pooled.
    Candidate padding still requires a separate candidate-valid mask.
    """

    def __init__(self, stats, dtype=torch.float32):
        super().__init__()
        for name, expected in (("source_order", SOURCE_ORDER),
                               ("subset_order", SUBSET_ORDER),
                               ("atom_order", ATOM_ORDER),
                               ("expert_order", EXPERT_ORDER)):
            if tuple(stats[name]) != expected:
                raise ValueError(f"incompatible {name}")
        if stats["schema_version"] != "PCTL_TORCH_SURD_v2":
            raise ValueError("unexpected schema version")
        self.poi_ids = list(stats["poi_ids"])
        self.manifest = stats["manifest"]
        self.row_of = {poi: row for row, poi in enumerate(self.poi_ids)}
        V = len(self.poi_ids)
        if stats["routing_prior"].shape != (V, 26) or stats["confidence"].shape != (V,):
            raise ValueError("invalid prior/confidence shape")
        for name in ("routing_prior", "confidence", "global_prior", "raw_surd",
                     "normalized_surd", "total_information", "target_count"):
            self.register_buffer(name, stats[name].detach().to(dtype=dtype).clone())
        for name in ("observed", "has_information", "near_tie"):
            self.register_buffer(name, stats[name].detach().to(torch.bool).clone())

    @classmethod
    def from_file(cls, path, dtype=torch.float32):
        stats = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(stats, dtype=dtype)

    def forward(self, candidate_rows):
        if candidate_rows.dtype not in INTEGER_DTYPES:
            raise TypeError("candidate_rows must be integer catalog row indices")
        rows = candidate_rows.to(device=self.routing_prior.device, dtype=torch.long)
        if ((rows < -1) | (rows >= len(self.poi_ids))).any():
            raise IndexError("only -1 or valid catalog rows are allowed")
        unknown = rows == -1
        safe_rows = rows.clamp_min(0)  # -1 must never silently index the last row.
        prior = self.routing_prior[safe_rows]
        weight = self.confidence[safe_rows]
        prior = torch.where(unknown[..., None], self.global_prior, prior)
        weight = torch.where(unknown, torch.zeros_like(weight), weight)
        return prior, weight


def gather_fold_priors(tables, fold_ids, candidate_rows):
    """Batch rows may come from different folds; each gets its own table."""
    if not len(tables):
        raise ValueError("at least one fold table is required")
    if fold_ids.dtype not in INTEGER_DTYPES or candidate_rows.ndim != 2:
        raise TypeError("expected integer fold_ids[B] and candidate_rows[B,M]")
    if fold_ids.shape != (candidate_rows.shape[0],):
        raise ValueError("fold_ids must have one entry per batch sample")
    if any(table.poi_ids != tables[0].poi_ids for table in tables):
        raise ValueError("all fold tables must share the candidate catalog order")
    device = tables[0].routing_prior.device
    folds = fold_ids.to(device=device, dtype=torch.long)
    rows = candidate_rows.to(device)
    if ((folds < 0) | (folds >= len(tables))).any():
        raise IndexError("invalid fold index")
    prior = tables[0].routing_prior.new_empty(*rows.shape, 26)
    confidence = tables[0].confidence.new_empty(rows.shape)
    for fold in folds.unique().tolist():
        selected = folds == fold
        prior[selected], confidence[selected] = tables[fold](rows[selected])
    return prior, confidence


def _self_test():
    def expect(joint_dict, expected, name, known=(0, 1), probability=False):
        keys, weights = zip(*joint_dict.items())
        catalog = list(known) + [99]
        row_of = {poi: row for row, poi in enumerate(catalog)}
        stats = build_offline_surd(
            torch.tensor([key[1:] for key in keys], dtype=torch.long),
            torch.tensor([row_of[key[0]] for key in keys], dtype=torch.long),
            catalog, sample_weight=weights,
            weight_semantics="probability_mass" if probability else "event_count")
        expected_raw = torch.tensor([expected.get(a, 0) for a in ATOM_ORDER],
                                    dtype=DTYPE).expand(len(known), -1)
        torch.testing.assert_close(stats["raw_surd"][:-1], expected_raw, atol=1e-12, rtol=1e-12)
        assert not stats["observed"][-1] and not stats["raw_surd"][-1].any()
        active = {key: round(float(value), 10) for key, value in
                  zip(ATOM_ORDER, stats["raw_surd"][0]) if value > 1e-12}
        print(f"{name}: atoms={active}; H(Y)={float(stats['H_Y']):.10f}; "
              f"H(Y|Q)={float(stats['H_cond']):.10f}")
        return stats

    copies = expect({(y, y, y, y, y): 1 for y in (0, 1)}, {"R_PCTL": 1}, "copies")
    unique = expect({(p, p, c, t, l): 1 for p, c, t, l in product((0, 1), repeat=4)},
                    {"U_P": 1}, "P only")
    expect({(p ^ c, p, c, t, l): 1 for p, c, t, l in product((0, 1), repeat=4)},
           {"S_PC": 1}, "P-C XOR")
    expect({(p ^ c ^ t ^ l, p, c, t, l): 1
            for p, c, t, l in product((0, 1), repeat=4)}, {"S_PCTL": 1}, "four-way XOR")
    noise = expect({(y, p, c, t, l): 1 for y, p, c, t, l in product((0, 1), repeat=5)},
                   {}, "independent noise")
    assert not noise["has_information"].any() and not noise["confidence"].any()

    reveal = (.4, .3, .1, .2)
    erasure = {}
    for y in (0, 1):
        for visible in product((0, 1), repeat=4):
            key = (y,) + tuple(y if flag else 2 for flag in visible)
            erasure[key] = .5 * math.prod(p if flag else 1 - p
                                         for p, flag in zip(reveal, visible))
    erased = expect(erasure,
                    {"U_P": .1, "R_PC": .1, "R_PCL": .1, "R_PCTL": .1,
                     "S_PC": .06, "S_PT": .02, "S_PL": .06, "S_CL": .04,
                     "S_PCT": .042, "S_PCL": .042, "S_PCTL": .0336},
                    "independent erasures", probability=True)
    torch.testing.assert_close(erased["H_cond"], torch.tensor(.3024, dtype=DTYPE),
                               atol=1e-12, rtol=1e-12)
    assert not erased["confidence"].any()  # Probability mass is not a sample count.
    for column, subset in enumerate(SUBSETS):
        exact = 1 - math.prod(1 - reveal[m] for m in subset)
        torch.testing.assert_close(erased["specific_information"][:2, column],
                                   torch.full((2,), exact, dtype=DTYPE), atol=1e-12, rtol=1e-12)
    expect({(p, p, p % 2, t, p // 2): 1 for p in range(4) for t in (0, 1)},
           {"U_P": 1, "R_PCL": 1}, "deterministic C/L", known=range(4))

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "surd.pt"
        save_surd_table(unique, path)
        table = FrozenSURDTable.from_file(path)
        prior, confidence = table(torch.tensor([[0, 1, 2, -1]]))
        assert prior.shape == (1, 4, 26) and confidence.shape == (1, 4)
        assert not prior.requires_grad and not confidence.requires_grad
        assert not list(table.parameters())
        assert confidence[0, 2] == 0 and confidence[0, 3] == 0
        torch.testing.assert_close(prior[0, -1], table.global_prior)
        torch.testing.assert_close(prior[0, 2], table.global_prior)
        for bad in (-2, len(table.poi_ids)):
            try:
                table(torch.tensor([[bad]]))
            except IndexError:
                pass
            else:
                raise AssertionError("invalid candidate row was not rejected")
        tables = nn.ModuleList([FrozenSURDTable(unique), FrozenSURDTable(copies)])
        folds = torch.tensor([0, 1])
        rows = torch.tensor([[0, -1], [1, 2]])
        mixed_prior, mixed_weight = gather_fold_priors(tables, folds, rows)
        assert mixed_prior.shape == (2, 2, 26) and mixed_weight.shape == (2, 2)
        for batch_row, fold in enumerate(folds.tolist()):
            p, w = tables[fold](rows[batch_row])
            torch.testing.assert_close(mixed_prior[batch_row], p)
            torch.testing.assert_close(mixed_weight[batch_row], w)
    print("All seven analytic cases, entropy identities, save/load and lookup checks passed.")
    print(f"torch={torch.__version__}; 15 marginals; 26 stored atoms; 15 expert coordinates.")


if __name__ == "__main__":
    _self_test()
```

## 附录 D. 可运行的基础来源摘要与量化器

将下列代码保存为 `basic_source_quantizer.py`，与附录 C 的 `surd_offline_torch.py` 放在同一目录。第 4.8、16.1 与 17.2 节使用同一套接口。

本模块直接完成固定 P/C 哈希、逐源摘要、逐源标准化及 K-means，提供 fit / transform / save / load。**聚类已经在本模块内部实现，调用后无需另外运行聚类算法。**

| 方法 | 实际职责 | 返回或保存的内容 |
| --- | --- | --- |
| `summarize()` | 四个来源分别对每条历史做固定摘要 | 四个实数矩阵，默认维数 193/193/19/25 |
| `fit()` | 计算四套标准化参数，并逐源调用 `_kmeans()` | 保存四张中心表，返回自身 |
| `transform()` | 使用已保存的参数标准化摘要，再取最近中心编号 | CPU 上的整数 tensor `[B,4]`，列顺序 P/C/T/L |
| `save()` / `load()` | 保存或恢复同一套固定变换 | 配置、均值、尺度与中心 |

本类的 `transform()` 返回簇编号；若将来改用 scikit-learn 实现，这一步对应 `KMeans.predict()`。scikit-learn 自身的 `KMeans.transform()` 返回到各个中心的距离，不能直接当作本文的离散状态 [6]。

T/L 接收第 11.1 节已构造的数值特征，默认维度分别为 6、8。P/C 的 PAD=-1、缺失 UNK=-2 是本离线接口约定；在线 nn.Embedding 应另行映射为其合法非负编号。`valid_mask=True` 表示真实事件，四路使用同一个右侧 padding mask。

实际输入维度会写入 config，默认摘要维度为 P=193、C=193、T=19、L=25。未知但有效的非负 P/C ID 可直接使用固定哈希；K-means 始终分配到已经保存的中心。全空历史不构造来源状态，由调用方单独处理。由于本版是 CPU 参考实现，先在可管理的参考集上运行；大型数据可以分批产生摘要，但需保持同一变换版本与相同计数语义。

合成检查使用 64 条对齐历史、4 个候选 POI：量化状态为 64×4，原始分解为 4×26，冻结查表输出为 2×4×26，confidence 为 2×4。量化器重载后结果相同；只修改一个来源时其他来源的摘要与状态保持不变。代码验证环境为 PyTorch 2.6.0 CPU；未在用户的真实数据上训练推荐模型。

```python
"""Basic, graph-free P/C/T/L history summaries and offline quantization.

P/C: integer [B,N], PAD=-1 only in padding, UNK=-2 at missing real events.
T/L: finite numeric [B,N,d] on real events; upstream implements Section 11's
time/location schema, including finite fills and validity flags (default 6/8).
valid_mask: shared bool [B,N], right-padded and nonempty for every history.
No user ID, target label, trainable embedding, or graph enters this module.
"""
import hashlib
import math
from pathlib import Path

import torch

SOURCES = ("P", "C", "T", "L")
INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
SCHEMA = "BASIC_SOURCE_QUANTIZER_v1"


class BasicSourceQuantizer:
    """Fit once on permitted reference histories; freeze for all later use."""

    def __init__(self, n_clusters=4, hash_dim=64, max_length=50, seed=0,
                 tau_pos=10.0, max_iter=100, tol=1e-6):
        for name, value in (("n_clusters", n_clusters), ("hash_dim", hash_dim),
                            ("max_length", max_length), ("max_iter", max_iter)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63 - 4:
            raise ValueError("seed must be an integer in [0, 2**63-4)")
        if not math.isfinite(tau_pos) or tau_pos <= 0 or not math.isfinite(tol) or tol < 0:
            raise ValueError("tau_pos must be positive; tol must be nonnegative and finite")
        self._params = dict(n_clusters=n_clusters, hash_dim=hash_dim,
                            max_length=max_length, seed=seed, tau_pos=float(tau_pos),
                            max_iter=max_iter, tol=float(tol))
        self._salts = {s: f"{s}/basic_fixed_features_v1" for s in SOURCES[:2]}
        self._state = {}
        self._input_dims = {}

    @property
    def effective_clusters(self):
        return {s: int(v["centers"].shape[0]) for s, v in self._state.items()}

    @property
    def config(self):
        """Serializable manifest; persist with the SURD table for reproducibility."""
        return {**self._params, "feature_method": "basic_fixed_features_v1",
                "hash_algorithm": "SHAKE-256: byte parity -> +/-1/sqrt(hash_dim)",
                "source_salts": dict(self._salts), "source_order": list(SOURCES),
                "pad_id": -1, "unk_id": -2, "standardize_epsilon": 1e-12,
                "source_input_dims": dict(self._input_dims),
                "summary_dims": {s: 3 * d + 1 for s, d in self._input_dims.items()},
                "effective_clusters": self.effective_clusters}

    def _checked(self, history, valid_mask):
        if set(history) != set(SOURCES):
            raise ValueError("history must have exactly P, C, T, L keys")
        mask = torch.as_tensor(valid_mask).detach().cpu()
        if mask.dtype != torch.bool or mask.ndim != 2 or min(mask.shape) == 0:
            raise ValueError("valid_mask must be a nonempty bool [B,N] tensor")
        if mask.shape[1] > self._params["max_length"]:
            raise ValueError("history exceeds configured max_length")
        if not mask.any(dim=1).all() or ((~mask[:, :-1]) & mask[:, 1:]).any():
            raise ValueError("each history must be nonempty and right-padded")
        values = {}
        for source in SOURCES:
            x = torch.as_tensor(history[source]).detach().cpu()
            if x.shape[:2] != mask.shape:
                raise ValueError(f"{source} must share valid_mask's [B,N]")
            if source in SOURCES[:2]:
                if x.ndim != 2 or x.dtype not in INT_DTYPES:
                    raise TypeError(f"{source} must be integer [B,N]")
                x = x.to(torch.long)
                if ((x[mask] < 0) & (x[mask] != -2)).any() or (x[~mask] != -1).any():
                    raise ValueError(f"{source}: real IDs >=0 or UNK=-2; padding must be PAD=-1")
            else:
                if x.ndim != 3 or x.shape[2] == 0 or x.is_complex() or x.dtype == torch.bool:
                    raise TypeError(f"{source} must be numeric [B,N,d], d>0")
                x = x.to(torch.float64)
                if not torch.isfinite(x[mask]).all():
                    raise ValueError(f"{source}: encode real-event missingness with finite fills and flags")
                x = torch.where(mask[..., None], x, 0.0)
            values[source] = x
        return values, mask

    def _hashed(self, ids, mask, source):
        unique, inverse = torch.unique(ids[mask], sorted=True, return_inverse=True)
        dim, seed = self._params["hash_dim"], self._params["seed"]
        rows = []
        for identity in unique.tolist():
            key = f"{seed}|{self._salts[source]}|{identity}".encode("utf-8")
            raw = hashlib.shake_256(key).digest(dim)
            rows.append([(1.0 if byte & 1 else -1.0) / math.sqrt(dim) for byte in raw])
        out = torch.zeros(*ids.shape, dim, dtype=torch.float64)
        out[mask] = torch.tensor(rows, dtype=torch.float64)[inverse]
        return out

    @torch.no_grad()
    def summarize(self, history, valid_mask):
        """Source-isolated deterministic [last, decayed mean, mean, length ratio]."""
        values, mask = self._checked(history, valid_mask)
        length = mask.sum(dim=1)
        position = torch.arange(mask.shape[1], dtype=torch.float64)[None, :]
        age = (length[:, None] - 1 - position).clamp_min(0)
        weight = torch.exp(-age / self._params["tau_pos"]) * mask
        weight /= weight.sum(dim=1, keepdim=True)
        ratio = length[:, None].to(torch.float64) / self._params["max_length"]
        summaries = {}
        for source in SOURCES:
            x = self._hashed(values[source], mask, source) if source in SOURCES[:2] else values[source]
            last = x[torch.arange(mask.shape[0]), length - 1]
            decay = (x * weight[..., None]).sum(dim=1)
            mean = x.sum(dim=1) / length[:, None]
            out = torch.cat((last, decay, mean, ratio), dim=1)
            if not torch.isfinite(out).all():
                raise ValueError(f"{source}: summary overflow; check numeric input units")
            summaries[source] = out
        return summaries

    def _kmeans(self, x, source_index):
        unique = torch.unique(x, dim=0, sorted=True)
        k = min(self._params["n_clusters"], unique.shape[0])
        generator = torch.Generator().manual_seed(self._params["seed"] + source_index)
        centers = unique[torch.randperm(unique.shape[0], generator=generator)[:k]].clone()
        for _ in range(self._params["max_iter"]):
            labels = torch.cdist(x, centers).argmin(dim=1)
            new = centers.clone()
            for j in range(k):
                selected = labels == j
                if selected.any():
                    new[j] = x[selected].mean(dim=0)
            shift = (new - centers).abs().max().item()
            centers = new
            if shift <= self._params["tol"]:
                break
        # Remove unused clusters; no special UNK cluster is invented.
        occupied = torch.unique(torch.cdist(x, centers).argmin(dim=1), sorted=True)
        return centers[occupied]

    @torch.no_grad()
    def fit(self, history, valid_mask):
        """No labels accepted. Fit only on histories permitted by the outer split."""
        summaries = self.summarize(history, valid_mask)
        state, dims = {}, {}
        for index, source in enumerate(SOURCES):
            x = summaries[source]
            mean, scale = x.mean(dim=0), x.std(dim=0, correction=0)
            scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
            standardized = (x - mean) / scale
            if not torch.isfinite(standardized).all():
                raise ValueError(f"{source}: invalid standardized values")
            state[source] = {"mean": mean, "scale": scale,
                             "centers": self._kmeans(standardized, index)}
            dims[source] = (x.shape[1] - 1) // 3
        self._state, self._input_dims = state, dims
        return self

    @torch.no_grad()
    def transform(self, history, valid_mask):
        """Return CPU long [B,4], ordered P,C,T,L; each column starts at 0."""
        if set(self._state) != set(SOURCES):
            raise RuntimeError("fit or load the quantizer before transform")
        summaries = self.summarize(history, valid_mask)
        states = []
        for source in SOURCES:
            x, fitted = summaries[source], self._state[source]
            if x.shape[1] != fitted["mean"].numel():
                raise ValueError(f"{source}: feature dimension differs from the fitted schema")
            z = (x - fitted["mean"]) / fitted["scale"]
            if not torch.isfinite(z).all():
                raise ValueError(f"{source}: standardized values overflow")
            states.append(torch.cdist(z, fitted["centers"]).argmin(dim=1))
        return torch.stack(states, dim=1).to(torch.long)

    def save(self, path):
        if set(self._state) != set(SOURCES):
            raise RuntimeError("cannot save an unfitted quantizer")
        torch.save({"schema_version": SCHEMA, "constructor": self._params,
                    "config": self.config, "state": self._state}, Path(path))

    @classmethod
    def load(cls, path):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload["schema_version"] != SCHEMA:
            raise ValueError("unsupported quantizer schema")
        obj = cls(**payload["constructor"])
        obj._state = payload["state"]
        obj._input_dims = payload["config"]["source_input_dims"]
        if set(obj._state) != set(SOURCES) or obj.config != payload["config"]:
            raise ValueError("quantizer manifest mismatch")
        for source, state in obj._state.items():
            d = 3 * obj._input_dims[source] + 1
            if (state["mean"].shape != (d,) or state["scale"].shape != (d,)
                    or state["centers"].shape != (obj.effective_clusters[source], d)
                    or obj.effective_clusters[source] < 1):
                raise ValueError("invalid quantizer tensor dimensions")
            if any(not torch.isfinite(x).all() for x in state.values()) or (state["scale"] <= 0).any():
                raise ValueError("invalid quantizer tensor values")
        return obj
```

## 参考文献与实现来源

[1] Martínez-Sánchez, Á., Arranz, G., & Lozano-Durán, A. (2024). *Decomposing causality into its synergistic, unique, and redundant components*. Nature Communications, 15, 9296.  
[论文链接](https://www.nature.com/articles/s41467-024-53373-4)

[2] 上述论文补充材料，重点为 S11（四来源分量）、S12（specific information）、S13–S15（排序分配）、S16（目标加权）。  
[补充材料](https://static-content.springer.com/esm/art:10.1038%2Fs41467-024-53373-4/MediaObjects/41467_2024_53373_MOESM1_ESM.pdf)

[3] 作者 SURD 实现仓库及核心分配程序。本文附录是根据所声明数学规则编写的独立参考实现；近似并列的数值策略已单独说明。  
[作者仓库](https://github.com/ALD-Lab/SURD) · [核心实现](https://github.com/ALD-Lab/SURD/blob/main/src/surd/core.py)


[4] PyTorch 官方文档：`Tensor.detach`。说明停止梯度的语义及与原 tensor 共享存储的行为。  
[官方说明](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)

[5] PyTorch 官方文档：`torch.unique`、`torch.load`、`nn.Module.register_buffer`。用于稀疏状态聚合、受限对象加载与冻结统计注册。  
[unique](https://docs.pytorch.org/docs/stable/generated/torch.unique.html) · [load](https://docs.pytorch.org/docs/stable/generated/torch.load.html) · [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)

[6] scikit-learn 官方文档：`KMeans`。`fit` 拟合中心，`predict` 返回最近中心的编号；`transform` 返回到中心的距离。用于澄清第 4 节的聚类/量化关系与附录 D 的接口命名，本文实现仍为独立 PyTorch 参考代码。
[官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

[7] SciPy 官方文档：*K-means clustering and vector quantization*。说明聚类中心、codebook、最近中心分配与平方距离目标之间的关系。
[官方说明](https://docs.scipy.org/doc/scipy/reference/cluster.vq.html)

[8] scikit-learn 官方示例：用轮廓系数辅助选择 K-means 簇数。该指标反映簇内与簇间距离，不直接评价下一 POI 推荐效果。
[官方说明](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

[9] Paninski, L. (2003). *Estimation of Entropy and Mutual Information*. Neural Computation, 15, 1191–1253。用于说明有限样本下离散信息量估计的偏差与误差问题。
[论文](https://www.cns.nyu.edu/pub/eero/paninski-infoEst-2003.pdf)
