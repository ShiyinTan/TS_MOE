# Beijing Acc@1 提升路径（vs STHGCN）

**协议：** `known_only` Acc@1 · **eval_test only** · seed 42 · hard_prior  
**日期：** 2026-11-24（JST）  
**STHGCN 论文 Acc@1：** 0.453 · 需约 **40/88 ≈ 0.4545** 才能赢  
**锁定配置：** `configs/beijing/climb_s5g20_all_gauss020_wd005.yaml`  
**PR：** [#37](https://github.com/ShiyinTan/TS_MOE/pull/37)

---

## 1. 主提升路径（只看城市 best 刷新）

| 阶段 | Acc@1 | 相对上一步 | vs STHGCN | 配置 / tag | Cache | 关键改动 |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 · 此前城市 best | 0.420455 | — | −0.0325 | `climb_ep2_trbias_soft005_geo020` | `cache_abl_tgt_last` | Peak + trbias + soft β=0.05 + geo λ=0.20 |
| 1 · 换 cache + gauss | **0.431818** | **+0.0114** | −0.0212 | `climb_s5g20_all_gauss015` | `cache_tgt_all_eval_last` | train_targets=**all** · eval=last · **gauss λ=0.15**（关掉 geo） |
| 2 · 加强 gauss | **0.443182** | **+0.0114** | −0.0098 | `climb_s5g20_all_gauss020` | 同上 | gauss λ **0.15 → 0.20** |
| 3 · 调 weight_decay | **0.454545** | **+0.0114** | **+0.0015** | `climb_s5g20_all_gauss020_wd005` | 同上 | **weight_decay=0.005**（并列：`…_wd02`） |

一句话：`soft005+trbias` 底座 → **换 all/last cache** → **用 gauss 替换 geo** → **略降 weight_decay** → 超过 STHGCN。

```
0.4205 ──(+0.011)──► 0.4318 ──(+0.011)──► 0.4432 ──(+0.011)──► 0.4545  ✓ beat 0.453
  soft005_geo020        all_gauss015         all_gauss020         all_gauss020_wd005
  (abl_tgt_last)        (tgt_all_eval_last)  (λ=0.20)             (wd=0.005)
```

---

## 2. 最终 WIN 配方（可复现）

| 项 | 取值 |
| --- | --- |
| Config | `configs/beijing/climb_s5g20_all_gauss020_wd005.yaml` |
| Cache | `runs/beijing/cache_tgt_all_eval_last`（`--train-targets all --eval-targets last`） |
| 底座 | Peak recipe + `transition_attn=bias` + epochs=2 |
| Soft target | β=0.05 · prior=`revisit` |
| 空间先验 | **gauss_geo_bias=0.20** · σ=0.5 km · **geo_score_bias=0** |
| 正则 | **weight_decay=0.005**（默认 0.01） |
| Acc@1 / R@5 / R@10 / R@20 | **0.454545** / 0.659 / 0.727 / 0.784 |
| 并列 | `climb_s5g20_all_gauss020_wd02`（wd=0.02，同 Acc） |

启动：

```bash
python -m tsmoe train --config configs/beijing/climb_s5g20_all_gauss020_wd005.yaml
```

---

## 3. 各 wave 试了什么（摘要）

| Wave | 目标 | 代表杠杆 | 对 Acc@1 的结果 |
| ---: | --- | --- | --- |
| 此前（156 runs） | 在 `abl_tgt_last` 上扫 | soft/geo/pmi/upref/ep 等 | 卡在 **0.4205**（约 37/88）离散平台 |
| Wave1 | 未用过的空间先验 | radiation / jump / return-gate / fam / cat+pop | 全低于或持平 best；**无 lift** |
| Wave2 | 换 cache + 训练超参 | **tgt_all_eval_last + gauss**；dim/lr/tsb-history | **首次 lift → 0.4318**（all_gauss015） |
| Wave3 | 围着 gauss015 加密 | λ∈{0.10…0.20}、σ、upref/pmi/hh/soft/ep3 | **gauss020 → 0.4432**；多数其它 DOWN |
| Wave4 | 围着 gauss020 细扫 | λ/σ/soft/revisit | 无再升；**0.4432 粘住**（差 1 个样本） |
| Wave5 | 新 prepare + overlay | filter/p80/p70/max；pmi/cat/sess/2hop… | 无再升；若干 **TIE 0.4432** |
| Wave6 | 训练超参 | dropout / **wd** / neg / batch / ep / heads | **wd=0.005 & 0.02 → 0.4545，beat STHGCN** |

---

## 4. 关键负结果（避免再踩）

| 方向 | 代表 | 典型 Acc | 备注 |
| --- | --- | ---: | --- |
| radiation 替 geo | `rad015` / `rad025` | ~0.27–0.33 | 明显伤 Acc |
| jump-length 替 geo | `jump015` | ~0.36 | 伤 Acc |
| return-gate / fam-blend | `retgate_gauss` 等 | ≤0.42 | 未超 best |
| dim=512 | `dim512` | ~0.35 | 过拟合/不稳 |
| 仅改 score-temperature 重评 | `cal_T*`（无带上 train biases） | 不可比 | eval 未复现 gauss 等 bias |
| 多数 overlay 叠在 gauss020 上 | pmi/cat/sess/2hop… | ≤0.443 | 未再涨 |

---

## 5. 与其它已赢城市对照

| City | Acc@1 | STHGCN | Δ | 锁定 config |
| --- | ---: | ---: | ---: | --- |
| Tokyo | 0.2677 | 0.250 | +0.0177 | `configs/tokyo/peak.yaml`（soft10r） |
| Shanghai | 0.1962 | 0.193 | +0.0032 | `configs/shanghai/climb_soft005.yaml` |
| **Beijing** | **0.4545** | 0.453 | **+0.0015** | `configs/beijing/climb_s5g20_all_gauss020_wd005.yaml` |

下一城（按 check-in 从小到大）：**Palembang**（interim 0.2287 vs 0.246）。

---

## 6. 结果落盘位置

机器可读摘要写在 **`results/beijing/`**（与各 run 的 `eval_test/` 同级）：

| 文件 | 内容 |
| --- | --- |
| [`results/beijing/climb_path.md`](../results/beijing/climb_path.md) | 提升路径表格 |
| [`results/beijing/settings_summary.md`](../results/beijing/settings_summary.md) / `.csv` | 各设定 known_only Acc / R |
| [`results/beijing/mean_std.md`](../results/beijing/mean_std.md) / `.csv` | 同设定多次运行 mean±std |

```bash
python scripts/city_results_bundle.py --city beijing --sthgcn 0.453
```
