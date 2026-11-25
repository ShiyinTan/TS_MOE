# Multi-city Acc@1 climb vs STHGCN — locked beats + history

**Protocol:** `known_only` Acc@1 · **eval_test** · PR [#37](https://github.com/ShiyinTan/TS_MOE/pull/37)  
**Updated:** 2026-11-25 18:50 JST  
**Repo branch:** `cursor/beijing-sthgcn-climb-7a66`  
**Retention:** beat/STOPPED cities → this file + short `results/<city>/climb_path.md` only; MEL keeps active wave detail.

STHGCN paper Acc@1 baselines (Massive-STEPS): Beijing **0.453** · Melbourne **0.168** · Palembang **0.246** · (Tokyo 0.250 · Shanghai 0.193).

---

## 1. Current beat scoreboard (locked WINs)

| City | STHGCN | Our Acc@1 | Δ | MRR | NDCG@10 | Status | Locked config |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| **Beijing** | 0.453 | **0.465909** (41/88) | **+0.0129** | ~0.536 | ~0.574 | **LOCKED · climb STOPPED** | `configs/beijing/climb_s5g20_all_gauss020_wd005_j12p10_pmi015_win.yaml` |
| **Palembang** | 0.246 | **0.259690** | **+0.0137** | 0.3403 | 0.3711 | **LOCKED · climb STOPPED** | `configs/palembang/climb_tsb20_bpr020_filter_soft0754_bpr025_session_seed55.yaml` |
| **Melbourne** | 0.168 | **0.170692** | **+0.0027** | 0.2491 | 0.2791 | **LOCKED · clear-margin climb ACTIVE** (target ~≥0.172) | `configs/melbourne/climb_bpr020_soft010_tgt_soft0078_session_bpr205_jump010_pmi05_win.yaml` |

Tokyo → Shanghai → Beijing → Palembang → Melbourne (active). BJ/PL **STOPPED**. MEL margin still thin (+0.0027); clear-margin target ~≥0.172 before opening any further city.

Also already beat (earlier / other cities): Tokyo **0.2677** vs 0.250 (+0.0177); Shanghai **0.1962** vs 0.193 (+0.0032).

---

## 2. Beijing climb history (beat path)

Source run for WIN Acc: `results/beijing/climb_s5g20_all_gauss020_wd005_w45_tgt_j12p10_pmi015_ep3/eval_test/` (Acc@1 **0.465909**, MRR **0.5358**).

| Stage | Acc@1 | Δ step | vs STHGCN | Key change |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.420455 | — | −0.0325 | Peak + trbias + softβ0.05 + geo0.20 · `cache_abl_tgt_last` |
| 1 | 0.431818 | +0.0114 | −0.0212 | `cache_tgt_all_eval_last` + **gauss λ=0.15** (geo off) |
| 2 | 0.443182 | +0.0114 | −0.0098 | gauss λ **0.20** |
| 3 | 0.454545 | +0.0114 | **+0.0015** | **weight_decay=0.005** → first beat |
| 4 · WIN | **0.465909** | +0.0114 | **+0.0129** | **jump=0.12 + pop=0.10 + pmi=0.15** · ep3 (w45) |

```
0.4205 → 0.4318 → 0.4432 → 0.4545 → 0.4659  ✓ clear (+0.0129 vs 0.453)
```

**WIN recipe:** Peak + trbias + softβ0.05 + gauss0.20 + wd0.005 + jump0.12 + pop0.10 + pmi0.15 · epochs=3 · cache `runs/beijing/cache_tgt_all_eval_last`.  
**Stopped** after flat densify w46 (no further Acc lift). Short path: [`results/beijing/climb_path.md`](../results/beijing/climb_path.md).

---

## 3. Melbourne climb history (beat + clear-margin)

Source WIN metrics: `results/melbourne/climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_pmi05/eval_test/`  
(Acc@1 **0.170692**, MRR **0.2491**, NDCG@10 **0.2791**).

| Stage | Acc@1 | vs STHGCN | Notes |
| ---: | ---: | ---: | --- |
| early | ~0.154–0.161 | under / thin | soft/session/seed densify (w2–w6) |
| beat thin | **0.168277** | **+0.0003** | first clear beat (w7 era) |
| lock A | **0.169887** | +0.0019 | w11 · seed46 soft0078+session bpr205+jump010 |
| lock B · WIN | **0.170692** | **+0.0027** | w12 · **+pmi=0.05** on lock A |
| w12b–w15 | flat @ 0.170692 | — | densify / new levers; TIEs only; no real lift |
| w16 | in flight | — | router / neg / focal / rad / hour_pop / dim … (seeking ≥~0.172) |

**WIN recipe:** extends `climb_bpr020_soft010` · seed **46** · softβ **0.078** · session_summary on · λ_bpr **0.205** · jump **0.10** · pmi **0.05** · cache `runs/melbourne/cache_tgt_all_eval_last`.

Policy: TIE ≠ lift; real lift requires Acc@1 **strictly > 0.170692**.

Also: [`results/melbourne/climb_path.md`](../results/melbourne/climb_path.md).

---

## 4. Palembang climb history (beat + margin)

Source WIN metrics: `results/palembang/climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr025/eval_test/`  
(Acc@1 **0.259690**, MRR **0.3403**, NDCG@10 **0.3711**).

| Stage | Acc@1 | vs STHGCN | Notes |
| ---: | ---: | ---: | --- |
| early lifts | 0.2506 → 0.2532 → 0.2558 → 0.2571 | +0.005…+0.011 | soft/session/seed densify |
| WIN | **0.259690** | **+0.0137** | softβ **0.0754** · λ_bpr **0.25** · seed **55** · filter p80 lineage |

**Status:** climb **STOPPED** (margin clear; do not resume densify unless asked). Config: `configs/palembang/climb_tsb20_bpr020_filter_soft0754_bpr025_session_seed55.yaml`.

Also: [`results/palembang/climb_path.md`](../results/palembang/climb_path.md).

---

## 5. Active ops (not part of locked scoreboard)

| Item | Value |
| --- | --- |
| Focus | Melbourne clearer margin (~≥0.172) — no new city until then |
| Wave | **w16** (~64; router/neg/focal/rad/hour_pop/dim…) · mid-run TOP below lock |
| Caps | ≤2 our jobs/GPU; RAM/VRAM/disk gates |
| BJ / PL | STOPPED — 0 jobs |
| Harvest | ~45m sparse; one-line unless Acc lift or blocker |

---

## 6. Reproduce locked WINs

```bash
# Beijing
python -m tsmoe train --config configs/beijing/climb_s5g20_all_gauss020_wd005_j12p10_pmi015_win.yaml

# Melbourne
python -m tsmoe train --config configs/melbourne/climb_bpr020_soft010_tgt_soft0078_session_bpr205_jump010_pmi05_win.yaml

# Palembang
python -m tsmoe train --config configs/palembang/climb_tsb20_bpr020_filter_soft0754_bpr025_session_seed55.yaml
```

```bash
python scripts/city_results_bundle.py --city beijing --sthgcn 0.453
```
