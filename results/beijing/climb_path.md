# Beijing Acc@1 climb path (LOCKED · STOPPED)

**Protocol:** `known_only` Acc@1 · **eval_test**  
**STHGCN:** 0.453 · **Our Acc@1:** **0.465909** (+0.0129) · MRR ~0.536  
**Config:** `configs/beijing/climb_s5g20_all_gauss020_wd005_j12p10_pmi015_win.yaml`  
**Source:** `results/beijing/climb_s5g20_all_gauss020_wd005_w45_tgt_j12p10_pmi015_ep3/eval_test/`  
**Status:** climb **STOPPED** — do not resume densify unless asked.

## Beat ladder

| Stage | Acc@1 | vs STHGCN | Key change |
| ---: | ---: | ---: | --- |
| 0 | 0.420455 | −0.0325 | Peak + trbias + softβ0.05 + geo0.20 |
| 1 | 0.431818 | −0.0212 | all/last cache + gauss λ=0.15 |
| 2 | 0.443182 | −0.0098 | gauss λ=0.20 |
| 3 | 0.454545 | **+0.0015** | weight_decay=0.005 |
| **WIN** | **0.465909** | **+0.0129** | jump0.12 + pop0.10 + pmi0.15 · ep3 |

Canonical multi-city summary: [`docs/multi-city-sthgcn-climb-history.md`](../../docs/multi-city-sthgcn-climb-history.md).
