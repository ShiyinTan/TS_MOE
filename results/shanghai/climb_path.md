# Shanghai Acc@1 climb path (LOCKED · STOPPED)

**Protocol:** `known_only` Acc@1 · **eval_test**  
**STHGCN:** 0.193 · **Our Acc@1:** **0.196226** (+0.0032) · MRR ~0.258  
**Config:** `configs/shanghai/climb_soft005_filter_last.yaml`  
**Source:** `results/shanghai/climb_soft005_filter_last/eval_test/` (tie: `tune_soft005`)  
**Status:** climb **STOPPED** — do not resume densify unless asked.

## Beat ladder

| Stage | Acc@1 | vs STHGCN | Key change |
| ---: | ---: | ---: | --- |
| base WIN lineage | ~0.193 | ~0 | `configs/shanghai/win.yaml` stack |
| **WIN** | **0.196226** | **+0.0032** | softβ **0.05** · filter-unknown / last-target cache |

Canonical multi-city summary: [`docs/multi-city-sthgcn-climb-history.md`](../../docs/multi-city-sthgcn-climb-history.md).
