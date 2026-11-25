# Palembang Acc@1 climb path (vs STHGCN)

**Protocol:** `known_only` Acc@1 · **eval_test**  
**Updated:** 2026-11-25 JST  
**STHGCN:** 0.246  
**Locked WIN:** Acc@1 **0.259690** (+0.0137) · MRR **0.3403** · NDCG@10 **0.3711**  
**Config:** `configs/palembang/climb_tsb20_bpr020_filter_soft0754_bpr025_session_seed55.yaml`  
**Source metrics:** `results/palembang/climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr025/eval_test/`  
**Status:** climb **STOPPED** (clear margin; do not resume densify unless asked)

## Beat ladder

| Stage | Acc@1 | vs STHGCN | Notes |
| ---: | ---: | ---: | --- |
| early | 0.2506 → 0.2532 → 0.2558 → 0.2571 | +0.005…+0.011 | soft/session/seed densify |
| **WIN** | **0.259690** | **+0.0137** | softβ **0.0754** · λ_bpr **0.25** · seed **55** |

## WIN recipe

| Knob | Value |
| --- | --- |
| soft_target_beta | 0.0754 |
| lambda_bpr | 0.25 |
| seed | 55 |
| lineage | filter p80 · tsb20 · session |

See also: [`docs/multi-city-sthgcn-climb-history.md`](../../docs/multi-city-sthgcn-climb-history.md).
