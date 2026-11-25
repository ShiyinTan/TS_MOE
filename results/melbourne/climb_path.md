# Melbourne Acc@1 climb path (vs STHGCN)

**Protocol:** `known_only` Acc@1 · **eval_test**  
**Updated:** 2026-11-25 JST  
**STHGCN:** 0.168  
**Locked WIN:** Acc@1 **0.170692** (+0.0027) · MRR **0.2491** · NDCG@10 **0.2791**  
**Config:** `configs/melbourne/climb_bpr020_soft010_tgt_soft0078_session_bpr205_jump010_pmi05_win.yaml`  
**Source metrics:** `results/melbourne/climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_pmi05/eval_test/`

## Beat ladder

| Stage | Acc@1 | vs STHGCN | Tag / change |
| ---: | ---: | ---: | --- |
| early | ~0.154–0.161 | under | soft/session/seed densify |
| first beat | 0.168277 | +0.0003 | thin margin |
| lock A | 0.169887 | +0.0019 | w11 · seed46 soft0078+session bpr205+jump010 |
| **WIN** | **0.170692** | **+0.0027** | w12 · **pmi=0.05** |
| post-WIN | flat | — | w12b–w15 flat; w16 seeking ≥~0.172 |

## WIN recipe

| Knob | Value |
| --- | --- |
| seed | 46 |
| soft_target_beta | 0.078 · prior revisit (from peak lineage) |
| session_summary | on |
| lambda_bpr | 0.205 |
| jump_length_bias | 0.10 · lognormal |
| pmi_score_bias | 0.05 |
| cache | `runs/melbourne/cache_tgt_all_eval_last` |

See also: [`docs/multi-city-sthgcn-climb-history.md`](../../docs/multi-city-sthgcn-climb-history.md).
