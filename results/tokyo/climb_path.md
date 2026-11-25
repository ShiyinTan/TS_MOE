# Tokyo Acc@1 climb path (LOCKED · STOPPED)

**Protocol:** `known_only` Acc@1 · **eval_test**  
**STHGCN:** 0.250 · **Our Acc@1:** **0.267650** (+0.0177)  
**Peak config:** `configs/tokyo/peak.yaml` · out `runs/tokyo/on_rev_soft10r`  
**WIN (first beat):** `configs/tokyo/win.yaml` · Acc@1 **0.260274** · `results/tokyo/hc_sg_Atime_bpr029_ssb026/eval_test/`  
**Status:** climb **STOPPED** — do not resume densify unless asked.

## Beat ladder

| Stage | Acc@1 | vs STHGCN | Key change |
| ---: | ---: | ---: | --- |
| WIN | 0.260274 | +0.0103 | dim256 + BPR0.29 + tsb2.0 + ssb0.26 · `cache_abl_tgt_last` |
| +1 | 0.263435 | +0.0134 | + transition-attn gate + geo λ=0.15 |
| +2 | 0.265543 | +0.0155 | + revisit λ=0.10 |
| **peak** | **0.267650** | **+0.0177** | + soft-target β=0.10 prior=revisit |

Canonical multi-city summary: [`docs/multi-city-sthgcn-climb-history.md`](../../docs/multi-city-sthgcn-climb-history.md).
