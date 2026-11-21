# Experiments — settings, Acc/R, Keep/Discard

**Protocol:** Tokyo · judge **`known_only` Acc@1** (Hit@1 / Recall@1) unless noted · cache usually `runs/tokyo/cache_abl_tgt_last` · seed 42 · ep1 · `--checkpoint-metric acc1`.  
**Base model settings:** [`../TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md`](../TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md).  
**Added knobs:** [`model-settings.md`](model-settings.md). **Why:** [`design-notes.md`](design-notes.md). **Commands:** [`runbook.md`](runbook.md).

Metrics live under `results/<run>/eval_*/{metrics.json,report.txt}` (gitignored). Checkpoints under `runs/`.

---

## Locked recipes (Keep)

| run | Acc@1 | R@5 | R@10 | R@20 | stack |
| --- | ---: | ---: | ---: | ---: | --- |
| STHGCN (ref) | 0.255788 | 0.441345 | 0.511687 | 0.574311 | — |
| `hc_sg_Atime_bpr029_ssb026` **WIN** | **0.260274** | 0.441517 | 0.519494 | 0.586934 | Base + WIN Added |
| `hc_win_trgate_gsb015` | 0.263435 | — | — | — | WIN + gate + geo0.15 |
| `on_rev010` | 0.265543 | — | — | — | above + revisit0.10 |
| `on_rev_soft10r` **peak** | **0.267650** | — | — | — | above + soft β0.10 prior=revisit |

Keep rule for leftovers: Acc@1 **>** peak → Keep and restack; else Discard.

---

## Acc climb leftovers on soft10r

Extra = only the lever under test on the peak stack.

| # / run | Extra | Acc@1 | 判定 |
| --- | --- | ---: | --- |
| soft05 / soft10 / soft10t | soft β/prior ≠ revisit@0.10 | ≤peak | Discard |
| `on_rev020` | revisit 0.20 | 0.2476 | Discard |
| #32 | history-hit / revisit-recency | 0.2624 | Discard |
| #33 | tsb session-gap | =peak | Discard |
| #38/#34/#36 | locked-ε / gamma / TTA | ≤peak | Discard |
| #10 | candidate spatial prior | cov≪0.98 | Discard |
| #37 | adaptive neg | ≤0.258 | Discard |
| #39 | EMA soft | 0.2561 | Discard |
| #43 | ssb/tsb micro | 0.2624 | Discard |
| #41/#40 | KD / geo-laplace | ≤0.259 | Discard |
| #42 | seed soup | no lift | Discard |
| #44 | fam-area-blend | 0.2561 | Discard |
| #46 | MC-dropout 10/20 | 0.2655 / 0.2634 | Discard |
| #45 `on_s10r_cat01/03` | λ_cat_aux 0.1/0.3 | 0.257 / 0.256 | Discard |
| #47 ep2/ep3 | epochs=2/3 | ep1=peak; later↓ | Discard |
| #6 `on_s10r_gp15/20` | geo-power 1.5/2.0 | 0.2634 / 0.2561 | Discard |
| #8 `on_s10r_hpop10/15` | hour-pop 0.10/0.15 | **0.2518 / 0.2624** | **Discard** |
| #4 `on_s10r_2hop03/05` | transition-2hop 0.3/0.5 | **0.2540 / 0.2466** | **Discard** |
| #9 `on_s10r_cstage01/02` | cat-stage-bias 0.1/0.2 | **0.2508 / 0.2413** | **Discard** |
| #11 | bilinear transition | — | not coded → skipped |
| #5 `on_s10r_T0.7` | eval `--score-temperature 0.7` | **0.267650** (=peak) | **Discard** (flat) |
| #5 T∈{0.85,1.15,1.3} | eval T on frozen soft10r | — | **running** |

Earlier Discard (radiation / jump / return-gate / Δt-bins / cat-stage soft / focal / BPR-max / tsb-decay / …): Acc ≪ peak — do not restack as-is.

---

## Results layout (short)

| Path | Contents |
| --- | --- |
| `results/<run>/eval_test/metrics.json` | Final test (read `known_only`) |
| `results/<run>/eval_validation/` | Best-ckpt val |
| `results/ablation_log.md` / `.csv` | Auto-append after train/evaluate |
| `python -m tsmoe ablation-summary` | Master table over `results/**` |

`known_only` = GT next POI **in the training catalog** (not “in candidate pool”; GT is always forced into the pool). Distinct from `prepare --filter-unknown on`.
