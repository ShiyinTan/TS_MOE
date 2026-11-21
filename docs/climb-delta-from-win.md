# Acc@1 climb — deltas vs LOCKED WIN

**Base (WIN):** `hc_sg_Atime_bpr029_ssb026` · known_only Acc@1 **0.260274**  
cache `runs/tokyo/cache_abl_tgt_last` · seed 42 · ep1 · `--checkpoint-metric acc1`

```
--router-mode hard_prior --dim 256 --heads 8
--lambda-bpr 0.29
--transition-score-bias 2.0
--surd-bias-gate unique_p --tsb-gate-mode residual --surd-score-bias 0.26
--candidate-context on --candidate-context-time last_slot
```

All climb runs below **inherit Base** and only list **Added** flags.

---

## Keep / peak

### `on_rev_soft10r` — Acc@1 **0.267650** · **Keep / locked peak**

| | |
| --- | --- |
| **Base** | WIN above |
| **Added** | `--transition-attn gate` · `--geo-score-bias 0.15` · `--revisit-score-bias 0.10` · `--soft-target-beta 0.10 --soft-target-prior revisit` |
| Δ vs WIN | +0.007376 Acc@1 |

Path: `runs/tokyo/on_rev_soft10r/best.pt` · `results/tokyo/on_rev_soft10r/eval_test/`

### Prior Keep (superseded by soft10r)

| run | Acc@1 | **Added** on WIN |
| --- | ---: | --- |
| `hc_win_trgate_gsb015` | 0.263435 | `transition_attn gate` + `geo 0.15` |
| `on_rev010` | 0.265543 | above + `revisit 0.10` |

---

## Leftover / overnight Discard (selected)

All on soft10r stack (= WIN + soft10r **Added**). Extra lever only:

| run | Acc@1 | Extra vs soft10r peak | 判定 |
| --- | ---: | --- | --- |
| `on_s10r_cat01` | 0.2571 | `--lambda-cat-aux 0.1` (#45) | Discard |
| `on_s10r_cat03` | 0.2561 | `--lambda-cat-aux 0.3` (#45) | Discard |
| `on_s10r_ep2b` | 0.267650 (ep1 best; ep2 0.2645) | `--epochs 2` (#47) | Discard (no lift) |
| `on_s10r_ep3b` | 0.267650 (ep1; ep3 0.237) | `--epochs 3` (#47) | Discard |
| `on_s10r_gp15` | 0.2634 | `--geo-score-power 1.5` (#6) | Discard |
| `on_s10r_gp20` | 0.2561 | `--geo-score-power 2.0` (#6) | Discard |
| MC mc10/mc20 | 0.2655 / 0.2634 | eval `--mc-dropout` (#46) | Discard |

Waves 13–24 (hit/recency, session-gap, ε/TTA, filters, adaptive neg, EMA, micro ssb/tsb, KD/laplace, soup, fam): all Discard ≤peak — see store `overnight-status.md`.

---

## Still queued

**Added** candidates (one-at-a-time on soft10r): `#4 --transition-2hop-bias` · `#8 --hour-pop-score-bias` · `#9 --cat-stage-bias` · `#11` learned bilinear · `#5 --score-temperature` / Platt on frozen peak.
