# Model settings (Added vs Basic)

**Base (authoritative):** [`TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md`](../TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md) — especially §19 starters.

This file lists **only settings that extend or change** that Base (WIN stack, Acc climb overlays, score biases, soft targets, aux losses).  
Outcomes → [`experiments.md`](experiments.md). Why → [`design-notes.md`](design-notes.md). Commands → [`runbook.md`](runbook.md).

---

## Base snapshot (do not restate the whole Basic doc)

| Item | Basic (§19 / code default) |
| --- | --- |
| Sources | P/C/T/L; **no user_id as model input** |
| MoE | 15 U/S experts; Top-K=**4**; T=1.0 start |
| Hidden | dim **128**, heads **4**, FFN 128→256→128 |
| Offline | `hash_dim=64`, `n_clusters=4`, `N_max=50`, `tau_pos=10` |
| λ | `lambda_prior=0.1`, `lambda_SURD=0.05`, `lambda_balance=0.001` |
| Optimizer | AdamW |
| Router (code) | `hard_prior` from candidate U/S SURD mass |

Anything below is **Added**.

---

## Added — LOCKED WIN stack

Recipe id: `hc_sg_Atime_bpr029_ssb026` · cache `runs/tokyo/cache_abl_tgt_last`

| Knob | Value | vs Basic |
| --- | --- | --- |
| `--dim` / `--heads` | **256** / **8** | 128 / 4 |
| `--lambda-bpr` | **0.29** | no BPR in Basic |
| `--transition-score-bias` | **2.0** | no tsb |
| `--surd-bias-gate` | `unique_p` | off |
| `--tsb-gate-mode` | `residual` | — |
| `--surd-score-bias` | **0.26** | 0 |
| `--candidate-context` | `on` | off |
| `--candidate-context-time` | `last_slot` | off |
| `--epochs` / `--checkpoint-metric` | `1` / `acc1` | protocol |

---

## Added — Acc climb peak stack (`on_rev_soft10r`)

= WIN Added **plus**:

| Knob | Value |
| --- | --- |
| `--transition-attn` | `gate` |
| `--geo-score-bias` | `0.15` |
| `--revisit-score-bias` | `0.10` |
| `--soft-target-beta` | `0.10` |
| `--soft-target-prior` | `revisit` |

Intermediate layers (superseded by peak): `transition_attn gate`+`geo 0.15` → `hc_win_trgate_gsb015`; +`revisit 0.10` → `on_rev010`.

---

## Added — other climb levers (flags only)

Tried on soft10r (= Base+WIN+peak) unless noted. Defaults are **off / 0**.

| Family | Flags |
| --- | --- |
| Soft-target variants | `--soft-target-beta` · `--soft-target-prior` ∈ {revisit, tsb, tsb+revisit, ema} |
| Score biases | `--revisit-score-bias` · `--geo-score-bias` · `--geo-score-power` · `--hour-pop-score-bias` · `--transition-2hop-bias` · `--cat-stage-bias` · `--history-hit-bias` · `--revisit-recency-bias` · radiation / jump / explore / gauss / … |
| SURD / tsb gates | `--surd-bias-gate` · `--surd-score-bias` · `--tsb-gate-mode` · `--surd-attn-gate` · `--tsb-session-gap-hours` · `--tsb-decay-hours` |
| Aux / loss | `--lambda-bpr` · `--lambda-cat-aux` · `--lambda-kd` · `--poi-geo-laplace` · focal / BPR-max |
| Eval-only | `--score-temperature` · Platt / gamma calib · `--mc-dropout` · TTA |
| Protocol | `--epochs` >1 · seed soup · fam-area-blend · adaptive neg |

Feature / overlay vocabulary (A/B/C/D/E, causal counts, routers): see [`candidate-context.md`](candidate-context.md), [`causal-counts.md`](causal-counts.md), [`router-modes.md`](router-modes.md).
