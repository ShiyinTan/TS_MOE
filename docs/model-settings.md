# Model settings (Added vs Basic)

**Base (authoritative):** [`TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md`](../TS_MoE_SURD_POI_Category_Time_Location_Basic_Updated.md) — especially §19 starters; T/L STHGCN schema in §4.3.

This file lists **only settings that extend or change** that Base (router alternatives, WIN stack, Acc climb overlays, score biases, soft targets, aux losses).  
Outcomes → [`experiments.md`](experiments.md). Why → [`design-notes.md`](design-notes.md). Commands → [`runbook.md`](runbook.md).

**Launch via config (preferred):** `configs/recipes/{base,win,peak}.yaml` plus `configs/<city>/{base,win,peak}.yaml` (city sets `cache` / `out`).  
`python -m tsmoe train --config configs/<city>/<recipe>.yaml` — see [`runbook.md`](runbook.md).

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
| T/L features | `sthgcn_tl_features_v1` (Basic §4.3) |
| Causal counts | **on**; `--transition-attn bias` (see [`causal-counts.md`](causal-counts.md)) |

Anything below is **Added** (or an alternate router).

---

## Router modes (`--router-mode`)

Mode is set at `train`, stored in checkpoint `config`, reused by `evaluate` / `explain` / `recommend`. Offline SURD: one train-all 26-D `routing_prior` per POI (`final.surd.pt`); train/val/test share it.

| CLI | Default | Meaning |
| --- | --- | --- |
| `--router-mode` | `hard_prior` | `hard_prior` \| `surd_mlp` \| `hybrid` |
| `--lambda-prior` | `0.1` | Scale on explicit SURD bias in **hybrid** |
| `--hybrid-lambda` | `0.1` | ε on the learned term in **hybrid** |
| `--temperature` | `1.0` | Softmax temperature for all modes |
| `--no-surd` | off | Zeros confidence → hard_prior → uniform π |

Online: **U/S mass** (15-D) = `concat(prior[..., :4], prior[..., 15:])`; redundancy `ρ` and `α = mass.sum` feed the residual gate / SURD MLP, not the 15 logits directly; confidence `ω` scales SURD inputs.

### `hard_prior` (Basic / default)

```text
surd_bias = ω · log1p(15 · mass)     # [B, M, 15], no sequence
logits    = surd_bias
π         = softmax(logits / T)
```

### `surd_mlp` (Added alternate)

```text
surd_feats = [prior_mlp(ω · prior26), ω·ρ, ω·α]   # 66-D, no sequence
logits     = MLP(surd_feats)                        # → 15
π          = softmax(logits / T)
```

### `hybrid` (Added alternate)

```text
surd_bias = ω · log1p(15 · mass)
learned   = RouterMLP(context, candidate, SURD feats)
logits    = λ_prior · surd_bias + hybrid_lambda · learned
π         = softmax(logits / T)
```

Legacy checkpoints without `router_mode` load as `hybrid` with `hybrid_lambda=1.0`. Demo/Tokyo copy-paste: [`../README.md`](../README.md) §3.6.

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

## Added — candidate-context overlays (A–E)

Runtime overlays; **all default `off`**. New signals enter **base / shared query / expert attn**, never `hard_prior` router logits. Full defaults matrix + behavior → [`candidates.md`](candidates.md) §Overlays. WIN uses A + `last_slot` only.

| Flag | Default | Needs |
| --- | --- | --- |
| `--candidate-context {off,on}` | `off` | — |
| `--candidate-context-time {off,last_slot,causal_slot}` | `off` | `causal_slot` → causal on |
| `--session-summary {off,on}` | `off` | — |
| `--user-pref {off,on}` | `off` | causal on |
| `--user-id-emb {off,on}` | `off` | — |
| `--transition-pmi {off,on}` | `off` | causal on |
| `--expert-cross-attn {off,hybrid,full}` | `off` | — |
| `--expert-cross-attn-sparse {off,on}` | `on` (when E ≠ off) | — |

---

## Added — other climb levers (flags only)

Tried on soft10r (= Base+WIN+peak) unless noted. Defaults are **off / 0**. Mechanism detail for causal score/attn biases → [`causal-counts.md`](causal-counts.md).

| Family | Flags |
| --- | --- |
| Soft-target variants | `--soft-target-beta` · `--soft-target-prior` ∈ {revisit, tsb, tsb+revisit, ema} |
| Score biases | `--revisit-score-bias` · `--geo-score-bias` · `--geo-score-power` · `--hour-pop-score-bias` · `--transition-2hop-bias` · `--pmi-score-bias` · `--cat-stage-bias` · `--history-hit-bias` · `--revisit-recency-bias` · radiation / jump / explore / gauss / … |
| SURD / tsb gates | `--surd-bias-gate` · `--surd-score-bias` · `--tsb-gate-mode` · `--surd-attn-gate` · `--tsb-session-gap-hours` · `--tsb-decay-hours` |
| Aux / loss | `--lambda-bpr` · `--lambda-cat-aux` · `--lambda-kd` · `--poi-geo-laplace` · focal / BPR-max |
| Eval-only | `--score-temperature` · Platt / gamma calib · `--mc-dropout` · TTA |
| Protocol | `--epochs` >1 · seed soup · fam-area-blend · adaptive neg |

Feature deep-dives (do not duplicate recipes here):

| Topic | Doc |
| --- | --- |
| Candidate pools + overlays A–E | [`candidates.md`](candidates.md) |
| Causal counts / tsb / SURD gates | [`causal-counts.md`](causal-counts.md) |
| T/L STHGCN features | Basic §4.3 |

**Leftover Acc queue exhausted** at peak `on_rev_soft10r` — outcomes in [`experiments.md`](experiments.md). `#5 --score-temperature` is Acc@1-invariant (scores÷T). `#11` bilinear not implemented.
