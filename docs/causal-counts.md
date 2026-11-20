# Causal online count tables + transition cross-attn

Runtime feature (no prepare / fingerprint change). Default **on**.

## What is stored

Counts of ground-truth next hops observed **before** the current prediction time:

| # | Table | Scope |
| ---: | --- | --- |
| 1 | user → POI visits | per user |
| 2 | user → category visits | per user |
| 3 | global POI visits | shared |
| 4 | global category visits | shared |
| 5 | global POI → POI transitions | last history POI → next POI |
| 6 | global category → category transitions | last history cat → next cat |
| 7 | global category → POI transitions | last history cat → next POI |

## Protocol (causal: score first, then update)

1. Sort samples by **prediction timestamp** ascending (**no shuffle** when enabled).
2. For each sample: **read** tables → build features / attention → forward (and train: backward) → **update** tables with the true next POI and category (and the last-history → next transitions).
3. **Val / test:** start from a **clone of the tables at end of training** (checkpoint `count_tables`, or replay train in order), then continue the same chronological score→update loop. Val and test each clone train-end independently (test does not inherit val updates).

Each train epoch resets tables to empty and replays the stream; visit-bin edges stay frozen from the first train fit.

## Features into embeddings

- **POI embedding** (history + candidates): add MLP over scaled features from user→POI and global POI counts:  
  `log1p(user)`, `log1p(global)`, `user/(user+global+ε)`, `log1p(user)-log1p(global)`.
- **Category embedding** (history): same with user→cat and global cat counts.
- **T and L:** do **not** use raw user visit counts. Use **equal-frequency bins** of the user’s **cumulative visit count** at prediction time. Fit bin edges on the train stream; freeze for val/test. Bin id → embedding → added to projected T and L (alongside existing STHGCN T/L). **Default bin count: 10** (`--visit-bins`).

## Transition → cross-attn (`--transition-attn`)

Only **P** and **C** sources use transition-aware cross-attn (candidate ← history). **T/L** keep plain multi-head attention.

| Mode | Plain-language effect | CLI |
| --- | --- | --- |
| **`bias` (DEFAULT)** | Add `λ · log1p(transition_count)` to attention logits for history step *j* vs candidate *c*. | `--transition-attn bias` |
| **`gate`** | A small MLP + sigmoid on `[history state; candidate; φ(count)]` scales attention weights (then renormalize). | `--transition-attn gate` |
| **`rel_key`** | Inject `MLP(φ(transition))` into the **keys** (candidate-dependent). | `--transition-attn rel_key` |

**Which transition table feeds which source (documented choice):**

- **P-source:** global **POI → POI** counts (history POI → candidate POI).
- **C-source:** global **category → POI** counts (history category → candidate POI).

(`category → category` is maintained for completeness / future use; C-source attn uses cat→POI so the candidate side stays POI-aligned.)

`φ(count) = log1p(count)`. Scale for bias mode: `--transition-lambda` (default `1.0`).

## Score-level transition prior (`--transition-score-bias`)

Optional **Acc@1** prior (default **0 = off**), independent of `--transition-attn`:

\[
\text{score}_c \mathrel{+}= \lambda \cdot \log(1 + \text{count}(\text{last history POI} \to c))
\]

Needs causal counts on. Tokyo hillclimb WIN used `--transition-score-bias 2.0` with `dim 256`, `--lambda-bpr 0.25`, `--epochs 1`, `--checkpoint-metric acc1` on a last/last multi-source cache.

## Extra score-level graph / feature priors (default **0 = off**)

All are Acc@1 score additives on top of the model; **no re-prepare**. Defaults keep the TM-34 WIN path unchanged.

| Flag | Formula (added to score) | Needs |
| --- | --- | --- |
| `--category-score-bias` | \(\lambda\cdot\log(1+\text{count}(\text{last cat}\to c))\) via `cat_transition` | causal on |
| `--pop-score-bias` | \(\lambda\cdot\log(1+\text{global visits}(c))\) from causal cand feats | causal on |
| `--geo-score-bias` | \(-\lambda\cdot\log(1+\text{haversine km})\) (nearer ranks higher) | `--candidate-context on` |

`--tsb-gate-mode` also modulates `--category-score-bias` when a SURD gate is active (same `mul` / `residual` rules as tsb).

## SURD / Specific-MI bias gates (default **off**)

Let the frozen per-POI SURD table (or its Specific MI) **control** how strongly transition / score biases fire per candidate. No re-prepare.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--surd-bias-gate` | `off` | Gate feature \(g_c\): `unique_p` / `smi_p` / `smi_pc` / `u_mass` / `confidence` |
| `--surd-score-bias` | `0` | Add \(\lambda\cdot g_c\) to ranking scores (SURD contributes directly) |
| `--tsb-gate-mode` | `off` | How \(g_c\) modulates `--transition-score-bias` / `--category-score-bias`: `mul` = term·g; `residual` = term·(1+g) |
| `--surd-attn-gate` | `off` | Same apply modes for `--transition-attn bias` λ·φ (broadcast over history) |

Gate definitions (all × SURD confidence except raw `confidence`):

- **`unique_p`** — routing-prior Unique_P atom
- **`u_mass`** — sum of Unique_P/C/T/L atoms
- **`smi_p` / `smi_pc`** — Specific MI for subset P / PC, normalized by Σ specific
- **`confidence`** — SURD confidence only

`residual` keeps the flat WIN-style term and **amplifies** high-SURD candidates; `mul` **focuses** the bias on high-SURD candidates (can weaken low-SURD ones).

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--causal-counts` / `--no-causal-counts` | **on** | Enable / disable the whole module |
| `--transition-attn {bias,gate,rel_key}` | `bias` | Transition injection mode |
| `--transition-lambda` | `1.0` | Bias scale for attn logits |
| `--transition-score-bias` | `0.0` | Add `λ·log1p(last→cand POI transition)` to ranking scores (0 = off) |
| `--category-score-bias` | `0.0` | Add `λ·log1p(last_cat→cand_poi)` (hypergraph; 0 = off) |
| `--pop-score-bias` | `0.0` | Add `λ·log1p(global POI visits)` (0 = off) |
| `--geo-score-bias` | `0.0` | Subtract `λ·log1p(haversine km)` (needs candidate-context; 0 = off) |
| `--surd-bias-gate` | `off` | SURD/SMI gate feature for score/attn biases |
| `--surd-score-bias` | `0.0` | Add `λ·gate` to scores (needs gate ≠ off) |
| `--tsb-gate-mode` | `off` | `off` / `mul` / `residual` on transition- / category-score-bias |
| `--surd-attn-gate` | `off` | `off` / `mul` / `residual` on transition-attn bias λ |
| `--visit-bins` | `10` | Equal-frequency bins for T/L visit embedding |

Stored in the checkpoint `config` (+ `count_tables` / `visit_bin_edges` when enabled) so `evaluate` / `explain` / `recommend` match training.

## Copy-paste

```bash
# Chronological train + default bias transition attn (causal counts on)
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_causal \
  --epochs 1 --dim 16 --heads 2 --layers 1

# Gate / rel_key modes
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_gate \
  --transition-attn gate --epochs 1 --dim 16 --heads 2 --layers 1
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_rel \
  --transition-attn rel_key --epochs 1 --dim 16 --heads 2 --layers 1

# Score-level last→cand transition prior (Acc@1 hillclimb)
python -m tsmoe train --cache runs/tokyo/cache_abl_tgt_last \
  --out runs/tokyo/hc_str_tsb20 --dim 256 --heads 8 \
  --lambda-bpr 0.25 --transition-score-bias 2.0 \
  --epochs 1 --checkpoint-metric acc1 --device cuda

# LOCKED WIN (TM-34): beats STHGCN Acc@1 + R@1/5/10/20 on known_only
# Acc 0.260274 · R@5 0.441517 · R@10 0.519494 · R@20 0.586934
python -m tsmoe train --cache runs/tokyo/cache_abl_tgt_last \
  --out runs/tokyo/hc_sg_Atime_bpr029_ssb026 --seed 42 --device cuda \
  --router-mode hard_prior --dim 256 --heads 8 \
  --lambda-bpr 0.29 --transition-score-bias 2.0 \
  --surd-bias-gate unique_p --tsb-gate-mode residual --surd-score-bias 0.26 \
  --candidate-context on --candidate-context-time last_slot \
  --epochs 1 --checkpoint-metric acc1

# Ablate the whole module (shuffled minibatches restored)
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_nocounts \
  --no-causal-counts --epochs 1 --dim 16 --heads 2 --layers 1
```

Results still under `results/`. No re-prepare required for this feature.
