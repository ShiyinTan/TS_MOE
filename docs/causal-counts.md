# Causal online count tables + transition cross-attn

Runtime feature (no prepare / fingerprint change). Default **on**.  
Added Acc knobs that *use* these tables (tsb, geo/revisit biases, SURD gates, soft targets) are listed in [`model-settings.md`](model-settings.md); recipes in [`runbook.md`](runbook.md).

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

Needs causal counts on. Locked Tokyo WIN / peak (tsb=2.0, λ_bpr=**0.29**, …): [`runbook.md`](runbook.md) · [`model-settings.md`](model-settings.md).

## Score / SURD gates that read these tables (default off)

Acc@1 additives and SURD gates that **consume** the tables above. Formulas stay here; recipe values live in [`model-settings.md`](model-settings.md). **No re-prepare.** Defaults **0 / off** so Basic trains stay unchanged.

| Flag | Formula / effect | Needs |
| --- | --- | --- |
| `--category-score-bias` | \(\lambda\cdot\log(1+\text{count}(\text{last cat}\to c))\) | causal on |
| `--pop-score-bias` | \(\lambda\cdot\log(1+\text{global visits}(c))\) | causal on |
| `--geo-score-bias` | \(-\lambda\cdot\log(1+\text{haversine km})\) | `--candidate-context on` |
| `--surd-bias-gate` | \(g_c\): `unique_p` / `smi_p` / `smi_pc` / `u_mass` / `confidence` | SURD on |
| `--surd-score-bias` | add \(\lambda\cdot g_c\) to scores | gate ≠ off |
| `--tsb-gate-mode` | `mul` = term·g; `residual` = term·(1+g) on tsb / category-score-bias | gate ≠ off |
| `--surd-attn-gate` | same apply modes on `--transition-attn bias` λ·φ | gate ≠ off |

Gate defs (all × SURD confidence except raw `confidence`): **`unique_p`** Unique_P atom; **`u_mass`** sum Unique_P/C/T/L; **`smi_p` / `smi_pc`** Specific MI for P / PC ÷ Σ specific; **`confidence`** only. `residual` amplifies high-SURD cands; `mul` focuses bias on them.

## Flags (module core)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--causal-counts` / `--no-causal-counts` | **on** | Enable / disable the whole module |
| `--transition-attn {bias,gate,rel_key}` | `bias` | Transition injection mode |
| `--transition-lambda` | `1.0` | Bias scale for attn logits |
| `--transition-score-bias` | `0.0` | Add `λ·log1p(last→cand POI transition)` to ranking scores (0 = off) |
| `--visit-bins` | `10` | Equal-frequency bins for T/L visit embedding |

Score-bias / SURD-gate flags: see table above + [`model-settings.md`](model-settings.md). Stored in checkpoint `config` (+ `count_tables` / `visit_bin_edges`) so `evaluate` / `explain` / `recommend` match training.

## Copy-paste

```bash
# Default: causal on + transition-attn=bias
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_causal \
  --epochs 1 --dim 16 --heads 2 --layers 1

python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_gate \
  --transition-attn gate --epochs 1 --dim 16 --heads 2 --layers 1

# Ablate module (shuffle restored)
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_nocounts \
  --no-causal-counts --epochs 1 --dim 16 --heads 2 --layers 1
```

WIN / peak recipes: [`runbook.md`](runbook.md). More demos: [`../README.md`](../README.md) §3.8. Results under `results/`. No re-prepare.
