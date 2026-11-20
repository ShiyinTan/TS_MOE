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

## Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--causal-counts` / `--no-causal-counts` | **on** | Enable / disable the whole module |
| `--transition-attn {bias,gate,rel_key}` | `bias` | Transition injection mode |
| `--transition-lambda` | `1.0` | Bias scale for attn logits |
| `--transition-score-bias` | `0.0` | Add `λ·log1p(last→cand POI transition)` to ranking scores (0 = off) |
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

# Ablate the whole module (shuffled minibatches restored)
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model_nocounts \
  --no-causal-counts --epochs 1 --dim 16 --heads 2 --layers 1
```

Results still under `results/`. No re-prepare required for this feature.
