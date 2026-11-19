# Router modes (`--router-mode`)

Expert routing over the 15 Unique/Synergy experts can be driven by the **candidate POI’s SURD row**, optionally with a small sequence×candidate learned term. The mode is set at `train` time, stored in the checkpoint `config`, and reused by `evaluate` / `explain` / `recommend` so eval matches train.

## Flag and defaults

| CLI | Default | Meaning |
| --- | --- | --- |
| `--router-mode` | `hard_prior` | `hard_prior` \| `surd_mlp` \| `hybrid` |
| `--lambda-prior` | `0.1` | Scale on explicit SURD bias in **hybrid** (`λ_prior · surd_bias`) |
| `--hybrid-lambda` | `0.1` | ε on the learned term in **hybrid** (`ε · learned`) |
| `--temperature` | `1.0` | Softmax temperature for all modes |
| `--no-surd` | off | Zeros confidence → turns off SURD guidance (hard_prior → uniform π) |

Legacy checkpoints without `router_mode` load as `hybrid` with `hybrid_lambda=1.0` (previous `learned + λ_prior · bias` behavior).

## What each mode uses

Offline SURD stores a 26-D `routing_prior` per POI (4 Unique + 11 Redundancy + 11 Synergy). Online:

- **U/S mass** (15-D, expert order): `mass = concat(prior[..., :4], prior[..., 15:])`
- **Redundancy** `ρ = prior[..., 4:15].sum(-1)` and **α = mass.sum(-1)** feed the residual gate (and SURD MLP features), not the 15 expert logits directly
- **Confidence** `ω` scales SURD inputs / bias; low-support rows contribute little

### `hard_prior` (default)

```text
surd_bias = ω · log1p(15 · mass)     # [B, M, 15], no sequence
logits    = surd_bias
π         = softmax(logits / T)
```

Strong interpretability: different POIs emphasize different U/S atoms; history does not enter the router (Base / Experts still see the sequence under that routing). Gate uses SURD-only features.

### `surd_mlp`

```text
surd_feats = [prior_mlp(ω · prior26), ω·ρ, ω·α]   # 66-D, no sequence
logits     = MLP(surd_feats)                        # → 15
π          = softmax(logits / T)
```

Learned map from candidate SURD only; still no sequence/context in the router.

### `hybrid`

```text
surd_bias = ω · log1p(15 · mass)
learned   = RouterMLP(context, candidate, SURD feats)   # sequence × candidate
logits    = λ_prior · surd_bias + hybrid_lambda · learned
π         = softmax(logits / T)
```

Defaults `λ_prior=0.1`, `hybrid_lambda=0.1`. For a SURD-primary hybrid with a small learned term, use e.g. `--lambda-prior 1 --hybrid-lambda 0.1`.

## Results

Metrics (including **Acc@1**) go under `results/<run>/eval_*` as usual; checkpoints stay in `runs/`. Use a distinct `--out` per mode so result paths do not collide.

See README §3.5 for copy-paste demo and Tokyo commands.
