# Candidate context / session / user-pref / expert cross-attn

Runtime overlays on top of the default Acc@1 path. **All default `off`** — leaving them unset (or `off`) keeps today’s behavior. New signals enter **base / shared query / expert attn**, never `hard_prior` router logits.

See also: [`causal-counts.md`](causal-counts.md) · Acc recipes [`runbook.md`](runbook.md) · results [`experiments.md`](experiments.md).

## Defaults matrix

| Flag | Default | Prepare? | Enters `hard_prior`? |
| --- | --- | ---: | ---: |
| `--candidate-context {off,on}` | **`off`** | **No** | **No** |
| `--candidate-context-time {off,last_slot,causal_slot}` | **`off`** | **No**† | **No** |
| `--session-summary {off,on}` | **`off`** | **No** | **No** |
| `--user-pref {off,on}` | **`off`** | **No**† | **No** |
| `--user-id-emb {off,on}` | **`off`** | **No** | **No** |
| `--transition-pmi {off,on}` | **`off`** | **No**† | **No** |
| `--expert-cross-attn {off,hybrid,full}` | **`off`** | **No** | **No** |
| `--expert-cross-attn-sparse {off,on}` | **`on`** (only when expert-cross-attn ≠ off) | **No** | **No** |

† `causal_slot` / `user-pref` / `transition-pmi` need **causal counts on** (extend online tables at runtime — **no re-prepare**). Text / venue embeddings: **deferred** (not implemented).

## A — `--candidate-context`

When `on`:

1. Candidate **category** embedding (same `C` table as history; POI→cat = mode category from train events at runtime).
2. **Δgeo** vs last history tip: haversine km + local east/north km, each `log1p` / signed-`log1p`.
3. Fuse `[e_P; e_C; geo; time]` → D **before** shared cross-attn / base.

`--candidate-context-time`:

| Value | Effect |
| --- | --- |
| `off` | no time side |
| `last_slot` | last hour (24) + weekday (7) one-hot → MLP |
| `causal_slot` | last_slot + causal same-hour POI visit features (needs causal tables) |

## B — `--session-summary`

Window category soft-hist, traj-length bin, activity-center→last distance → MLP added into the shared `full` vector (base / hybrid context). Not candidate-specific; not router.

## C — `--user-pref`

Causal **user→cat** and **user→hour** preference vectors → MLP → `full`. Requires `--causal-counts`. **Forbidden** in `hard_prior` logits (enforced by architecture).

## C2 — `--user-id-emb`

Learned **`Embedding(train user_id)`** → Linear → `full` (base/context). Vocab = distinct `user_id` on the train split of the cache; val/test users missing from train map to **UNK**. **No re-prepare.** Distinct from `--user-pref` (counts). **Forbidden** in `hard_prior`. Suggested run name: `abl_user_id` / hillclimb `hc_user_id_*`.

## D — `--transition-pmi`

When `on`, P/C transition cross-attn uses **PMI** instead of raw counts (no extra `log1p` inside the attn module). Requires causal counts. Category/venue **text embeddings**: deferred.

## E — `--expert-cross-attn`

| Value | Meaning |
| --- | --- |
| `off` | shared cross-attn only (today) |
| `hybrid` (**E1**) | keep shared attn (+ transition on shared P/C) for base; MoE residual uses typed expert-local MHA |
| `full` (**E2**) | MoE path uses expert-local attn only (no shared transition on that path) |

`--expert-cross-attn-sparse on` (**E3**): only compute expert-local attn for sources that appear in router top-k experts.

Transition stays on **shared P/C** for hybrid (plan §7).

## Copy-paste overlays

Default train (unchanged Acc@1 path):

```bash
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/abl_ctx_off --device cuda
```

Add **one** overlay flag at a time (same `--cache` / seed). Locked Acc recipes: [`runbook.md`](runbook.md).

| setting | add flags |
| --- | --- |
| A cat+geo | `--candidate-context on` |
| A + time | `--candidate-context on --candidate-context-time last_slot` |
| B session | `--session-summary on` |
| C user-pref | `--user-pref on` |
| D PMI | `--transition-pmi on` |
| E1 hybrid | `--expert-cross-attn hybrid` |
| user id emb | `--user-id-emb on` |
