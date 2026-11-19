# STHGCN-aligned T/L features

TS-MoE’s time (T) and location (L) **input features** follow the Massive-STEPS next-POI baseline **STHGCN** (Yan et al., SIGIR 2023; official code [alipay/Spatio-Temporal-Hypergraph-Model](https://github.com/alipay/Spatio-Temporal-Hypergraph-Model), Massive-STEPS submodule [w11wo/Spatio-Temporal-Hypergraph-Model](https://github.com/w11wo/Spatio-Temporal-Hypergraph-Model)).

Cache `feature_method`: **`sthgcn_tl_features_v1`**. Changing this (or `t_dim` / `l_dim` / `sthgcn_basis_dim`) changes the prepare fingerprint — **re-prepare** existing caches.

## What STHGCN does

1. **Check-in absolute time** (`CheckinEmbedding`): discrete **hour-of-day** (0–23) and **day-of-week** (0–6) IDs → embeddings (with POI / category / user).
2. **Relative time** (`TimeEncoder`): for edge Δt in **hours** (`delta_ts / (60*60)`):
   \[
   \Phi(\Delta t)_i = \cos(\Delta t \cdot 10^{-\mathrm{linspace}(0,9,d)}),\quad
   \mathbf{t}=W_t\,\Phi(\Delta t)
   \]
   (learnable `basis_freq` / `phase` init to that schedule; optional linear `W_t`).
3. **Relative space** (best configs set `distance_encoder_type: time`): Haversine **Δs in km** is encoded with the **same** `TimeEncoder` (not absolute lat/lon, not bearing).

Paper: *Spatio-Temporal Hypergraph Learning for Next POI Recommendation* (Eq. 3–5). Code: `layer/st_encoder.py`, `model/sthgcn.py`, `conf/best_conf/*.yml`.

## What TS-MoE stores / projects

| Source | Dim | Layout |
| --- | ---: | --- |
| **T** | 48 | hour one-hot (24) ‖ weekday one-hot (7) ‖ Φ(Δt hours, basis=16) ‖ `gap_valid` |
| **L** | 18 | Φ(Δs km, same basis=16) ‖ `coord_valid` ‖ `step_valid` |

- Φ uses the **fixed TimeEncoder init** (phase 0); the T/L **MLP → `dim`** plays the role of \(W_t\) / \(W_s\).
- Window **first step** clears relative Φ / `gap_valid` / `step_valid` (no previous event); hour/weekday and `coord_valid` stay.
- Absolute unit-sphere coords / log-gap / bearing from the old `basic_fixed_features_v1` scheme are **removed** so L matches STHGCN’s relative distance encoding.

P/C hash salts remain `*/basic_fixed_features_v1` (identity encoding unchanged).

## Re-prepare

```bash
# Example: Tokyo (use a new --out or delete the old cache dir)
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_sthgcn
```

Same CLI as before; fingerprint now includes `feature_method`, `t_dim`, `l_dim`, `sthgcn_basis_dim`. Old caches with `basic_fixed_features_v1` will not load under the new check.
