# Runtime hotspots (prepare / train / evaluate)

Where wall time tends to go on current `main`: **multi-source candidate pools**, offline SURD folds, and MoE scoring. Complexity first; local CPU micro-benchmarks on demo + Tokyo-shaped synthetics support the ordering. GPU Tokyo wall-clock was **not** measured here.

## Relative order (multi-epoch train)

1. **Train** — usually dominates (epochs × MoE forward/backward over sampled negatives).
2. **Evaluate** — per-epoch validation + final test; ranks the **full constructed pool** (neighborhood ∪ popular ∪ user-frequent ∪ random); much worse with `--candidate-radius full`.
3. **Prepare** — one-shot; SURD + summaries matter on large catalogs but usually less than a full train.

Tokyo smoke scale (`examples/tokyo_smoke.json`): ~5855 / 808 / 1694 train/val/test prefixes, catalog **V ≈ 2494**.

## Top sinks

| Rank | Hotspot | Why | Scale note |
| --- | --- | --- | --- |
| 1 | Eval candidate scoring (`score_pools` / `all_scores`) | Every pool (or catalog) row goes through 4× cross-attn + sparse MoE; eval does **not** use `--negatives` sampling | CPU microbench V=2494, B=16, dim 128: width≈550 ~0.14 s/batch vs full catalog ~0.70 s |
| 2 | Train-step MoE forward/backward | Same scorer on 1+`--negatives` (default 63) each step | Dominates multi-epoch wall time; GPU helps most here |
| 3 | Per-sample multi-source pool resolve (`pools_for_split`) | Up to **two** O(V) haversine scans (local + popular) + dedupe + random negatives; **rebuilt inside every `evaluate`** | Synth N=5855×V=2494: multi-source ~5–6 s/split vs spatial-only ~1–2 s; not GPU-accelerated |
| 4 | Offline SURD (`surd_event_torch` per observed POI) | Python loop × fold tables during prepare | Synth ~2k observed POIs ~0.7–0.9 s/table; × folds |
| 5 | Full-catalog / fallback / wide pools | `full` or `fallback_full_catalog` → `all_scores` over V; large popular radius widens eval W | Watch pool stats + resolution counts at train start |

## Quick wins (small)

- Cache resolved pools and reuse across val epochs (avoid repeated O(N·V) in `evaluate`).
- Compute local + popular neighborhoods from **one** distance pass (two radius thresholds).
- Avoid rebuilding a full `range(V)` list for every random-negative sample.
- Prefer finite `--negatives` / optional `--max-train-candidates` on train; eval still uses the full pool.
- Tune `--eval-batch-size` / `--chunk` for GPU memory vs launch overhead; avoid `full` unless needed.
- Later: spatial index; vectorize SURD if prepare dominates.

## Code map

- `tsmoe/prepare.py` — summaries + SURD fold tables + candidate construction fingerprint  
- `tsmoe/surd.py` — `build_offline_surd` / `surd_event_torch`  
- `tsmoe/candidates.py` — `resolve_candidate_pool` / `pools_for_split` / `popular_indices`  
- `tsmoe/engine.py` — `train` / `evaluate` / `score_pools` / `all_scores`  
- `tsmoe/model.py` — `encode` / `score`

See also [`candidates.md`](candidates.md) for pool semantics and [`results.md`](results.md) for metric paths.
