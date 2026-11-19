# Multi-source candidates

Train and evaluate share one candidate construction, fixed at `prepare` time and stored in the cache manifest. The default pool is the **union** of four sources (deduped). Evaluation ranks inside that **full constructed pool**; training may subsample negatives from it.

**Ground truth:** whenever the true next POI is in the training catalog, it is **always** present in the val/test (and train) candidate pool. No resolution path may omit it (`full_catalog` / `fallback_full_catalog` contain every row; `multi_source` force-appends the target). OOV targets (not in the catalog) cannot appear in the pool and are scored as outside-catalog misses. Pool construction verifies this invariant.

## Why (vs a fixed global catalog)

**Yes — preferred for this next-POI setup:** candidates are **dynamic and location-aware** (anchored at the last observed check-in), enriched with popular, user-habitual, and random negatives, and **train/eval stay consistent** because both read the same construction from the cache. A fixed global set scores many geographically implausible venues and can mismatch sampled training negatives.

## Construction rule (default)

For each sample, with anchor = lat/lon of the check-in immediately before the target (`end - 1`):

1. **Neighborhood** — training-catalog POIs within local `radius_km` of the anchor (`--candidate-radius`, default **p90**).
2. **Larger-range popular** — within `popular_radius_km`, take top `--popular-top-k` POIs by train `--popular-by` frequency (`visit` = check-in counts, or `target` = next-POI target counts).
3. **User-frequent** — this user’s top `--user-frequent-top-k` training venues with at least `--user-frequent-min-visits` visits.
4. **Union + dedupe**; **always add the true target** when it is in the catalog.
5. **Random negatives** — up to `--random-negatives` uniform catalog rows not already in the union (deterministic from `--random-negatives-seed` + sample index). Enlarges sparse spatial pools.
6. **Fallback** — if none of (1)–(3) contribute any member (empty union before the target include) → use the **full catalog** for that sample only (random negatives unused; target already included).

`--candidate-radius full` disables spatial / popular / user-frequent / random sources and uses the full training catalog for every sample.

## Prepare knobs (fingerprint)

Changing any of these **requires re-prepare**; train/evaluate do not override them. The same `--out` **overwrites by default** when the fingerprint differs (use distinct `--out` paths to keep parallel caches; `--no-overwrite` refuses replacement).

```bash
python -m tsmoe prepare ... \
  --candidate-radius p90 \              # local: p70|p80|p90|mean|max|full (default p90)
  --popular-radius max \                # popular: max|p90|p80|p70|mean|multiplier (default max)
  --popular-radius-multiplier 2.0 \     # only when --popular-radius multiplier
  --popular-top-k 100 \                 # 0 disables popular source
  --popular-by visit \                  # visit|target
  --user-frequent-top-k 50 \            # 0 disables user-frequent source
  --user-frequent-min-visits 2 \
  --random-negatives 50 \               # 0 disables; default 50
  --random-negatives-seed 0
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--candidate-radius` | `p90` | Local radius from consecutive train check-in distances, or `full` |
| `--popular-radius` | `max` | How to set the larger popular radius (`multiplier` → local × multiplier) |
| `--popular-radius-multiplier` | `2.0` | Scale when `--popular-radius multiplier` |
| `--popular-top-k` | `100` | Top-K popular POIs inside the larger radius (`0` = off) |
| `--popular-by` | `visit` | Rank by train visit count or next-POI target count |
| `--user-frequent-top-k` | `50` | Per-user top venues from **training** history (`0` = off) |
| `--user-frequent-min-visits` | `2` | Min train visits to count as user-frequent |
| `--random-negatives` | `50` | Extra uniform random catalog negatives per sample (`0` = off) |
| `--random-negatives-seed` | `0` | Seed mixed with sample index for deterministic random negatives |

### Which radii work where

| Statistic | `--candidate-radius` | `--popular-radius` | Distance stats key |
| --- | --- | --- | --- |
| `p70` | yes | yes | `p70_km` |
| `p80` | yes | yes | `p80_km` |
| `p90` (local default) | yes | yes | `p90_km` |
| `mean` | yes | yes | `mean_km` |
| `max` (popular default) | yes | yes | `max_km` |
| `multiplier` | no | yes only | `local_radius_km × --popular-radius-multiplier` |
| `full` | yes only | no | disables multi-source (full catalog) |

## Train sampling vs eval

```bash
python -m tsmoe train ... --negatives 63 --max-train-candidates -1
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--negatives` | `63` | Distinct negatives per positive from the constructed pool; `-1` = all other pool members |
| `--max-train-candidates` | `-1` | Optional cap on **total** training candidates including the positive; `-1` = no extra cap |

**Evaluation always uses the full constructed pool** (no `--negatives` / `--max-train-candidates` sampling). Metrics (including **Acc@1**) are still written under `results/…/eval_*/`.

## Pool-size stats at train start

At the start of `train` (and again at the start of standalone `evaluate`), the code builds the same **full** resolved pools used for scoring and prints per-split size statistics:

```text
candidates method=multi_source radius_statistic=p90 radius_km=1.23 popular_radius=max popular_radius_km=5.67 popular_top_k=100 popular_by=visit user_frequent_top_k=50 random_negatives=50 eval_pool=full_constructed ground_truth_in_pool=always
candidates train: n=5855 min=12 max=2494 mean=180.45 median=95.00  (fallback_full_catalog=3 multi_source=5852)
candidates validation: n=808 min=8 max=2494 mean=165.20 median=88.00  (multi_source=808)
candidates test: n=1694 min=10 max=2494 mean=172.11 median=90.00  (fallback_full_catalog=1 multi_source=1693)
```

Counts are **after** union, force-including the in-catalog target, random negatives, and empty-union fallback. A JSON copy is written to `results/<run>/candidate_pool_stats.json`; each `eval_*/metrics.json` also carries that split’s `candidate_pool_stats`.

## Re-prepare

Fingerprint now includes random-negative knobs (and p70/p80 choices). Re-run `prepare` when those change (same `--out` overwrites; keep a distinct `--out` only if you still need the old cache):

```bash
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache_ms \
  --candidate-radius p90 \
  --popular-radius max \
  --popular-top-k 100 \
  --popular-by visit \
  --user-frequent-top-k 50 \
  --user-frequent-min-visits 2 \
  --random-negatives 50 \
  --random-negatives-seed 0

python -m tsmoe train --cache runs/tokyo/cache_ms --out runs/tokyo/model_ms \
  --negatives 63 --max-train-candidates -1 --device cuda
```

Demo:

```bash
python -m tsmoe prepare --data-root data --city demo --out runs/demo/cache
python -m tsmoe train --cache runs/demo/cache --out runs/demo/model \
  --epochs 1 --dim 16 --heads 2 --layers 1
```

## Manifest fields

`manifest.json` records:

- `config.candidate_radius`, `popular_radius`, `popular_radius_multiplier`, `popular_top_k`, `popular_by`, `user_frequent_top_k`, `user_frequent_min_visits`, `random_negatives`, `random_negatives_seed` — fingerprint inputs
- `candidate_construction` — method `multi_source` (or `full_catalog`), both radii, sources (incl. `random_negatives`), `ground_truth_always_in_pool`, distance stats (`p70_km`/`p80_km`/`p90_km`/…), fallback notes, visit/target count vectors
- `poi_coordinates` — finite mean lat/lon per catalog POI (imputed when check-ins lack coords)
- `candidate_construction.coordinate_imputation` — counts per fallback source
- `user_frequent.pt` — map `user_id → [catalog_row, …]` (digested in `files`)

### Coordinate imputation (prepare)

When a catalog POI never has a valid training lat/lon, prepare fills a finite location in this order:

1. **observed** — mean of that POI's check-ins with coords (no imputation)
2. **adjacent_midpoint** — for each visit, midpoint of immediate previous/next trail neighbors that have a location (the neighbor check-in's coords, else that neighbor POI's observed mean); average across visits
3. **trail_mean** — mean of check-in coordinates on trails where the POI appears
4. **global_mean** — mean of all observed catalog POI means

Prepare prints one console line from this audit, e.g. `coordinate_imputation: observed=… adjacent_midpoint=… trail_mean=… global_mean=… (imputed=…/…)`.

Empty-union fallback at **train/eval** is separate: it only expands that sample's candidate pool to the full catalog when neighborhood, popular, and user-frequent all contribute nothing.

`metrics.json` echoes `candidate_construction` for reproducibility.
