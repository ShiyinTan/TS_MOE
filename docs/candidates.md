# Spatial neighborhood candidates

Train and evaluate share one candidate construction, fixed at `prepare` time and stored in the cache manifest.

## Why (vs a fixed global catalog)

**Yes — preferred for this next-POI setup:** candidates are **dynamic and location-aware** (anchored at the last observed check-in), and **train/eval stay consistent** because both read the same `radius_km` / pool rule from the cache. A fixed global set scores many geographically implausible venues and can mismatch sampled training negatives.

## Knob

```bash
python -m tsmoe prepare ... --candidate-radius p90   # default
python -m tsmoe prepare ... --candidate-radius mean
python -m tsmoe prepare ... --candidate-radius max
python -m tsmoe prepare ... --candidate-radius full  # legacy full training catalog
```

| Value | Radius |
| --- | --- |
| `p90` (default) | 90th percentile of consecutive train check-in great-circle distances (km) |
| `mean` | mean of those distances |
| `max` | maximum |
| `full` | disable spatial filtering; candidates = entire training POI catalog |

Changing the statistic **requires a new cache** (`prepare --out …`); train/evaluate do not override it.

## Per-sample rule

1. **Anchor** = latitude/longitude of the check-in immediately before the target (`end - 1`).
2. **Pool** = training-catalog POIs whose mean training coordinates lie within `radius_km` of the anchor.
3. **Target** = always added when it is in the catalog (even if outside the radius).
4. **Fallback** = if the anchor has no valid coordinates **or** the neighborhood is empty before force-including the target → use the **full catalog** for that sample only.

Training negatives (`train --negatives`) are drawn from this pool. Evaluation ranks the true next POI **inside the same pool**. Metrics (including **Acc@1**) are still written under `results/…/eval_*/`.

## Manifest fields

`manifest.json` records:

- `config.candidate_radius` — fingerprint input
- `candidate_construction` — method, `radius_km`, distance stats, fallback notes
- `poi_coordinates` — finite mean lat/lon per catalog POI (imputed when check-ins lack coords)
- `candidate_construction.coordinate_imputation` — counts per fallback source

### Coordinate imputation (prepare)

When a catalog POI never has a valid training lat/lon, prepare fills a finite location in this order:

1. **observed** — mean of that POI's check-ins with coords (no imputation)
2. **adjacent_midpoint** — for each visit, midpoint of immediate previous/next trail neighbors that have a location (the neighbor check-in's coords, else that neighbor POI's observed mean); average across visits
3. **trail_mean** — mean of check-in coordinates on trails where the POI appears
4. **global_mean** — mean of all observed catalog POI means

Prepare prints one console line from this audit, e.g. `coordinate_imputation: observed=… adjacent_midpoint=… trail_mean=… global_mean=… (imputed=…/…)`.

Empty-neighborhood fallback at **train/eval** is separate: it only expands that sample's candidate pool to the full catalog when the anchor is invalid or no catalog POI lies inside `radius_km`.

`metrics.json` echoes `candidate_construction` for reproducibility.
