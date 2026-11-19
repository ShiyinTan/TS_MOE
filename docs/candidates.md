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
- `poi_coordinates` — mean lat/lon per catalog POI (NaN if never observed with coords)

`metrics.json` echoes `candidate_construction` for reproducibility.
