"""Spatial neighborhood candidate construction shared by prepare, train, and evaluate.

Default radius is the 90th percentile (p90) of consecutive check-in great-circle
distances on the training split. Train and evaluate resolve the same per-sample
pool: catalog POIs within that radius of the previous check-in's coordinates.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

import torch

EARTH_RADIUS_KM = 6371.0088
RADIUS_CHOICES = ("p90", "mean", "max", "full")
DEFAULT_RADIUS = "p90"
# Catalog POI location fallback order when a venue never has a finite check-in coord.
COORD_FALLBACK_ORDER = (
    "observed",           # mean of this POI's training check-ins with coords
    "adjacent_midpoint",  # mean over visits: midpoint of trail-adjacent locations
    "trail_mean",         # mean of all coords on trails where this POI appears
    "global_mean",        # mean of all observed catalog POI means
)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Scalars or tensors; degrees in, km out."""
    to_tensor = isinstance(lat1, torch.Tensor) or isinstance(lat2, torch.Tensor)
    if to_tensor:
        lat1 = torch.as_tensor(lat1, dtype=torch.float64)
        lon1 = torch.as_tensor(lon1, dtype=torch.float64)
        lat2 = torch.as_tensor(lat2, dtype=torch.float64)
        lon2 = torch.as_tensor(lon2, dtype=torch.float64)
        phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
        dphi = phi2 - phi1
        dl = torch.deg2rad(lon2 - lon1)
        hav = torch.sin(dphi / 2).pow(2) + torch.cos(phi1) * torch.cos(phi2) * torch.sin(dl / 2).pow(2)
        return 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(hav.clamp(0., 1.)))
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lon2 - lon1)
    hav = (math.sin(dphi / 2) ** 2
           + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1., max(0., hav))))


def consecutive_visit_distances(groups) -> list[float]:
    """Training consecutive event pairs with valid coordinates → km distances."""
    distances = []
    for events in groups.values():
        for previous, current in zip(events, events[1:]):
            if previous.coord is None or current.coord is None:
                continue
            distances.append(haversine_km(*previous.coord, *current.coord))
    return distances


def summarize_distances(distances: Sequence[float]) -> dict:
    """Mean / max / p90 (and helpers) over consecutive visit distances in km."""
    if not distances:
        return dict(count=0, mean_km=None, max_km=None, min_km=None,
                    median_km=None, p90_km=None)
    values = sorted(float(d) for d in distances)
    n = len(values)

    def percentile(p):
        if n == 1:
            return values[0]
        pos = (n - 1) * (p / 100.0)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return values[lo]
        return values[lo] * (hi - pos) + values[hi] * (pos - lo)

    return dict(
        count=n,
        mean_km=sum(values) / n,
        max_km=values[-1],
        min_km=values[0],
        median_km=percentile(50),
        p90_km=percentile(90),
    )


def pick_radius_km(stats: dict, method: str) -> float | None:
    """Map --candidate-radius choice to km, or None for full-catalog mode."""
    if method not in RADIUS_CHOICES:
        raise ValueError(f"candidate-radius must be one of {RADIUS_CHOICES}, got {method!r}")
    if method == "full":
        return None
    if not stats.get("count"):
        raise ValueError(
            "no consecutive check-in pairs with coordinates; cannot derive a spatial "
            "radius (use --candidate-radius full or fix coordinates)")
    key = {"p90": "p90_km", "mean": "mean_km", "max": "max_km"}[method]
    value = stats[key]
    if value is None or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid distance statistic {key}={value!r}")
    return float(value)


def _average_latlon(points: Sequence[tuple[float, float]]) -> list[float]:
    n = len(points)
    if not n:
        raise ValueError("cannot average an empty coordinate list")
    return [sum(lat for lat, _ in points) / n, sum(lon for _, lon in points) / n]


def _neighbor_location(event, observed: dict[int, list[float]]) -> tuple[float, float] | None:
    """Prefer the check-in's own coords; else that POI's observed catalog mean."""
    if event.coord is not None:
        lat, lon = event.coord
        if math.isfinite(lat) and math.isfinite(lon):
            return float(lat), float(lon)
    mean = observed.get(event.poi)
    if mean is not None:
        return float(mean[0]), float(mean[1])
    return None


def catalog_coordinates(groups, pois: Sequence[int]) -> tuple[list[list[float]], dict]:
    """Finite mean lat/lon per catalog POI, with trail-based imputation when needed.

    Fallback order (see ``COORD_FALLBACK_ORDER``):
      1. ``observed`` — mean of this POI's training check-ins with valid coords
      2. ``adjacent_midpoint`` — for each visit, midpoint of immediate trail
         neighbors that have a location (check-in coord, else that neighbor
         POI's observed mean); average those midpoints across visits
      3. ``trail_mean`` — mean of check-in coords on trails where the POI appears
      4. ``global_mean`` — mean of all observed catalog POI means

    Returns ``(coords, audit)`` where ``coords[i]`` aligns with ``pois[i]`` and
    every entry is a finite ``[lat, lon]``. Raises if no training coordinates
    exist to impute from.
    """
    sums = {poi: [0.0, 0.0, 0] for poi in pois}
    for events in groups.values():
        for event in events:
            if event.poi not in sums or event.coord is None:
                continue
            lat, lon = event.coord
            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            sums[event.poi][0] += lat
            sums[event.poi][1] += lon
            sums[event.poi][2] += 1

    observed: dict[int, list[float]] = {}
    for poi in pois:
        lat_sum, lon_sum, count = sums[poi]
        if count:
            observed[poi] = [lat_sum / count, lon_sum / count]

    missing = [poi for poi in pois if poi not in observed]
    sources = {poi: "observed" for poi in observed}

    if missing:
        adjacent_hits: dict[int, list[tuple[float, float]]] = {poi: [] for poi in missing}
        trail_hits: dict[int, list[tuple[float, float]]] = {poi: [] for poi in missing}
        for events in groups.values():
            trail_coords = [
                (float(e.coord[0]), float(e.coord[1]))
                for e in events
                if e.coord is not None and math.isfinite(e.coord[0]) and math.isfinite(e.coord[1])
            ]
            trail_mean = _average_latlon(trail_coords) if trail_coords else None
            seen_on_trail = set()
            for index, event in enumerate(events):
                if event.poi not in adjacent_hits:
                    continue
                neighbors = []
                if index > 0:
                    left = _neighbor_location(events[index - 1], observed)
                    if left is not None:
                        neighbors.append(left)
                if index + 1 < len(events):
                    right = _neighbor_location(events[index + 1], observed)
                    if right is not None:
                        neighbors.append(right)
                if neighbors:
                    adjacent_hits[event.poi].append(
                        (sum(p[0] for p in neighbors) / len(neighbors),
                         sum(p[1] for p in neighbors) / len(neighbors)))
                if trail_mean is not None and event.poi not in seen_on_trail:
                    trail_hits[event.poi].append((trail_mean[0], trail_mean[1]))
                    seen_on_trail.add(event.poi)

        still = []
        for poi in missing:
            samples = adjacent_hits[poi]
            if samples:
                observed[poi] = _average_latlon(samples)
                sources[poi] = "adjacent_midpoint"
            else:
                still.append(poi)

        missing = still
        still = []
        for poi in missing:
            samples = trail_hits[poi]
            if samples:
                observed[poi] = _average_latlon(samples)
                sources[poi] = "trail_mean"
            else:
                still.append(poi)
        missing = still

    if missing:
        # Global prior uses only directly observed POI means (not already-imputed).
        observed_means = [
            (sums[poi][0] / sums[poi][2], sums[poi][1] / sums[poi][2])
            for poi in pois if sums[poi][2]
        ]
        if not observed_means:
            raise ValueError(
                "no training check-ins with valid coordinates; cannot build or impute "
                "catalog poi_coordinates")
        global_mean = _average_latlon(observed_means)
        for poi in missing:
            observed[poi] = list(global_mean)
            sources[poi] = "global_mean"

    coords = []
    for poi in pois:
        lat, lon = observed[poi]
        if not (math.isfinite(lat) and math.isfinite(lon)):
            raise ValueError(f"non-finite catalog coordinate for poi={poi}: {[lat, lon]}")
        coords.append([float(lat), float(lon)])

    counts = {name: 0 for name in COORD_FALLBACK_ORDER}
    for poi in pois:
        counts[sources[poi]] += 1
    audit = dict(
        fallback_order=list(COORD_FALLBACK_ORDER),
        counts=counts,
        imputed=sum(counts[name] for name in COORD_FALLBACK_ORDER if name != "observed"),
        catalog_size=len(pois),
    )
    return coords, audit


def neighborhood_indices(anchor_lat: float, anchor_lon: float,
                         catalog_coords: torch.Tensor, radius_km: float) -> torch.Tensor:
    """Catalog row indices within radius_km of the anchor (degrees)."""
    if not (math.isfinite(anchor_lat) and math.isfinite(anchor_lon)):
        return torch.empty(0, dtype=torch.long)
    coords = torch.as_tensor(catalog_coords, dtype=torch.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("catalog_coords must be [V, 2]")
    valid = torch.isfinite(coords).all(dim=1)
    if not valid.any():
        return torch.empty(0, dtype=torch.long)
    dist = torch.full((coords.shape[0],), float("inf"), dtype=torch.float64)
    dist[valid] = haversine_km(anchor_lat, anchor_lon, coords[valid, 0], coords[valid, 1])
    return torch.where(dist <= radius_km)[0]


def resolve_candidate_pool(anchor_lat: float, anchor_lon: float, target_row: int,
                           catalog_coords: torch.Tensor, radius_km: float | None,
                           catalog_size: int) -> tuple[torch.Tensor, str]:
    """Resolve the train/eval candidate set for one sample.

    Returns (row_indices, resolution) where resolution is:
      - ``full_catalog`` when radius is disabled
      - ``spatial`` when the neighborhood is nonempty
      - ``fallback_full_catalog`` when the anchor is invalid or the neighborhood
        is empty before force-including the target

    Known in-catalog targets are always included in a spatial pool.
    """
    full = torch.arange(catalog_size, dtype=torch.long)
    if radius_km is None:
        return full, "full_catalog"
    if not (math.isfinite(anchor_lat) and math.isfinite(anchor_lon)):
        return full, "fallback_full_catalog"
    pool = neighborhood_indices(anchor_lat, anchor_lon, catalog_coords, radius_km)
    if pool.numel() == 0:
        return full, "fallback_full_catalog"
    if target_row >= 0 and not (pool == target_row).any():
        pool = torch.cat([pool, torch.tensor([target_row], dtype=torch.long)])
    return pool, "spatial"


def pools_for_split(data, catalog_coords, radius_km: float | None, catalog_size: int):
    """Build per-sample candidate pools and resolution labels for a split tensor dict.

    Supervised splits store the true next POI at ``end``. Recommend / predict
    mode uses ``end`` as an exclusive past-the-last-event index (no target row);
    the anchor is still ``end - 1`` and ``target_row`` is ``-1``.
    """
    ends = data["end"]
    lats, lons = data["lat"], data["lon"]
    n_events = len(data["poi_token"])
    pools, resolutions = [], []
    for i in range(len(ends)):
        end = int(ends[i])
        prev = end - 1
        if not 0 <= prev < n_events:
            raise IndexError(f"sample {i}: missing previous check-in for candidate anchor")
        if end < n_events:
            target_row = int(data["poi_token"][end]) - 2
        else:
            target_row = -1
        pool, how = resolve_candidate_pool(
            float(lats[prev]), float(lons[prev]), target_row,
            catalog_coords, radius_km, catalog_size)
        pools.append(pool)
        resolutions.append(how)
    return pools, resolutions


def summarize_pool_sizes(pools: Sequence[torch.Tensor],
                         resolutions: Sequence[str] | None = None) -> dict:
    """Per-sample candidate-pool size stats after resolve (neighborhood + target + fallback).

    ``len(pool)`` matches the set used for training negatives and eval ranking.
    """
    sizes = [int(pool.numel()) for pool in pools]
    n = len(sizes)
    if not n:
        out = dict(n=0, min=None, max=None, mean=None, median=None)
    else:
        ordered = sorted(sizes)
        mid = n // 2
        median = float(ordered[mid]) if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
        out = dict(
            n=n,
            min=int(ordered[0]),
            max=int(ordered[-1]),
            mean=float(sum(sizes) / n),
            median=median,
        )
    if resolutions is not None:
        if len(resolutions) != n:
            raise ValueError("resolutions must align with pools")
        counts: dict[str, int] = {}
        for how in resolutions:
            counts[how] = counts.get(how, 0) + 1
        out["resolutions"] = counts
    return out


def format_pool_size_line(split: str, stats: dict) -> str:
    """One console line: ``candidates train: n=… min=… max=… mean=… median=…``."""
    def fmt(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(int(value))

    line = (f"candidates {split}: n={stats.get('n', 0)} "
            f"min={fmt(stats.get('min'))} max={fmt(stats.get('max'))} "
            f"mean={fmt(stats.get('mean'))} median={fmt(stats.get('median'))}")
    resolutions = stats.get("resolutions")
    if resolutions:
        parts = " ".join(f"{key}={value}" for key, value in sorted(resolutions.items()))
        line = f"{line}  ({parts})"
    return line


def sample_candidates(targets: torch.Tensor, catalog_size: int, negatives: int,
                      rng: random.Random, pools: Sequence[torch.Tensor] | None = None):
    """Sample training rows: positive at column 0, then negatives from each pool.

    Returns ``(rows, labels, candidate_mask)``. ``pools is None`` preserves the
    historical uniform negative sampler over the full catalog. With spatial
    pools, negatives are drawn only from that sample's resolved neighborhood
    (``negatives=-1`` ⇒ all other pool members).
    """
    if pools is None:
        if negatives < 0 or negatives >= catalog_size - 1:
            rows = torch.arange(catalog_size).expand(len(targets), -1).contiguous()
            return rows, targets.clone(), torch.ones_like(rows, dtype=torch.bool)
        row_lists = []
        for target in targets.tolist():
            chosen = rng.sample(range(catalog_size - 1), negatives)
            row_lists.append([target, *(n + (n >= target) for n in chosen)])
        rows = torch.tensor(row_lists, dtype=torch.long)
        return rows, torch.zeros(len(targets), dtype=torch.long), torch.ones_like(rows, dtype=torch.bool)

    if len(pools) != len(targets):
        raise ValueError("pools must align with targets")
    row_lists = []
    for target, pool in zip(targets.tolist(), pools):
        members = list(dict.fromkeys(int(x) for x in pool.tolist()))
        if target >= 0 and target not in members:
            members.append(target)
        if target < 0:
            raise ValueError("training target outside catalog")
        others = [row for row in members if row != target]
        if negatives < 0 or negatives >= len(others):
            chosen = others
        else:
            chosen = rng.sample(others, negatives)
        row_lists.append([target, *chosen])
    width = max(len(row) for row in row_lists)
    padded = torch.full((len(row_lists), width), -1, dtype=torch.long)
    mask = torch.zeros(len(row_lists), width, dtype=torch.bool)
    for i, row in enumerate(row_lists):
        padded[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, :len(row)] = True
    return padded, torch.zeros(len(targets), dtype=torch.long), mask


def candidate_construction_record(method: str, stats: dict, radius_km: float | None,
                                  coordinate_imputation: dict | None = None) -> dict:
    """Manifest block documenting how train/eval candidates are built."""
    spatial = method != "full"
    return dict(
        method="spatial_radius" if spatial else "full_catalog",
        radius_statistic=method,
        radius_km=radius_km,
        default_statistic=DEFAULT_RADIUS,
        distance_stats=stats,
        anchor="previous_checkin_coordinates",
        catalog_coordinates=(
            "finite mean lat/lon per catalog POI; missing check-in coords are imputed "
            "via trail-adjacent midpoints, then trail mean, then global observed mean"
        ),
        coordinate_imputation=coordinate_imputation or dict(
            fallback_order=list(COORD_FALLBACK_ORDER)),
        include_target=True,
        empty_neighborhood_fallback="full_catalog",
        train_eval_shared=True,
        notes=(
            "Candidates are training-catalog POIs within radius_km of the last "
            "observed check-in's coordinates. Train and evaluate use this same "
            "construction. If the anchor lacks coordinates or no catalog POI lies "
            "inside the radius, that sample falls back to the full catalog. The "
            "true target is always added when it is in the catalog. Catalog "
            "coordinates prefer observed check-in means; otherwise prepare imputes "
            "from trail-adjacent locations so poi_coordinates stay JSON-finite."
        ),
    )
