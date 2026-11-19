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
MISSING_COORD = float("nan")


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


def catalog_coordinates(groups, pois: Sequence[int]) -> list[list[float]]:
    """Mean lat/lon per catalog POI from training check-ins; NaN if never observed."""
    sums = {poi: [0.0, 0.0, 0] for poi in pois}
    for events in groups.values():
        for event in events:
            if event.poi not in sums or event.coord is None:
                continue
            lat, lon = event.coord
            sums[event.poi][0] += lat
            sums[event.poi][1] += lon
            sums[event.poi][2] += 1
    coords = []
    for poi in pois:
        lat_sum, lon_sum, count = sums[poi]
        if count:
            coords.append([lat_sum / count, lon_sum / count])
        else:
            coords.append([MISSING_COORD, MISSING_COORD])
    return coords


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


def candidate_construction_record(method: str, stats: dict, radius_km: float | None) -> dict:
    """Manifest block documenting how train/eval candidates are built."""
    spatial = method != "full"
    return dict(
        method="spatial_radius" if spatial else "full_catalog",
        radius_statistic=method,
        radius_km=radius_km,
        default_statistic=DEFAULT_RADIUS,
        distance_stats=stats,
        anchor="previous_checkin_coordinates",
        catalog_coordinates="mean lat/lon of training check-ins with valid coords; NaN if none",
        include_target=True,
        empty_neighborhood_fallback="full_catalog",
        train_eval_shared=True,
        notes=(
            "Candidates are training-catalog POIs within radius_km of the last "
            "observed check-in's coordinates. Train and evaluate use this same "
            "construction. If the anchor lacks coordinates or no catalog POI lies "
            "inside the radius, that sample falls back to the full catalog. The "
            "true target is always added when it is in the catalog."
        ),
    )
