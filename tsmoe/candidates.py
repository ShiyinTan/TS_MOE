"""Multi-source candidate construction shared by prepare, train, and evaluate.

Default pools union four parts (deduped; true target always included when in
catalog):

1. **Spatial neighborhood** — catalog POIs within ``radius_km`` of the previous
   check-in's coordinates (``--candidate-radius`` p70/p80/p90/mean/max).
2. **Larger-range popular POIs** — within a larger radius, top-K by train
   visit or target frequency.
3. **User-frequent POIs** — venues this user often visits in training history.
4. **Random negatives** — uniform catalog rows not already in the union
   (``--random-negatives``, default 50; ``0`` disables).

``--candidate-radius full`` keeps the legacy full-catalog pool.
``--candidate-radius random`` builds a **random-only** pool: ground-truth (when
in catalog) plus up to ``--random-negatives`` uniform catalog rows (same sampler
/ seed as multi-source extras; neighborhood / popular / user-frequent unused).
Training may sample negatives / cap candidates; evaluation ranks inside the
**full** constructed pool (no train-time sampling). Val/test pools always
contain the in-catalog ground-truth target (no path may omit it).
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Mapping, Sequence

import torch

EARTH_RADIUS_KM = 6371.0088
RADIUS_CHOICES = ("p70", "p80", "p90", "mean", "max", "full", "random")
DEFAULT_RADIUS = "p90"
DISTANCE_STAT_KEYS = {
    "p70": "p70_km",
    "p80": "p80_km",
    "p90": "p90_km",
    "mean": "mean_km",
    "max": "max_km",
}
POPULAR_RADIUS_CHOICES = ("max", "p90", "p80", "p70", "mean", "multiplier")
DEFAULT_POPULAR_RADIUS = "max"
DEFAULT_POPULAR_RADIUS_MULTIPLIER = 2.0
DEFAULT_POPULAR_TOP_K = 100
POPULAR_BY_CHOICES = ("visit", "target")
DEFAULT_POPULAR_BY = "visit"
DEFAULT_USER_FREQUENT_TOP_K = 50
DEFAULT_USER_FREQUENT_MIN_VISITS = 2
DEFAULT_RANDOM_NEGATIVES = 50
DEFAULT_RANDOM_NEGATIVES_SEED = 0
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
    """Mean / max / p70 / p80 / p90 (and helpers) over consecutive visit distances in km."""
    if not distances:
        return dict(count=0, mean_km=None, max_km=None, min_km=None,
                    median_km=None, p70_km=None, p80_km=None, p90_km=None)
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
        p70_km=percentile(70),
        p80_km=percentile(80),
        p90_km=percentile(90),
    )


def pick_radius_km(stats: dict, method: str) -> float | None:
    """Map --candidate-radius choice to km, or None for full / random modes."""
    if method not in RADIUS_CHOICES:
        raise ValueError(f"candidate-radius must be one of {RADIUS_CHOICES}, got {method!r}")
    if method in ("full", "random"):
        # full = entire catalog; random = random_pool (no spatial radius).
        return None
    if not stats.get("count"):
        raise ValueError(
            "no consecutive check-in pairs with coordinates; cannot derive a spatial "
            "radius (use --candidate-radius full|random or fix coordinates)")
    key = DISTANCE_STAT_KEYS[method]
    value = stats[key]
    if value is None or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid distance statistic {key}={value!r}")
    return float(value)


def pick_popular_radius_km(stats: dict, local_radius_km: float | None, method: str,
                           multiplier: float) -> float | None:
    """Larger radius for popular POIs, or None when local radius is disabled."""
    if local_radius_km is None:
        return None
    if method not in POPULAR_RADIUS_CHOICES:
        raise ValueError(
            f"popular-radius must be one of {POPULAR_RADIUS_CHOICES}, got {method!r}")
    if method == "multiplier":
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(f"popular-radius-multiplier must be positive finite, got {multiplier!r}")
        return float(local_radius_km) * float(multiplier)
    key = DISTANCE_STAT_KEYS[method]
    if not stats.get("count"):
        raise ValueError(
            "no consecutive check-in pairs with coordinates; cannot derive a popular "
            "radius (use --candidate-radius full or fix coordinates)")
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


def catalog_visit_counts(groups, pois: Sequence[int]) -> list[int]:
    """Training check-in visit counts aligned with catalog ``pois``."""
    counts = Counter()
    for events in groups.values():
        for event in events:
            if event.poi >= 0:
                counts[event.poi] += 1
    return [int(counts.get(poi, 0)) for poi in pois]


def catalog_target_counts(groups, pois: Sequence[int]) -> list[int]:
    """Training next-POI target counts (consecutive trail steps) aligned with ``pois``."""
    counts = Counter()
    for events in groups.values():
        for _previous, current in zip(events, events[1:]):
            if current.poi >= 0:
                counts[current.poi] += 1
    return [int(counts.get(poi, 0)) for poi in pois]


def build_user_frequent(groups, pois: Sequence[int], top_k: int,
                        min_visits: int) -> dict[str, list[int]]:
    """Per-user top catalog rows by training visit count (min_visits filter)."""
    if top_k <= 0:
        return {}
    if min_visits < 1:
        raise ValueError(f"user-frequent-min-visits must be >= 1, got {min_visits!r}")
    poi_to_row = {poi: i for i, poi in enumerate(pois)}
    by_user: dict[str, Counter] = defaultdict(Counter)
    for (user, _trail), events in groups.items():
        for event in events:
            if event.poi in poi_to_row:
                by_user[str(user)][event.poi] += 1
    out: dict[str, list[int]] = {}
    for user, counter in by_user.items():
        ranked = [poi for poi, count in counter.most_common() if count >= min_visits]
        rows = [poi_to_row[poi] for poi in ranked[:top_k]]
        if rows:
            out[user] = rows
    return out


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


def popular_indices(anchor_lat: float, anchor_lon: float, catalog_coords: torch.Tensor,
                    popular_radius_km: float, popularity: torch.Tensor,
                    top_k: int) -> torch.Tensor:
    """Top-K popular catalog rows within the larger radius (ties → lower row index)."""
    if top_k <= 0:
        return torch.empty(0, dtype=torch.long)
    near = neighborhood_indices(anchor_lat, anchor_lon, catalog_coords, popular_radius_km)
    if near.numel() == 0:
        return near
    scores = torch.as_tensor(popularity, dtype=torch.float64)[near]
    # argsort ascending then reverse → descending; stable keeps lower index first on ties.
    order = torch.argsort(scores, descending=True, stable=True)
    return near[order[: min(int(top_k), int(near.numel()))]]


def _dedupe_rows(*parts: Sequence[int]) -> list[int]:
    seen = set()
    out = []
    for part in parts:
        for row in part:
            value = int(row)
            if value < 0 or value in seen:
                continue
            seen.add(value)
            out.append(value)
    return out


def sample_random_catalog_rows(catalog_size: int, exclude: set[int], count: int,
                               rng: random.Random) -> list[int]:
    """Uniform sample of catalog rows not in ``exclude`` (up to ``count``)."""
    if count <= 0 or catalog_size <= 0:
        return []
    available = [i for i in range(catalog_size) if i not in exclude]
    if not available:
        return []
    return rng.sample(available, min(int(count), len(available)))


def _rng_for_sample(seed: int, sample_index: int) -> random.Random:
    """Deterministic per-sample RNG from prepare-time seed + sample index."""
    # Mix with a fixed odd constant so nearby indices do not share streams.
    mixed = (int(seed) ^ ((int(sample_index) + 1) * 0x9E3779B9)) & 0xFFFFFFFF
    return random.Random(mixed)


def resolve_candidate_pool(anchor_lat: float, anchor_lon: float, target_row: int,
                           catalog_coords: torch.Tensor, radius_km: float | None,
                           catalog_size: int,
                           *,
                           popular_radius_km: float | None = None,
                           popularity: torch.Tensor | None = None,
                           popular_top_k: int = 0,
                           user_frequent_rows: Sequence[int] | None = None,
                           random_negatives: int = 0,
                           random_negatives_seed: int = 0,
                           sample_index: int = 0,
                           random_only: bool = False) -> tuple[torch.Tensor, str]:
    """Resolve the train/eval candidate set for one sample.

    Returns (row_indices, resolution) where resolution is:
      - ``full_catalog`` when radius is disabled (``--candidate-radius full``)
      - ``random_pool`` when ``random_only`` (``--candidate-radius random``):
        ground-truth plus up to ``random_negatives`` uniform catalog rows
      - ``multi_source`` when the union of neighborhood / popular / user-frequent
        is nonempty (random negatives may further enlarge the pool)
      - ``fallback_full_catalog`` when no primary source contributes and the pool
        would otherwise be empty aside from a possible lone target

    Known in-catalog targets are **always** included in a constructed pool
    (including val/test). No resolution path may omit an in-catalog ground-truth
    target: ``full_catalog`` / ``fallback_full_catalog`` contain every row;
    ``multi_source`` / ``random_pool`` force-append ``target_row`` when
    ``target_row >= 0``.
    """
    full = torch.arange(catalog_size, dtype=torch.long)
    if random_only:
        # Random-only: reuse the same seeded sampler as multi-source extras.
        # Neighborhood / popular / user-frequent are unused (no empty-union
        # fallback to the full catalog).
        union: list[int] = []
        if target_row >= 0:
            union.append(int(target_row))
        if random_negatives > 0:
            rng = _rng_for_sample(random_negatives_seed, sample_index)
            extras = sample_random_catalog_rows(
                catalog_size, set(union), random_negatives, rng)
            if extras:
                union.extend(extras)
        if not union:
            # No GT and no random draws — degenerate; fall back so scoring has a set.
            return full, "fallback_full_catalog"
        if target_row >= 0 and target_row not in set(union):
            union.append(int(target_row))
        return torch.tensor(union, dtype=torch.long), "random_pool"
    if radius_km is None:
        return full, "full_catalog"

    neighborhood: list[int] = []
    popular: list[int] = []
    user_rows: list[int] = []
    anchor_ok = math.isfinite(anchor_lat) and math.isfinite(anchor_lon)

    if anchor_ok:
        neighborhood = neighborhood_indices(
            anchor_lat, anchor_lon, catalog_coords, radius_km).tolist()
        if (popular_radius_km is not None and popular_top_k > 0
                and popularity is not None):
            popular = popular_indices(
                anchor_lat, anchor_lon, catalog_coords, popular_radius_km,
                popularity, popular_top_k).tolist()

    if user_frequent_rows:
        user_rows = [int(x) for x in user_frequent_rows if 0 <= int(x) < catalog_size]

    union = _dedupe_rows(neighborhood, popular, user_rows)
    contributed = bool(union)
    if target_row >= 0 and target_row not in set(union):
        union.append(int(target_row))

    if not contributed:
        # No neighborhood / popular / user-frequent members — same spirit as the
        # old empty-neighborhood fallback (avoid ranking in a singleton {target}).
        # Full catalog already includes the target; random negatives are unused.
        return full, "fallback_full_catalog"

    if random_negatives > 0:
        rng = _rng_for_sample(random_negatives_seed, sample_index)
        extras = sample_random_catalog_rows(
            catalog_size, set(union), random_negatives, rng)
        if extras:
            union.extend(extras)

    # Final GT guarantee for multi_source (target may already be present).
    if target_row >= 0 and target_row not in set(union):
        union.append(int(target_row))
    return torch.tensor(union, dtype=torch.long), "multi_source"


def ensure_targets_in_pools(data, pools: Sequence[torch.Tensor]) -> None:
    """Raise if any supervised in-catalog target is missing from its pool.

    Recommend / predict rows (``end`` past the last event) are skipped. OOV
    targets (token → row ``< 0``) cannot appear in the catalog pool and are
    skipped. Every other path must keep the ground-truth row in the pool.
    """
    ends = data["end"]
    n_events = len(data["poi_token"])
    if len(pools) != len(ends):
        raise ValueError("pools must align with samples")
    for i, pool in enumerate(pools):
        end = int(ends[i])
        if end >= n_events:
            continue
        target_row = int(data["poi_token"][end]) - 2
        if target_row < 0:
            continue
        if not (pool == target_row).any():
            raise AssertionError(
                f"sample {i}: in-catalog ground-truth row {target_row} missing "
                f"from candidate pool (size={int(pool.numel())}); val/test pools "
                "must always include the target")


def pools_for_split(data, catalog_coords, radius_km: float | None, catalog_size: int,
                    *,
                    popular_radius_km: float | None = None,
                    popularity: torch.Tensor | None = None,
                    popular_top_k: int = 0,
                    user_frequent: Mapping[str, Sequence[int]] | None = None,
                    random_negatives: int = 0,
                    random_negatives_seed: int = 0,
                    random_only: bool = False,
                    verify_targets: bool = True):
    """Build per-sample candidate pools and resolution labels for a split tensor dict.

    Supervised splits store the true next POI at ``end``. Recommend / predict
    mode uses ``end`` as an exclusive past-the-last-event index (no target row);
    the anchor is still ``end - 1`` and ``target_row`` is ``-1``.

    When ``verify_targets`` is True (default), every in-catalog supervised
    target is checked to be present in its pool (covers train / val / test).
    """
    ends = data["end"]
    lats, lons = data["lat"], data["lon"]
    n_events = len(data["poi_token"])
    trails = data["trails"]
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
        trail_index = int(data["trail"][prev])
        user = str(trails[trail_index][0])
        uf = None
        if user_frequent is not None and not random_only:
            uf = user_frequent.get(user)
        pool, how = resolve_candidate_pool(
            float(lats[prev]), float(lons[prev]), target_row,
            catalog_coords, radius_km, catalog_size,
            popular_radius_km=None if random_only else popular_radius_km,
            popularity=None if random_only else popularity,
            popular_top_k=0 if random_only else popular_top_k,
            user_frequent_rows=uf,
            random_negatives=random_negatives,
            random_negatives_seed=random_negatives_seed,
            sample_index=i,
            random_only=random_only)
        pools.append(pool)
        resolutions.append(how)
    if verify_targets:
        ensure_targets_in_pools(data, pools)
    return pools, resolutions


def shrink_pools_spatial_prior(
        data, pools: Sequence[torch.Tensor], catalog_coords: torch.Tensor,
        *,
        prior: str = "geo",
        top_m: int = 64,
        popularity: torch.Tensor | None = None,
        gravity_beta: float = 1.0,
        radiation_soft_km: float = 0.25,
        user_frequent: Mapping[str, Sequence[int]] | None = None,
        poi_hour: torch.Tensor | None = None,
        mix_weights: Mapping[str, float] | None = None):
    """Coverage-safe Acc@1 shortlist (#10): keep top-M by prior + always GT.

    ``prior``:
      - ``geo``: nearest by haversine (−km)
      - ``gravity``: ``log1p(pop) − β·log1p(km)`` (visit mass from ``popularity``)
      - ``radiation``: Simini hard intervening-opportunity over the pool
      - ``mix``: weighted sum of geo / global-pop / user-frequent / history-revisit /
        tip-hour POI mass (see ``mix_weights``). Missing signals contribute 0.

    Returns ``(new_pools, stats)`` where ``stats['natural_gt_coverage']`` is the
    fraction of in-catalog GT that already sat in the natural top-M (before the
    force-include). Ranking coverage stays 100% for known GT because we always
    append the target when missing.
    """
    prior = str(prior or "geo").lower()
    if prior not in ("geo", "gravity", "radiation", "mix"):
        raise ValueError("candidate spatial prior must be geo|gravity|radiation|mix")
    top_m = int(top_m)
    if top_m < 1:
        raise ValueError("candidate-prior-top-m must be >= 1")
    coords = torch.as_tensor(catalog_coords, dtype=torch.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("catalog_coords must be [V, 2]")
    pop = None
    if popularity is not None:
        pop = torch.as_tensor(popularity, dtype=torch.float64).reshape(-1)
        if pop.numel() != coords.shape[0]:
            raise ValueError("popularity must align with catalog_coords")
    if prior in ("gravity", "radiation") and pop is None:
        raise ValueError(f"prior={prior} requires catalog popularity/visit mass")
    hour_mat = None
    if poi_hour is not None:
        hour_mat = torch.as_tensor(poi_hour, dtype=torch.float64)
        if hour_mat.ndim != 2 or hour_mat.shape[0] != coords.shape[0] or hour_mat.shape[1] != 24:
            raise ValueError("poi_hour must be [V, 24] aligned with catalog_coords")
    weights = {
        "geo": 1.0,
        "pop": 0.5,
        "user": 1.0,
        "hist": 1.0,
        "hour": 0.5,
    }
    if mix_weights:
        for key, value in mix_weights.items():
            if key in weights:
                weights[key] = float(value)

    ends = data["end"]
    starts = data.get("start")
    trails = data.get("trails")
    trail_idx = data.get("trail")
    lats, lons = data["lat"], data["lon"]
    n_events = len(data["poi_token"])
    new_pools = []
    known = 0
    natural_hits = 0
    sizes_before, sizes_after = [], []
    for i, pool in enumerate(pools):
        end = int(ends[i])
        prev = end - 1
        if not 0 <= prev < n_events:
            raise IndexError(f"sample {i}: missing previous check-in for spatial prior")
        if end < n_events:
            target_row = int(data["poi_token"][end]) - 2
        else:
            target_row = -1
        rows = pool.long().tolist()
        sizes_before.append(len(rows))
        if len(rows) <= top_m:
            new_pools.append(pool.long().clone())
            sizes_after.append(len(rows))
            if target_row >= 0:
                known += 1
                natural_hits += 1
            continue
        alat, alon = float(lats[prev]), float(lons[prev])
        if not (math.isfinite(alat) and math.isfinite(alon)):
            # No anchor → keep original pool (cannot justify spatial shrink).
            new_pools.append(pool.long().clone())
            sizes_after.append(len(rows))
            if target_row >= 0:
                known += 1
                natural_hits += 1
            continue
        idx = torch.tensor(rows, dtype=torch.long)
        cand = coords[idx]
        km = haversine_km(alat, alon, cand[:, 0], cand[:, 1])
        if prior == "geo":
            score = -km
        elif prior == "gravity":
            mass = pop[idx].clamp_min(0)
            score = torch.log1p(mass) - float(gravity_beta) * torch.log1p(km.clamp_min(0))
        elif prior == "radiation":
            # Hard radiation over this pool (origin mass = median cand mass).
            mass = pop[idx].clamp_min(0)
            m_i = float(mass.median().item()) if mass.numel() else 0.0
            order = km.argsort()
            m_sorted = mass[order]
            cum = torch.cumsum(m_sorted, dim=0)
            s_sorted = torch.zeros_like(cum)
            s_sorted[1:] = cum[:-1]
            s_ij = torch.zeros_like(km)
            s_ij[order] = s_sorted
            eps = 1e-6
            score = (
                torch.log(mass.clamp_min(eps))
                - torch.log((m_i + s_ij).clamp_min(eps))
                - torch.log((m_i + mass + s_ij).clamp_min(eps))
            )
        else:  # mix
            score = weights["geo"] * (-torch.log1p(km.clamp_min(0)))
            if pop is not None and weights["pop"] != 0:
                score = score + weights["pop"] * torch.log1p(pop[idx].clamp_min(0))
            if weights["user"] != 0 and user_frequent and trails is not None and trail_idx is not None:
                user = str(trails[int(trail_idx[prev])][0])
                uf = user_frequent.get(user) or []
                uf_map = {int(r): 1.0 / (rank + 1.0) for rank, r in enumerate(uf)}
                user_s = torch.tensor([uf_map.get(int(r), 0.0) for r in rows],
                                      dtype=torch.float64)
                score = score + weights["user"] * torch.log1p(user_s)
            if weights["hist"] != 0 and starts is not None:
                s0 = int(starts[i])
                hist = data["poi_token"][s0:end]
                hist_rows = (hist.long() - 2).tolist()
                hist_count: dict[int, int] = {}
                for hr in hist_rows:
                    if hr >= 0:
                        hist_count[hr] = hist_count.get(hr, 0) + 1
                hist_s = torch.tensor([float(hist_count.get(int(r), 0)) for r in rows],
                                      dtype=torch.float64)
                score = score + weights["hist"] * torch.log1p(hist_s)
            if hour_mat is not None and weights["hour"] != 0:
                from .features import us_to_hour_weekday
                tip_hour, _wd = us_to_hour_weekday(int(data["us"][prev]))
                hour_s = hour_mat[idx, int(tip_hour) % 24].clamp_min(0)
                score = score + weights["hour"] * torch.log1p(hour_s)
        # Stable top-M: higher score first.
        keep_n = min(top_m, len(rows))
        top_idx = torch.argsort(score, descending=True, stable=True)[:keep_n].tolist()
        kept = [rows[j] for j in top_idx]
        natural_set = set(kept)
        if target_row >= 0:
            known += 1
            if target_row in natural_set:
                natural_hits += 1
            elif target_row not in natural_set:
                kept.append(int(target_row))
        new_pools.append(torch.tensor(kept, dtype=torch.long))
        sizes_after.append(len(kept))
    ensure_targets_in_pools(data, new_pools)
    coverage = (natural_hits / known) if known else None
    stats = dict(
        prior=prior,
        top_m=top_m,
        gravity_beta=float(gravity_beta),
        mix_weights=dict(weights) if prior == "mix" else None,
        n=len(new_pools),
        known_targets=known,
        natural_gt_hits=natural_hits,
        natural_gt_coverage=coverage,
        size_before_mean=(sum(sizes_before) / len(sizes_before)) if sizes_before else None,
        size_after_mean=(sum(sizes_after) / len(sizes_after)) if sizes_after else None,
        size_after_max=max(sizes_after) if sizes_after else None,
        size_after_min=min(sizes_after) if sizes_after else None,
    )
    return new_pools, stats


def summarize_pool_sizes(pools: Sequence[torch.Tensor],
                         resolutions: Sequence[str] | None = None) -> dict:
    """Per-sample candidate-pool size stats after resolve (union + target + fallback).

    ``len(pool)`` matches the set used for evaluation ranking (and as the parent
    set for training negatives).
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


def mix_adaptive_negatives(probe_rows: torch.Tensor, probe_mask: torch.Tensor,
                           probe_scores: torch.Tensor, targets: torch.Tensor,
                           n_negatives: int, hard_frac: float, rng: random.Random):
    """Build train rows from a scored probe: hard top-k + random fill (#37).

    ``probe_rows`` / ``probe_scores`` are [B, P] with positive typically at col 0
    (as from ``sample_candidates``). Returns ``(rows, labels, mask)`` with
    positive at column 0 and ``n_negatives`` mixed negatives (or fewer if pool thin).
    """
    if not 0.0 <= float(hard_frac) <= 1.0:
        raise ValueError("neg-hard-frac must be in [0, 1]")
    if int(n_negatives) < 1:
        raise ValueError("adaptive_mix requires a positive --negatives count")
    bsz = int(probe_rows.shape[0])
    row_lists = []
    for b in range(bsz):
        tgt = int(targets[b].item())
        valid = probe_mask[b].tolist()
        members = [int(probe_rows[b, j].item()) for j, ok in enumerate(valid) if ok]
        # Ensure target present; others are probe negatives / extras.
        others = [r for r in dict.fromkeys(members) if r != tgt]
        if not others:
            row_lists.append([tgt])
            continue
        want = min(int(n_negatives), len(others))
        n_hard = int(round(want * float(hard_frac)))
        n_hard = max(0, min(n_hard, want))
        n_rand = want - n_hard
        # Score only probe columns that are true negatives.
        scored = []
        for j, ok in enumerate(valid):
            if not ok:
                continue
            row = int(probe_rows[b, j].item())
            if row == tgt:
                continue
            scored.append((float(probe_scores[b, j].item()), row))
        scored.sort(key=lambda x: x[0], reverse=True)
        hard = []
        seen = set()
        for _, row in scored:
            if row in seen:
                continue
            hard.append(row)
            seen.add(row)
            if len(hard) >= n_hard:
                break
        remain = [r for r in others if r not in seen]
        if n_rand > 0:
            if n_rand >= len(remain):
                rand = remain
            else:
                rand = rng.sample(remain, n_rand)
        else:
            rand = []
        # If hard short (flat scores / tiny probe), backfill from remain.
        chosen = hard + rand
        if len(chosen) < want:
            extra_pool = [r for r in others if r not in set(chosen)]
            need = want - len(chosen)
            if need >= len(extra_pool):
                chosen.extend(extra_pool)
            elif need > 0:
                chosen.extend(rng.sample(extra_pool, need))
        row_lists.append([tgt, *chosen[:want]])
    width = max(len(row) for row in row_lists)
    padded = torch.full((len(row_lists), width), -1, dtype=torch.long)
    mask = torch.zeros(len(row_lists), width, dtype=torch.bool)
    for i, row in enumerate(row_lists):
        padded[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, :len(row)] = True
    labels = torch.zeros(len(row_lists), dtype=torch.long)
    return padded, labels, mask


def sample_candidates(targets: torch.Tensor, catalog_size: int, negatives: int,
                      rng: random.Random, pools: Sequence[torch.Tensor] | None = None,
                      max_candidates: int | None = None):
    """Sample training rows: positive at column 0, then negatives from each pool.

    Returns ``(rows, labels, candidate_mask)``. ``pools is None`` preserves the
    historical uniform negative sampler over the full catalog. With constructed
    pools, negatives are drawn only from that sample's resolved set
    (``negatives=-1`` ⇒ all other pool members, optionally capped by
    ``max_candidates``). Evaluation should pass the full pool without sampling.
    """
    def negative_budget(n_others: int) -> int:
        limit = n_others
        if negatives >= 0:
            limit = min(limit, negatives)
        if max_candidates is not None and max_candidates > 0:
            limit = min(limit, max_candidates - 1)
        return max(0, limit)

    if pools is None:
        if max_candidates is not None and max_candidates > 0:
            # Cap including the positive.
            cap_neg = max(0, max_candidates - 1)
        else:
            cap_neg = catalog_size - 1
        if negatives < 0:
            want = cap_neg
        else:
            want = min(negatives, cap_neg)
        if want >= catalog_size - 1:
            rows = torch.arange(catalog_size).expand(len(targets), -1).contiguous()
            return rows, targets.clone(), torch.ones_like(rows, dtype=torch.bool)
        row_lists = []
        for target in targets.tolist():
            chosen = rng.sample(range(catalog_size - 1), want)
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
        want = negative_budget(len(others))
        if want >= len(others):
            chosen = others
        else:
            chosen = rng.sample(others, want)
        row_lists.append([target, *chosen])
    width = max(len(row) for row in row_lists)
    padded = torch.full((len(row_lists), width), -1, dtype=torch.long)
    mask = torch.zeros(len(row_lists), width, dtype=torch.bool)
    for i, row in enumerate(row_lists):
        padded[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, :len(row)] = True
    return padded, torch.zeros(len(targets), dtype=torch.long), mask


def enrich_pool_with_extra_random(
        pool: Sequence[int] | torch.Tensor, catalog_size: int, extra_random: int,
        rng: random.Random) -> list[int]:
    """Union a constructed pool with uniform catalog rows.

    ``extra_random``:
      * ``> 0`` — add up to that many new catalog rows
      * ``0`` — no extras
      * ``< 0`` — add about as many as the current pool size (``len(pool)``)
    """
    members = list(dict.fromkeys(int(x) for x in (
        pool.tolist() if torch.is_tensor(pool) else pool)))
    extra = int(extra_random or 0)
    if extra < 0:
        extra = len(members)
    if extra <= 0 or catalog_size <= 0:
        return members
    existing = set(members)
    candidates = [i for i in range(catalog_size) if i not in existing]
    if not candidates:
        return members
    take = min(extra, len(candidates))
    members.extend(rng.sample(candidates, take))
    return members


def finalize_candidate_pool(
        members: Sequence[int], target: int, final_negatives: int,
        rng: random.Random) -> list[int]:
    """Force-include in-catalog GT, then sample ``final_negatives`` other rows.

    Total size is ``≤ final_negatives + 1`` (GT + negatives). Example: 63
    negatives → up to **64** candidates with GT first.
    If the enriched pool is smaller, keeps everyone (GT first).
    """
    n_neg = int(final_negatives)
    if n_neg < 0:
        raise ValueError("final_negatives must be >= 0")
    members = list(dict.fromkeys(int(x) for x in members))
    if target >= 0 and target not in members:
        members.append(target)
    if target < 0:
        # OOV target: cannot force-include; still cap to n_neg (no +1).
        if len(members) <= n_neg:
            return members
        return rng.sample(members, n_neg) if n_neg else []
    others = [row for row in members if row != target]
    want = min(len(others), n_neg)
    if want >= len(others):
        chosen = others
    else:
        chosen = rng.sample(others, want)
    return [target, *chosen]


def build_final_pools(
        pools: Sequence[torch.Tensor],
        targets: Sequence[int] | torch.Tensor,
        catalog_size: int,
        *,
        final_negatives: int,
        extra_random: int = -1,
        seed: int = 0,
        sample_indices: Sequence[int] | None = None) -> list[torch.Tensor]:
    """Eval-style pools ∪ extra random → sample ``final_negatives`` + GT.

    ``extra_random < 0`` ⇒ add ~``len(pool)`` random catalog rows per sample.
    Deterministic per sample via ``seed + sample_index``. Used for train and
    eval/test so both share one shortlist protocol.
    """
    if torch.is_tensor(targets):
        target_list = [int(x) for x in targets.tolist()]
    else:
        target_list = [int(x) for x in targets]
    if len(pools) != len(target_list):
        raise ValueError("pools must align with targets")
    out: list[torch.Tensor] = []
    for i, (pool, target) in enumerate(zip(pools, target_list)):
        sample_i = int(sample_indices[i]) if sample_indices is not None else i
        rng = random.Random(int(seed) + 10007 * sample_i + 17)
        enriched = enrich_pool_with_extra_random(
            pool, catalog_size, extra_random, rng)
        final = finalize_candidate_pool(enriched, target, final_negatives, rng)
        out.append(torch.tensor(final, dtype=torch.long))
    return out


def sample_final_candidates(
        targets: torch.Tensor, catalog_size: int, final_negatives: int,
        rng: random.Random, pools: Sequence[torch.Tensor],
        *,
        extra_random: int = -1,
        seed: int = 0,
        sample_indices: Sequence[int] | None = None):
    """Train-time rows/labels/mask under the final-negatives+GT shortlist protocol."""
    if pools is None:
        raise ValueError("sample_final_candidates requires constructed pools")
    finals = build_final_pools(
        pools, targets, catalog_size,
        final_negatives=final_negatives, extra_random=extra_random, seed=seed,
        sample_indices=sample_indices)
    # Rebuild with a shared RNG path already used above; rows put GT at col 0.
    row_lists = []
    for target, pool in zip(targets.tolist(), finals):
        members = [int(x) for x in pool.tolist()]
        if target >= 0:
            others = [r for r in members if r != target]
            row_lists.append([target, *others])
        else:
            row_lists.append(members)
    width = max((len(r) for r in row_lists), default=0)
    padded = torch.full((len(row_lists), width), -1, dtype=torch.long)
    mask = torch.zeros(len(row_lists), width, dtype=torch.bool)
    for i, row in enumerate(row_lists):
        if not row:
            continue
        padded[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, :len(row)] = True
    return padded, torch.zeros(len(targets), dtype=torch.long), mask


def candidate_construction_record(
        method: str, stats: dict, radius_km: float | None,
        coordinate_imputation: dict | None = None,
        *,
        popular_radius: str = DEFAULT_POPULAR_RADIUS,
        popular_radius_multiplier: float = DEFAULT_POPULAR_RADIUS_MULTIPLIER,
        popular_radius_km: float | None = None,
        popular_top_k: int = DEFAULT_POPULAR_TOP_K,
        popular_by: str = DEFAULT_POPULAR_BY,
        user_frequent_top_k: int = DEFAULT_USER_FREQUENT_TOP_K,
        user_frequent_min_visits: int = DEFAULT_USER_FREQUENT_MIN_VISITS,
        random_negatives: int = DEFAULT_RANDOM_NEGATIVES,
        random_negatives_seed: int = DEFAULT_RANDOM_NEGATIVES_SEED,
        poi_visit_counts: Sequence[int] | None = None,
        poi_target_counts: Sequence[int] | None = None) -> dict:
    """Manifest block documenting how train/eval candidates are built."""
    if method == "full":
        mode = "full_catalog"
        sources = ["full_catalog"]
        spatial = False
        random_mode = False
    elif method == "random":
        mode = "random_pool"
        sources = ["random_negatives"]
        spatial = False
        random_mode = True
    else:
        mode = "multi_source"
        sources = ["neighborhood", "popular", "user_frequent", "random_negatives"]
        spatial = True
        random_mode = False
    if random_mode:
        notes = (
            "Candidates = up to "
            f"{random_negatives} uniform random catalog rows (seeded via "
            "random_negatives_seed + sample index), always including the in-catalog "
            "true target. Neighborhood / popular / user-frequent sources are unused "
            "(no empty-union fallback to the full catalog). Train may sample "
            "negatives from the pool; evaluation ranks inside the full constructed pool. "
            "Pool size is typically smaller than multi-source — Acc@1 is not pool-size "
            "normalized; raise --random-negatives toward baseline mean pool size for a "
            "fairer difficulty match."
        )
        empty_fallback = None
    elif spatial:
        notes = (
            "Candidates = union of (1) training-catalog POIs within radius_km of the "
            "last observed check-in, (2) top popular_top_k POIs by train "
            f"{popular_by} frequency within popular_radius_km, (3) this user's "
            "top user_frequent_top_k training venues (min visits), (4) up to "
            f"{random_negatives} uniform random catalog negatives (seeded). Deduped; "
            "true target always added when in catalog (val/test included — no path "
            "may omit it). If neighborhood/popular/user-frequent all contribute "
            "nothing, that sample falls back to the full catalog. Train may sample "
            "negatives from the pool; evaluation ranks inside the full constructed pool."
        )
        empty_fallback = "full_catalog"
    else:
        notes = (
            "Candidates = the full training catalog for every sample "
            "(--candidate-radius full). Popular / user-frequent / random-negative "
            "sources are disabled. Train may sample negatives; evaluation ranks "
            "inside the full catalog."
        )
        empty_fallback = None
    return dict(
        method=mode,
        radius_statistic=method,
        radius_km=radius_km if spatial else None,
        default_statistic=DEFAULT_RADIUS,
        distance_stats=stats,
        popular_radius=popular_radius if spatial else None,
        popular_radius_multiplier=popular_radius_multiplier if spatial else None,
        popular_radius_km=popular_radius_km if spatial else None,
        popular_top_k=popular_top_k if spatial else 0,
        popular_by=popular_by if spatial else None,
        user_frequent_top_k=user_frequent_top_k if spatial else 0,
        user_frequent_min_visits=user_frequent_min_visits if spatial else None,
        random_negatives=(
            random_negatives if (spatial or random_mode) else 0),
        random_negatives_seed=(
            random_negatives_seed if (spatial or random_mode) else None),
        sources=sources,
        anchor="previous_checkin_coordinates" if spatial else (
            "none_random_pool" if random_mode else "none_full_catalog"),
        catalog_coordinates=(
            "finite mean lat/lon per catalog POI; missing check-in coords are imputed "
            "via trail-adjacent midpoints, then trail mean, then global observed mean"
        ),
        coordinate_imputation=coordinate_imputation or dict(
            fallback_order=list(COORD_FALLBACK_ORDER)),
        poi_visit_counts=list(poi_visit_counts) if poi_visit_counts is not None else None,
        poi_target_counts=list(poi_target_counts) if poi_target_counts is not None else None,
        include_target=True,
        ground_truth_always_in_pool=True,
        empty_union_fallback=empty_fallback,
        train_eval_shared=True,
        train_sampling=(
            "train may subsample via --negatives / --max-train-candidates; "
            "evaluate uses the full constructed pool"
        ),
        eval_uses_full_pool=True,
        notes=notes,
    )
