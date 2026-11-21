"""Read official Massive-STEPS check-in splits and construct aligned prefix windows."""
import csv
import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

SPLITS = ("train", "validation", "test")
FIELDS = {"trail_id", "user_id", "venue_id", "latitude", "longitude", "timestamp"}
NULL = {"", "na", "nan", "none", "null", "-1", "-2"}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# STHGCN-aligned T/L schema (Yan et al., SIGIR 2023; Massive-STEPS baseline).
# TimeEncoder / distance_encoder_type=time use Φ(Δ)_i = cos(Δ · 10^{-linspace(0,9,D)}).
# Δt in hours (seconds/3600); Δs Haversine km. Absolute hour/weekday as one-hots
# (check-in Embedding inputs). MLP projection ≈ W_t / W_s after Φ.
FEATURE_METHOD = "sthgcn_tl_features_v1"
STHGCN_BASIS_DIM = 16
T_HOUR_DIM = 24
T_WEEKDAY_DIM = 7
T_DIM = T_HOUR_DIM + T_WEEKDAY_DIM + STHGCN_BASIS_DIM + 1  # 48
L_DIM = STHGCN_BASIS_DIM + 2  # 18
# Slice layout for make_batch first-step relative reset.
T_RELATIVE_SLICE = slice(T_HOUR_DIM + T_WEEKDAY_DIM, T_DIM)  # Φ(Δt) + gap_valid
L_BASIS_SLICE = slice(0, STHGCN_BASIS_DIM)
L_STEP_VALID_INDEX = STHGCN_BASIS_DIM + 1


def sthgcn_basis_freqs(dim=STHGCN_BASIS_DIM):
    """Fixed TimeEncoder init frequencies: 1 / 10^{linspace(0, 9, dim)}."""
    if dim < 1:
        raise ValueError("basis dim must be positive")
    if dim == 1:
        return (1.0,)
    return tuple(10.0 ** (-9.0 * i / (dim - 1)) for i in range(dim))


_STHGCN_FREQS = sthgcn_basis_freqs(STHGCN_BASIS_DIM)


def sthgcn_phi(delta, freqs=_STHGCN_FREQS):
    """STHGCN relative encoding Φ(Δ)_i = cos(Δ · freq_i) with phase 0 (init)."""
    return [math.cos(float(delta) * f) for f in freqs]


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def integer_id(value):
    text = (value or "").strip()
    if text.lower() in NULL:
        return -2
    ident = int(text)
    if not 0 <= ident < 2**63:
        raise ValueError(f"expected a nonnegative int64 identifier, got {text!r}")
    return ident


@dataclass(frozen=True)
class Event:
    poi: int
    stable_category: str
    category_code: int
    time: datetime
    coord: tuple | None
    line: int


def read_split(path):
    groups = defaultdict(list)
    owners, seen, counts = {}, set(), Counter()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        if not FIELDS <= fields or not fields.intersection({"venue_category_id_code", "venue_category_id"}):
            raise ValueError(f"{path}: expected an official check-in CSV. Missing {FIELDS - fields}; check category fields.")
        for line, row in enumerate(reader, 2):
            counts["csv_rows"] += 1
            user, trail = row["user_id"].strip(), row["trail_id"].strip()
            if not user or not trail:
                raise ValueError(f"{path}:{line}: missing user_id or trail_id")
            if owners.setdefault(trail, user) != user:
                raise ValueError(f"{path}:{line}: trail {trail!r} has multiple owners")
            try:
                time = datetime.fromisoformat(row["timestamp"].strip().replace("Z", "+00:00"))
                time = (time if time.tzinfo else time.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
            except (ValueError, OverflowError, AttributeError):
                counts["invalid_timestamp_removed"] += 1
                continue
            poi = integer_id(row["venue_id"])
            event_key = (user, trail, time, poi)
            if event_key in seen:
                counts["duplicates_removed"] += 1
                continue
            seen.add(event_key)
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
                coord = (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None
            except (TypeError, ValueError):
                coord = None
            if coord is None:
                counts["missing_coordinates"] += 1
            stable = (row.get("venue_category_id") or "").strip()
            stable = "" if stable.lower() in NULL else stable
            code = integer_id(row.get("venue_category_id_code"))
            groups[user, trail].append(Event(poi, stable, code, time, coord, line))
    for events in groups.values():
        events.sort(key=lambda e: (e.time, e.line))
    counts["trails"] = len(groups)
    return dict(groups), dict(counts)


class Categories:
    """Use consistent training codes; reconcile recoded splits through stable IDs."""
    def __init__(self, groups):
        ids_by_code, codes_by_id = defaultdict(set), defaultdict(set)
        for events in groups.values():
            for e in events:
                if e.stable_category and e.category_code >= 0:
                    ids_by_code[e.category_code].add(e.stable_category)
                    codes_by_id[e.stable_category].add(e.category_code)
        self.ambiguous = {c for c, ids in ids_by_code.items() if len(ids) > 1}
        self.codes = {key: next(iter(codes)) for key, codes in codes_by_id.items()
                      if len(codes) == 1 and not codes & self.ambiguous}

    @classmethod
    def from_manifest(cls, manifest):
        mapping = cls({})
        mapping.codes = manifest["category_mapping"]
        mapping.ambiguous = set(manifest["ambiguous_category_codes"])
        return mapping

    def __call__(self, event):
        if event.stable_category:
            if event.stable_category in self.codes:
                return self.codes[event.stable_category]
            # Separate namespace from Massive-STEPS int category codes, including zero.
            digest = hashlib.sha256(event.stable_category.encode()).digest()
            return 2**62 + (int.from_bytes(digest[:8], "big") % 2**62)
        if event.category_code >= 2**62:
            raise ValueError("venue_category_id_code outside the supported Massive-STEPS range")
        return -2 if event.category_code in self.ambiguous else event.category_code


def numeric_features(event, previous):
    """STHGCN-aligned T/L vectors (see FEATURE_METHOD / Basic §4.3)."""
    from .candidates import haversine_km

    t = event.time
    hour_oh = [0.0] * T_HOUR_DIM
    hour_oh[t.hour % T_HOUR_DIM] = 1.0
    weekday_oh = [0.0] * T_WEEKDAY_DIM
    weekday_oh[t.weekday() % T_WEEKDAY_DIM] = 1.0
    # No previous → zero Φ (not Φ(0)=1…1) and gap_valid=0; make_batch also clears step 0.
    if previous is None:
        time = hour_oh + weekday_oh + [0.0] * STHGCN_BASIS_DIM + [0.0]
    else:
        gap_hours = (t - previous.time).total_seconds() / 3600.0
        time = hour_oh + weekday_oh + sthgcn_phi(gap_hours) + [1.0]

    loc = [0.0] * L_DIM
    if event.coord is not None:
        loc[STHGCN_BASIS_DIM] = 1.0  # coord_valid
        if previous and previous.coord is not None:
            distance_km = haversine_km(*previous.coord, *event.coord)
            loc[:STHGCN_BASIS_DIM] = sthgcn_phi(distance_km)
            loc[L_STEP_VALID_INDEX] = 1.0
    return time, loc


def train_user_ids(groups):
    """User ids that own at least one trail in ``groups`` (Massive-STEPS ``user_id``)."""
    return {user for user, _trail in groups}


def filter_groups_by_users(groups, allowed_users):
    """Keep trails whose ``user_id`` is in ``allowed_users`` (STHGCN-style)."""
    allowed = set(allowed_users)
    return {key: events for key, events in groups.items() if key[0] in allowed}


def build_split(groups, categories, catalog, cat_vocab, max_length, target_mode,
                *, require_in_catalog_target=False):
    """Store each event once. 'end' identifies the true target, excluded from history.

    When ``require_in_catalog_target`` is True (prepare ``--filter-unknown on``),
    skip prediction samples whose true next POI is outside the training catalog —
    matching STHGCN ``remove_unseen_user_poi`` on the sample's target ``PoiId``.
    History tokens for OOV POIs still map to UNK; only the supervised target is filtered.
    """
    poi_rows = {value: i for i, value in enumerate(catalog)}
    cat_rows = {value: i for i, value in enumerate(cat_vocab)}
    columns = defaultdict(list)
    starts, ends, trail_keys = [], [], []
    skipped_unknown_targets = 0
    for key, events in sorted(groups.items()):
        trail_index, offset = len(trail_keys), len(columns["P"])
        trail_keys.append(list(key))
        for j, event in enumerate(events):
            category = categories(event)
            t, l = numeric_features(event, events[j - 1] if j else None)
            delta = event.time - EPOCH
            time_us = (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds
            lat, lon = (event.coord if event.coord is not None
                        else (float("nan"), float("nan")))
            fields = {"P": event.poi, "C": category, "T": t, "L": l,
                      "lat": lat, "lon": lon,
                      "poi_token": poi_rows.get(event.poi, -1) + 2,
                      "cat_token": cat_rows.get(category, -1) + 2, "us": time_us,
                      "line": event.line, "trail": trail_index}
            for name, value in fields.items():
                columns[name].append(value)
            if j and event.poi >= 0 and (target_mode == "all" or (target_mode == "last" and j == len(events) - 1)):
                if require_in_catalog_target and event.poi not in poi_rows:
                    skipped_unknown_targets += 1
                    continue
                starts.append(offset + max(0, j - max_length))
                ends.append(offset + j)
        if target_mode == "predict" and events:
            starts.append(offset + max(0, len(events) - max_length))
            ends.append(offset + len(events))
    if not ends:
        raise ValueError("split has no usable next-POI target after preprocessing")
    data = {k: torch.tensor(v, dtype=torch.float64 if k in ("T", "L", "lat", "lon") else torch.long)
            for k, v in columns.items()}
    data = {**data, "start": torch.tensor(starts), "end": torch.tensor(ends), "trails": trail_keys}
    return data, dict(skipped_unknown_targets=skipped_unknown_targets)


def make_batch(data, indices, *, offline=False, labels=True):
    """Gather once with a shared right-padding mask, resetting differences at the left edge."""
    start, end = data["start"][indices], data["end"][indices]
    length = end - start
    pos = torch.arange(int(length.max()))[None, :]
    mask = pos < length[:, None]
    ix = torch.minimum(start[:, None] + pos, end[:, None] - 1)
    t, l = data["T"][ix], data["L"][ix]
    # First history step has no previous event: clear relative Φ / validity only.
    t[:, 0, T_RELATIVE_SLICE] = 0.
    l[:, 0, L_BASIS_SLICE] = 0.
    l[:, 0, L_STEP_VALID_INDEX] = 0.
    p, c, pad = ("P", "C", -1) if offline else ("poi_token", "cat_token", 0)
    history = {"P": data[p][ix].masked_fill(~mask, pad), "C": data[c][ix].masked_fill(~mask, pad),
               "T": t.masked_fill(~mask[..., None], 0.), "L": l.masked_fill(~mask[..., None], 0.)}
    if not offline:
        history["T"], history["L"] = history["T"].float(), history["L"].float()
    return history, mask, data["poi_token"][end] - 2 if labels else None
