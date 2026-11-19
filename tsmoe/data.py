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
    from .candidates import haversine_km

    t = event.time
    hour = 2 * math.pi * (t.hour + t.minute / 60 + t.second / 3600) / 24
    day = 2 * math.pi * t.weekday() / 7
    gap = (t - previous.time).total_seconds() / 3600 if previous else 0.0
    time = [math.sin(hour), math.cos(hour), math.sin(day), math.cos(day),
            math.log1p(gap), float(previous is not None)]
    loc = [0.0] * 8
    if event.coord is not None:
        lat, lon = map(math.radians, event.coord)
        loc[:3] = [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]
        loc[6] = 1.0
        if previous and previous.coord is not None:
            distance = haversine_km(*previous.coord, *event.coord)
            prev_lat, prev_lon = map(math.radians, previous.coord)
            dlon = lon - prev_lon
            east = math.sin(dlon) * math.cos(lat)
            north = math.cos(prev_lat) * math.sin(lat) - math.sin(prev_lat) * math.cos(lat) * math.cos(dlon)
            norm = math.hypot(east, north)
            loc[3:6] = [math.log1p(distance), east / norm if norm > 1e-12 else 0.,
                         north / norm if norm > 1e-12 else 0.]
            loc[7] = 1.0
    return time, loc


def build_split(groups, categories, catalog, cat_vocab, max_length, target_mode):
    """Store each event once. 'end' identifies the true target, excluded from history."""
    poi_rows = {value: i for i, value in enumerate(catalog)}
    cat_rows = {value: i for i, value in enumerate(cat_vocab)}
    columns = defaultdict(list)
    starts, ends, trail_keys = [], [], []
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
                starts.append(offset + max(0, j - max_length))
                ends.append(offset + j)
        if target_mode == "predict" and events:
            starts.append(offset + max(0, len(events) - max_length))
            ends.append(offset + len(events))
    if not ends:
        raise ValueError("split has no usable next-POI target after preprocessing")
    data = {k: torch.tensor(v, dtype=torch.float64 if k in ("T", "L", "lat", "lon") else torch.long)
            for k, v in columns.items()}
    return {**data, "start": torch.tensor(starts), "end": torch.tensor(ends), "trails": trail_keys}


def make_batch(data, indices, *, offline=False, labels=True):
    """Gather once with a shared right-padding mask, resetting differences at the left edge."""
    start, end = data["start"][indices], data["end"][indices]
    length = end - start
    pos = torch.arange(int(length.max()))[None, :]
    mask = pos < length[:, None]
    ix = torch.minimum(start[:, None] + pos, end[:, None] - 1)
    t, l = data["T"][ix], data["L"][ix]
    t[:, 0, 4:] = 0.
    l[:, 0, 3:6] = 0.
    l[:, 0, 7] = 0.
    p, c, pad = ("P", "C", -1) if offline else ("poi_token", "cat_token", 0)
    history = {"P": data[p][ix].masked_fill(~mask, pad), "C": data[c][ix].masked_fill(~mask, pad),
               "T": t.masked_fill(~mask[..., None], 0.), "L": l.masked_fill(~mask[..., None], 0.)}
    if not offline:
        history["T"], history["L"] = history["T"].float(), history["L"].float()
    return history, mask, data["poi_token"][end] - 2 if labels else None
