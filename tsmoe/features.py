"""Optional candidate / session / preference side features (runtime; no prepare).

Used when ``--candidate-context`` / ``--session-summary`` / ``--user-pref`` are on.
Defaults are off; gathering is a no-op when all related flags are off.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone

import torch

from .candidates import haversine_km

# Δgeo: log1p(haversine_km), signed_log1p(east_km), signed_log1p(north_km).
GEO_FEAT_DIM = 3
# last hour one-hot(24) + weekday one-hot(7).
LAST_SLOT_DIM = 24 + 7
# Session: traj-length scalar + activity-center→last km (log1p) + valid flag.
SESSION_SCALAR_DIM = 3
# User-hour soft hist (24) — concatenated with user-cat soft hist in the model.
USER_HOUR_DIM = 24
TRAJ_LENGTH_BINS = 8


def signed_log1p(x):
    """``sign(x) * log1p(|x|)`` for east/north offsets (km)."""
    if torch.is_tensor(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    if x == 0:
        return 0.0
    return math.copysign(math.log1p(abs(x)), x)


def east_north_km(lat1, lon1, lat2, lon2):
    """Local east / north offsets in km (equirectangular; fine for short hops)."""
    to_tensor = any(torch.is_tensor(v) for v in (lat1, lon1, lat2, lon2))
    if to_tensor:
        lat1 = torch.as_tensor(lat1, dtype=torch.float64)
        lon1 = torch.as_tensor(lon1, dtype=torch.float64)
        lat2 = torch.as_tensor(lat2, dtype=torch.float64)
        lon2 = torch.as_tensor(lon2, dtype=torch.float64)
        north = (lat2 - lat1) * 111.0
        east = (lon2 - lon1) * 111.0 * torch.cos(torch.deg2rad(lat1))
        return east, north
    north = (lat2 - lat1) * 111.0
    east = (lon2 - lon1) * 111.0 * math.cos(math.radians(lat1))
    return east, north


def delta_geo_features(lat1, lon1, lat2, lon2):
    """Return ``[log1p(hav), signed_log1p(east), signed_log1p(north)]`` (km)."""
    if any(v != v for v in (lat1, lon1, lat2, lon2)):  # NaN check
        return [0.0, 0.0, 0.0]
    hav = float(haversine_km(lat1, lon1, lat2, lon2))
    east, north = east_north_km(lat1, lon1, lat2, lon2)
    return [math.log1p(hav), signed_log1p(east), signed_log1p(north)]


def us_to_hour_weekday(us):
    """UTC hour (0–23) and weekday (0=Mon … 6=Sun) from microseconds since epoch."""
    dt = datetime.fromtimestamp(int(us) / 1_000_000.0, tz=timezone.utc)
    return dt.hour % 24, dt.weekday() % 7


def last_slot_onehot(hour, weekday):
    vec = [0.0] * LAST_SLOT_DIM
    vec[int(hour) % 24] = 1.0
    vec[24 + (int(weekday) % 7)] = 1.0
    return vec


def traj_length_bin(length, max_length, n_bins=TRAJ_LENGTH_BINS):
    """Equal-width bin id in ``0 .. n_bins-1`` from history length."""
    if max_length < 1 or n_bins < 1:
        return 0
    length = max(0, min(int(length), int(max_length)))
    # Map 0..max_length → 0..n_bins-1.
    return min(n_bins - 1, (length * n_bins) // (max_length + 1))


def build_poi_cat_table(data, n_pois):
    """Mode category catalog-row per POI from split events (``-1`` if unseen)."""
    counts = [Counter() for _ in range(n_pois)]
    poi_tok, cat_tok = data["poi_token"], data["cat_token"]
    for i in range(len(poi_tok)):
        pr = int(poi_tok[i]) - 2
        cr = int(cat_tok[i]) - 2
        if 0 <= pr < n_pois and cr >= 0:
            counts[pr][cr] += 1
    table = torch.full((n_pois,), -1, dtype=torch.long)
    for i, c in enumerate(counts):
        if c:
            table[i] = c.most_common(1)[0][0]
    return table


def sample_last_tip(data, sample_index):
    """Last observed history event index (tip) for a sample, or ``None``."""
    start = int(data["start"][sample_index])
    end = int(data["end"][sample_index])
    if end <= start:
        return None
    return end - 1


def sample_activity_center(data, sample_index):
    """Mean lat/lon over history events with valid coords; ``None`` if empty."""
    start = int(data["start"][sample_index])
    end = int(data["end"][sample_index])
    lats, lons = [], []
    for i in range(start, end):
        lat = float(data["lat"][i])
        lon = float(data["lon"][i])
        if lat == lat and lon == lon:
            lats.append(lat)
            lons.append(lon)
    if not lats:
        return None
    return sum(lats) / len(lats), sum(lons) / len(lons)


def gather_side_features(
    data,
    indices,
    candidate_rows_list,
    *,
    poi_coords,
    poi_cat_table,
    n_categories,
    max_length,
    tables=None,
    candidate_context=False,
    candidate_context_time="off",
    session_summary=False,
    user_pref=False,
    user_id_emb=False,
    user_vocab=None,
    transition_pmi=False,
):
    """Build optional side-feature tensors aligned with a chronological batch.

    Returns a dict of tensors (may be empty). Does **not** update count tables.
    """
    out = {}
    idxs = indices.tolist() if torch.is_tensor(indices) else list(indices)
    widths = [len(c) for c in candidate_rows_list]
    width = max(widths) if widths else 0
    n_pois = int(poi_coords.shape[0]) if poi_coords is not None else 0

    if user_id_emb:
        from .counts import user_id_of
        # pad=0, UNK=1, train user row i → token i+2 (mirrors POI/cat token scheme).
        vocab = user_vocab or {}
        tokens = torch.ones(len(idxs), dtype=torch.long)  # default UNK
        for bi, sample_index in enumerate(idxs):
            uid = str(user_id_of(data, sample_index))
            if uid in vocab:
                tokens[bi] = int(vocab[uid]) + 2
        out["user_token"] = tokens

    need_cand = candidate_context or candidate_context_time != "off"
    if need_cand:
        cat_rows = torch.full((len(idxs), width), -1, dtype=torch.long)
        geo = torch.zeros(len(idxs), width, GEO_FEAT_DIM, dtype=torch.float32)
        last_slot = torch.zeros(len(idxs), LAST_SLOT_DIM, dtype=torch.float32)
        slot_feat = torch.zeros(len(idxs), width, 4, dtype=torch.float32)
        for bi, sample_index in enumerate(idxs):
            tip = sample_last_tip(data, sample_index)
            last_lat = last_lon = float("nan")
            hour = weekday = 0
            if tip is not None:
                last_lat = float(data["lat"][tip])
                last_lon = float(data["lon"][tip])
                hour, weekday = us_to_hour_weekday(int(data["us"][tip]))
                if candidate_context_time in ("last_slot", "causal_slot"):
                    last_slot[bi] = torch.tensor(last_slot_onehot(hour, weekday))
            rows = candidate_rows_list[bi]
            if not isinstance(rows, list):
                rows = list(rows) if not torch.is_tensor(rows) else rows.tolist()
            for ci, row in enumerate(rows):
                r = int(row)
                if 0 <= r < n_pois and poi_cat_table is not None:
                    cat_rows[bi, ci] = int(poi_cat_table[r])
                if candidate_context and 0 <= r < n_pois:
                    clat = float(poi_coords[r, 0])
                    clon = float(poi_coords[r, 1])
                    geo[bi, ci] = torch.tensor(
                        delta_geo_features(last_lat, last_lon, clat, clon),
                        dtype=torch.float32)
                if (candidate_context_time == "causal_slot" and tables is not None
                        and 0 <= r < n_pois):
                    from .counts import user_id_of
                    user = user_id_of(data, sample_index)
                    slot_feat[bi, ci] = torch.tensor(
                        tables.poi_hour_feat(user, r, hour), dtype=torch.float32)
        out["cand_cat_rows"] = cat_rows
        out["delta_geo"] = geo
        if candidate_context_time != "off":
            out["last_slot"] = last_slot
        if candidate_context_time == "causal_slot":
            out["cand_slot_feat"] = slot_feat

    if session_summary:
        sess = torch.zeros(len(idxs), SESSION_SCALAR_DIM, dtype=torch.float32)
        length_bins = torch.zeros(len(idxs), dtype=torch.long)
        # Soft category hist over the window (log1p counts, L1-normalized).
        cat_hist = torch.zeros(len(idxs), n_categories, dtype=torch.float32)
        for bi, sample_index in enumerate(idxs):
            start = int(data["start"][sample_index])
            end = int(data["end"][sample_index])
            length = max(0, end - start)
            length_bins[bi] = traj_length_bin(length, max_length)
            sess[bi, 0] = float(length) / float(max(max_length, 1))
            tip = sample_last_tip(data, sample_index)
            center = sample_activity_center(data, sample_index)
            if tip is not None and center is not None:
                last_lat = float(data["lat"][tip])
                last_lon = float(data["lon"][tip])
                if last_lat == last_lat and last_lon == last_lon:
                    dist = float(haversine_km(center[0], center[1], last_lat, last_lon))
                    sess[bi, 1] = math.log1p(dist)
                    sess[bi, 2] = 1.0
            for i in range(start, end):
                cr = int(data["cat_token"][i]) - 2
                if 0 <= cr < n_categories:
                    cat_hist[bi, cr] += 1.0
            total = float(cat_hist[bi].sum())
            if total > 0:
                cat_hist[bi] = torch.log1p(cat_hist[bi])
                cat_hist[bi] = cat_hist[bi] / cat_hist[bi].sum().clamp_min(1e-8)
        out["session_scalars"] = sess
        out["session_length_bin"] = length_bins
        out["session_cat_hist"] = cat_hist

    if user_pref and tables is not None:
        from .counts import user_id_of
        cat_pref = torch.zeros(len(idxs), n_categories, dtype=torch.float32)
        hour_pref = torch.zeros(len(idxs), USER_HOUR_DIM, dtype=torch.float32)
        for bi, sample_index in enumerate(idxs):
            user = user_id_of(data, sample_index)
            cat_pref[bi] = torch.tensor(tables.user_cat_pref_vector(user), dtype=torch.float32)
            hour_pref[bi] = torch.tensor(tables.user_hour_pref_vector(user), dtype=torch.float32)
        out["user_cat_pref"] = cat_pref
        out["user_hour_pref"] = hour_pref

    # PMI is applied inside gather_batch_context when requested; nothing here.
    _ = transition_pmi
    return out


def merge_feature_context(count_ctx, side_ctx):
    """Merge side features into the count/feature context dict (or create one)."""
    if not side_ctx:
        return count_ctx
    if count_ctx is None:
        return dict(side_ctx)
    merged = dict(count_ctx)
    merged.update(side_ctx)
    return merged


def slice_side_ctx(ctx, start, end):
    """Slice candidate-aligned side features for a scoring chunk."""
    if ctx is None:
        return None
    out = {}
    for key, value in ctx.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if key in ("cand_cat_rows", "delta_geo", "cand_slot_feat",
                   "cand_poi_feat", "poi_transition", "cat_transition",
                   "poi_transition_dt", "cat_stage_count"):
            out[key] = value[:, start:end]
        else:
            out[key] = value
    return out
