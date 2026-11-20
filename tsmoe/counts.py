"""Causal online visit / transition count tables for TS-MoE.

Protocol (train, val, and test when enabled):
  1. Sort samples by prediction timestamp ascending (no shuffle).
  2. For each sample: **read** tables → build features / attention bias → score
     (and train: backward) → **update** tables with the ground-truth next POI
     and category (and the last-history → next transitions).

Tables only contain events observed as ground-truth next hops **before** the
current prediction. Val / test start from a **clone of the tables at end of
training** (or after replaying the train stream in order), then continue the
same score-then-update loop chronologically.

Visit-count bins for T/L are equal-frequency over the user's cumulative visit
count on the train stream; edges are frozen for val/test.

Optional extensions (no prepare change; used when CLI flags enable them):
  * user → hour visits (``--user-pref``)
  * POI × hour visits (``--candidate-context-time causal_slot``)
  * transition PMI (``--transition-pmi``)
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy

import torch

from .features import USER_HOUR_DIM, us_to_hour_weekday

# Default number of equal-frequency bins for user cumulative visit count → T/L.
DEFAULT_VISIT_BINS = 10

TRANSITION_ATTN_MODES = ("bias", "gate", "rel_key")

# Count feature φ dim: log1p(user), log1p(global), relative share, log-diff.
COUNT_FEAT_DIM = 4

# φ(transition) for gate / rel_key: log1p(count) only (1-D) unless expanded.
TRANSITION_PHI_DIM = 1


def prediction_timestamps(data):
    """UTC microseconds of each sample's true next-hop event."""
    return data["us"][data["end"]]


def chronological_indices(data):
    """Stable ascending order by prediction timestamp (no shuffle)."""
    return torch.argsort(prediction_timestamps(data), stable=True)


def user_id_of(data, sample_index):
    """User id string for a sample (from the last observed history event)."""
    start = int(data["start"][sample_index])
    end = int(data["end"][sample_index])
    # Supervised targets index ``end``; predict mode uses ``end`` one past history.
    tip = end - 1 if end > start else start
    tip = min(max(tip, 0), len(data["trail"]) - 1)
    trail = int(data["trail"][tip])
    return str(data["trails"][trail][0])


def sample_hop(data, sample_index):
    """Last-history and ground-truth next catalog rows / category rows.

    Returns ``(user, prev_poi, prev_cat, next_poi, next_cat)`` where POI/cat
    rows are catalog indices ``>= 0``, or ``-1`` when unknown / OOV / missing
    history. Tokens in ``data`` use pad=0, OOV=1, catalog=row+2.

    Only valid for supervised splits (``end`` indexes the true next event).
    """
    start = int(data["start"][sample_index])
    end = int(data["end"][sample_index])
    user = user_id_of(data, sample_index)
    next_poi = int(data["poi_token"][end]) - 2
    next_cat = int(data["cat_token"][end]) - 2
    if end - start < 1:
        return user, -1, -1, next_poi, next_cat
    prev_poi = int(data["poi_token"][end - 1]) - 2
    prev_cat = int(data["cat_token"][end - 1]) - 2
    return user, prev_poi, prev_cat, next_poi, next_cat


def count_features(user_count, global_count, eps=1e-6):
    """Scaled count features so user ≪ global remains informative.

    Returns a length-4 list:
    ``log1p(user), log1p(global), user/(user+global+eps), log1p(user)-log1p(global)``.
    """
    u = float(user_count)
    g = float(global_count)
    return [
        math_log1p(u),
        math_log1p(g),
        u / (u + g + eps),
        math_log1p(u) - math_log1p(g),
    ]


def math_log1p(x):
    return float(torch.log1p(torch.tensor(x, dtype=torch.float64)))


def equal_frequency_edges(values, n_bins):
    """Interior edges for ``torch.bucketize`` → bin ids in ``0 .. n_bins-1``.

    Duplicate quantiles are dropped so empty bins may collapse; callers still
    allocate ``n_bins`` embedding rows (unused rows stay idle).
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if not values:
        return torch.zeros(0, dtype=torch.float64)
    t = torch.as_tensor(values, dtype=torch.float64)
    if n_bins == 1 or t.numel() == 1:
        return torch.zeros(0, dtype=torch.float64)
    # Interior quantiles; bucketize with right=False → n_bins bins when edges unique.
    qs = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)[1:-1]
    edges = torch.quantile(t, qs)
    # Strictly increasing edges (bucketize requires sorted unique boundaries).
    uniq = edges.unique_consecutive()
    return uniq


def visit_bin_id(count, edges):
    """Map a cumulative visit count to a bin id in ``0 .. n_bins-1``."""
    if edges is None or edges.numel() == 0:
        return 0
    # bucketize: index of first edge > count → 0..len(edges); clamp to n_requested-1 later
    return int(torch.bucketize(torch.as_tensor(float(count), dtype=torch.float64),
                               edges.to(dtype=torch.float64)))


class CausalCountTables:
    """Online causal visit and transition counters.

    Indices are **catalog rows** (0 .. n-1), not raw venue ids. Unknown / OOV
    hops (row ``< 0``) are skipped on update and read as zero.
    """

    def __init__(self, n_pois, n_categories, *, visit_bins=DEFAULT_VISIT_BINS,
                 visit_bin_edges=None):
        if n_pois < 1 or n_categories < 1:
            raise ValueError("n_pois and n_categories must be positive")
        if visit_bins < 1:
            raise ValueError("visit_bins must be positive")
        self.n_pois = int(n_pois)
        self.n_categories = int(n_categories)
        self.visit_bins = int(visit_bins)
        self.visit_bin_edges = (None if visit_bin_edges is None
                                else torch.as_tensor(visit_bin_edges, dtype=torch.float64))
        self.reset()

    def reset(self):
        """Clear all counts (bin edges unchanged)."""
        self.user_poi = defaultdict(Counter)
        self.user_cat = defaultdict(Counter)
        self.user_hour = defaultdict(Counter)  # user → hour → count
        self.user_visits = Counter()
        self.global_poi = torch.zeros(self.n_pois, dtype=torch.float64)
        self.global_cat = torch.zeros(self.n_categories, dtype=torch.float64)
        self.global_hour = torch.zeros(USER_HOUR_DIM, dtype=torch.float64)
        # POI × hour (causal_slot): user and global.
        self.user_poi_hour = defaultdict(lambda: defaultdict(Counter))
        self.global_poi_hour = torch.zeros(self.n_pois, USER_HOUR_DIM, dtype=torch.float64)
        self.poi_poi = defaultdict(Counter)
        self.cat_cat = defaultdict(Counter)
        self.cat_poi = defaultdict(Counter)
        # Marginal transition totals for PMI (src out-count, dst in-count).
        self.poi_out = torch.zeros(self.n_pois, dtype=torch.float64)
        self.poi_in = torch.zeros(self.n_pois, dtype=torch.float64)
        self.cat_out = torch.zeros(self.n_categories, dtype=torch.float64)
        self.poi_poi_total = 0.0
        self.cat_poi_total = 0.0
        return self

    def clone(self):
        """Deep copy of counts + frozen bin edges (for val/test from train-end)."""
        other = CausalCountTables(
            self.n_pois, self.n_categories, visit_bins=self.visit_bins,
            visit_bin_edges=None if self.visit_bin_edges is None
            else self.visit_bin_edges.clone())
        other.user_poi = deepcopy(self.user_poi)
        other.user_cat = deepcopy(self.user_cat)
        other.user_hour = deepcopy(self.user_hour)
        other.user_visits = Counter(self.user_visits)
        other.global_poi = self.global_poi.clone()
        other.global_cat = self.global_cat.clone()
        other.global_hour = self.global_hour.clone()
        other.user_poi_hour = deepcopy(self.user_poi_hour)
        other.global_poi_hour = self.global_poi_hour.clone()
        other.poi_poi = deepcopy(self.poi_poi)
        other.cat_cat = deepcopy(self.cat_cat)
        other.cat_poi = deepcopy(self.cat_poi)
        other.poi_out = self.poi_out.clone()
        other.poi_in = self.poi_in.clone()
        other.cat_out = self.cat_out.clone()
        other.poi_poi_total = float(self.poi_poi_total)
        other.cat_poi_total = float(self.cat_poi_total)
        return other

    def state_dict(self):
        return dict(
            n_pois=self.n_pois,
            n_categories=self.n_categories,
            visit_bins=self.visit_bins,
            visit_bin_edges=None if self.visit_bin_edges is None
            else self.visit_bin_edges.tolist(),
            user_poi={u: dict(c) for u, c in self.user_poi.items()},
            user_cat={u: dict(c) for u, c in self.user_cat.items()},
            user_hour={u: dict(c) for u, c in self.user_hour.items()},
            user_visits=dict(self.user_visits),
            global_poi=self.global_poi.tolist(),
            global_cat=self.global_cat.tolist(),
            global_hour=self.global_hour.tolist(),
            user_poi_hour={
                u: {str(h): dict(c) for h, c in hours.items()}
                for u, hours in self.user_poi_hour.items()
            },
            global_poi_hour=self.global_poi_hour.tolist(),
            poi_poi={str(a): dict(b) for a, b in self.poi_poi.items()},
            cat_cat={str(a): dict(b) for a, b in self.cat_cat.items()},
            cat_poi={str(a): dict(b) for a, b in self.cat_poi.items()},
            poi_out=self.poi_out.tolist(),
            poi_in=self.poi_in.tolist(),
            cat_out=self.cat_out.tolist(),
            poi_poi_total=float(self.poi_poi_total),
            cat_poi_total=float(self.cat_poi_total),
        )

    def load_state_dict(self, state):
        self.n_pois = int(state["n_pois"])
        self.n_categories = int(state["n_categories"])
        self.visit_bins = int(state["visit_bins"])
        edges = state.get("visit_bin_edges")
        self.visit_bin_edges = (None if edges is None
                                else torch.as_tensor(edges, dtype=torch.float64))
        self.reset()
        for u, c in state["user_poi"].items():
            self.user_poi[u] = Counter({int(k): int(v) for k, v in c.items()})
        for u, c in state["user_cat"].items():
            self.user_cat[u] = Counter({int(k): int(v) for k, v in c.items()})
        for u, c in state.get("user_hour", {}).items():
            self.user_hour[u] = Counter({int(k): int(v) for k, v in c.items()})
        self.user_visits = Counter({u: int(v) for u, v in state["user_visits"].items()})
        self.global_poi = torch.as_tensor(state["global_poi"], dtype=torch.float64)
        self.global_cat = torch.as_tensor(state["global_cat"], dtype=torch.float64)
        if "global_hour" in state:
            self.global_hour = torch.as_tensor(state["global_hour"], dtype=torch.float64)
        for u, hours in state.get("user_poi_hour", {}).items():
            for h, c in hours.items():
                self.user_poi_hour[u][int(h)] = Counter(
                    {int(k): int(v) for k, v in c.items()})
        if "global_poi_hour" in state:
            self.global_poi_hour = torch.as_tensor(
                state["global_poi_hour"], dtype=torch.float64)
        for a, b in state["poi_poi"].items():
            self.poi_poi[int(a)] = Counter({int(k): int(v) for k, v in b.items()})
        for a, b in state["cat_cat"].items():
            self.cat_cat[int(a)] = Counter({int(k): int(v) for k, v in b.items()})
        for a, b in state["cat_poi"].items():
            self.cat_poi[int(a)] = Counter({int(k): int(v) for k, v in b.items()})
        if "poi_out" in state:
            self.poi_out = torch.as_tensor(state["poi_out"], dtype=torch.float64)
            self.poi_in = torch.as_tensor(state["poi_in"], dtype=torch.float64)
            self.cat_out = torch.as_tensor(state["cat_out"], dtype=torch.float64)
            self.poi_poi_total = float(state.get("poi_poi_total", 0.0))
            self.cat_poi_total = float(state.get("cat_poi_total", 0.0))
        else:
            # Rebuild marginals from transition maps (legacy checkpoints).
            for src, dsts in self.poi_poi.items():
                for dst, c in dsts.items():
                    self.poi_out[src] += c
                    self.poi_in[dst] += c
                    self.poi_poi_total += c
            for src, dsts in self.cat_poi.items():
                for dst, c in dsts.items():
                    self.cat_out[src] += c
                    self.cat_poi_total += c
        return self

    def _poi_ok(self, row):
        return 0 <= row < self.n_pois

    def _cat_ok(self, row):
        return 0 <= row < self.n_categories

    def update(self, user, prev_poi, prev_cat, next_poi, next_cat, *, hour=None):
        """Apply one ground-truth next hop (after scoring)."""
        user = str(user)
        if self._poi_ok(next_poi):
            self.user_poi[user][next_poi] += 1
            self.global_poi[next_poi] += 1
            self.user_visits[user] += 1
            if hour is not None:
                h = int(hour) % USER_HOUR_DIM
                self.user_hour[user][h] += 1
                self.global_hour[h] += 1
                self.user_poi_hour[user][h][next_poi] += 1
                self.global_poi_hour[next_poi, h] += 1
            if self._poi_ok(prev_poi):
                self.poi_poi[prev_poi][next_poi] += 1
                self.poi_out[prev_poi] += 1
                self.poi_in[next_poi] += 1
                self.poi_poi_total += 1
            if self._cat_ok(prev_cat):
                self.cat_poi[prev_cat][next_poi] += 1
                self.cat_out[prev_cat] += 1
                self.cat_poi_total += 1
        if self._cat_ok(next_cat):
            self.user_cat[user][next_cat] += 1
            self.global_cat[next_cat] += 1
            if self._cat_ok(prev_cat):
                self.cat_cat[prev_cat][next_cat] += 1

    def update_sample(self, data, sample_index):
        user, prev_poi, prev_cat, next_poi, next_cat = sample_hop(data, sample_index)
        end = int(data["end"][sample_index])
        hour, _weekday = us_to_hour_weekday(int(data["us"][end]))
        self.update(user, prev_poi, prev_cat, next_poi, next_cat, hour=hour)

    def replay(self, data, *, order=None):
        """Chronological score-less updates (rebuild train-end tables)."""
        idxs = chronological_indices(data) if order is None else order
        for i in idxs.tolist():
            self.update_sample(data, i)
        return self

    def fit_visit_bins(self, data, n_bins=None):
        """Replay train, fit equal-frequency edges on pre-update visit counts, reset."""
        n_bins = int(n_bins if n_bins is not None else self.visit_bins)
        self.visit_bins = n_bins
        self.reset()
        values = []
        for i in chronological_indices(data).tolist():
            user = user_id_of(data, i)
            values.append(int(self.user_visits[user]))
            self.update_sample(data, i)
        self.visit_bin_edges = equal_frequency_edges(values, n_bins)
        self.reset()
        return self.visit_bin_edges

    def user_visit_bin(self, user):
        count = int(self.user_visits[str(user)])
        bid = visit_bin_id(count, self.visit_bin_edges)
        return min(bid, self.visit_bins - 1)

    def poi_feat(self, user, poi_row):
        if not self._poi_ok(poi_row):
            return count_features(0, 0)
        u = self.user_poi[str(user)][poi_row]
        g = float(self.global_poi[poi_row])
        return count_features(u, g)

    def cat_feat(self, user, cat_row):
        if not self._cat_ok(cat_row):
            return count_features(0, 0)
        u = self.user_cat[str(user)][cat_row]
        g = float(self.global_cat[cat_row])
        return count_features(u, g)

    def poi_poi_count(self, src, dst):
        if not (self._poi_ok(src) and self._poi_ok(dst)):
            return 0.0
        return float(self.poi_poi[src][dst])

    def cat_poi_count(self, src_cat, dst_poi):
        if not (self._cat_ok(src_cat) and self._poi_ok(dst_poi)):
            return 0.0
        return float(self.cat_poi[src_cat][dst_poi])

    def poi_poi_pmi(self, src, dst, eps=1e-6):
        """Pointwise mutual information for POI→POI (0 when either side unknown)."""
        if not (self._poi_ok(src) and self._poi_ok(dst)):
            return 0.0
        joint = float(self.poi_poi[src][dst])
        if joint <= 0 or self.poi_poi_total <= 0:
            return 0.0
        src_m = float(self.poi_out[src]) + eps
        dst_m = float(self.poi_in[dst]) + eps
        return float(torch.log(torch.tensor(
            (joint * self.poi_poi_total) / (src_m * dst_m) + eps,
            dtype=torch.float64)))

    def cat_poi_pmi(self, src_cat, dst_poi, eps=1e-6):
        """PMI for category→POI transitions."""
        if not (self._cat_ok(src_cat) and self._poi_ok(dst_poi)):
            return 0.0
        joint = float(self.cat_poi[src_cat][dst_poi])
        if joint <= 0 or self.cat_poi_total <= 0:
            return 0.0
        src_m = float(self.cat_out[src_cat]) + eps
        # Destination marginal ≈ global POI visits as a proxy for cat→poi.
        dst_m = float(self.global_poi[dst_poi]) + eps
        return float(torch.log(torch.tensor(
            (joint * self.cat_poi_total) / (src_m * dst_m) + eps,
            dtype=torch.float64)))

    def poi_hour_feat(self, user, poi_row, hour):
        """Causal same-hour visit features for a candidate POI."""
        if not self._poi_ok(poi_row):
            return count_features(0, 0)
        h = int(hour) % USER_HOUR_DIM
        u = self.user_poi_hour[str(user)][h][poi_row]
        g = float(self.global_poi_hour[poi_row, h])
        return count_features(u, g)

    def user_cat_pref_vector(self, user):
        """L1-normalized log1p user→cat preference over all categories."""
        vec = [0.0] * self.n_categories
        counts = self.user_cat[str(user)]
        for cat, c in counts.items():
            if self._cat_ok(cat):
                vec[cat] = math_log1p(c)
        total = sum(vec)
        if total > 0:
            vec = [v / total for v in vec]
        return vec

    def user_hour_pref_vector(self, user):
        """L1-normalized log1p user→hour preference (length 24)."""
        vec = [0.0] * USER_HOUR_DIM
        counts = self.user_hour[str(user)]
        for hour, c in counts.items():
            vec[int(hour) % USER_HOUR_DIM] = math_log1p(c)
        total = sum(vec)
        if total > 0:
            vec = [v / total for v in vec]
        return vec

    def gather_sample_features(self, data, sample_index, history_poi_rows, history_cat_rows,
                               candidate_rows, *, max_length, transition_pmi=False):
        """Read-only feature pack for one sample at the current table state.

        ``history_*_rows`` / ``candidate_rows`` are catalog rows (``-1`` = pad/OOV).
        When ``transition_pmi`` is True, transition tensors store PMI (can be
        negative) instead of raw counts.
        """
        user = user_id_of(data, sample_index)
        length = len(history_poi_rows)
        hist_poi = [self.poi_feat(user, int(r)) for r in history_poi_rows]
        hist_cat = [self.cat_feat(user, int(r)) for r in history_cat_rows]
        # Pad to max_length for stacking.
        pad = count_features(0, 0)
        while len(hist_poi) < max_length:
            hist_poi.append(pad)
            hist_cat.append(pad)
        cand = [self.poi_feat(user, int(r)) for r in candidate_rows]
        visit_bin = self.user_visit_bin(user)
        # Transitions: [n_cand, max_length] — P-source uses poi→poi; C-source cat→poi.
        n_cand = len(candidate_rows)
        poi_trans = torch.zeros(n_cand, max_length, dtype=torch.float32)
        cat_trans = torch.zeros(n_cand, max_length, dtype=torch.float32)
        for j in range(length):
            hp, hc = int(history_poi_rows[j]), int(history_cat_rows[j])
            for c, row in enumerate(candidate_rows):
                dst = int(row)
                if transition_pmi:
                    poi_trans[c, j] = self.poi_poi_pmi(hp, dst)
                    cat_trans[c, j] = self.cat_poi_pmi(hc, dst)
                else:
                    poi_trans[c, j] = self.poi_poi_count(hp, dst)
                    cat_trans[c, j] = self.cat_poi_count(hc, dst)
        return dict(
            hist_poi_feat=torch.tensor(hist_poi[:max_length], dtype=torch.float32),
            hist_cat_feat=torch.tensor(hist_cat[:max_length], dtype=torch.float32),
            cand_poi_feat=torch.tensor(cand, dtype=torch.float32),
            visit_bin=visit_bin,
            poi_transition=poi_trans,
            cat_transition=cat_trans,
        )


def history_catalog_rows(data, sample_index):
    """Catalog POI/cat rows for the observed history prefix (no pad)."""
    start = int(data["start"][sample_index])
    end = int(data["end"][sample_index])
    poi = (data["poi_token"][start:end] - 2).tolist()
    cat = (data["cat_token"][start:end] - 2).tolist()
    return poi, cat


def gather_batch_context(tables, data, indices, candidate_rows_list, max_length,
                         *, update=True, transition_pmi=False):
    """Causal read (then optional update) for a chronological batch.

    ``candidate_rows_list[i]`` is a 1-D LongTensor / list of catalog rows for
    sample ``indices[i]`` (padded candidates should be excluded by the caller
    or passed as ``-1``).

    When ``update=True``, each sample updates the tables with its GT next hop
    **after** its features are read (strict causal online protocol).
    """
    hist_poi, hist_cat, cand_feat = [], [], []
    visit_bins, poi_tr, cat_tr = [], [], []
    widths = [len(c) for c in candidate_rows_list]
    width = max(widths) if widths else 0
    for local, sample_index in enumerate(indices.tolist()):
        hp, hc = history_catalog_rows(data, sample_index)
        rows = candidate_rows_list[local]
        if not isinstance(rows, list):
            rows = list(rows) if not torch.is_tensor(rows) else rows.tolist()
        # Pad candidate feature rows to batch width with -1 (zero feats / transitions).
        padded = list(rows) + [-1] * (width - len(rows))
        feat = tables.gather_sample_features(
            data, sample_index, hp, hc, padded, max_length=max_length,
            transition_pmi=transition_pmi)
        hist_poi.append(feat["hist_poi_feat"])
        hist_cat.append(feat["hist_cat_feat"])
        cand_feat.append(feat["cand_poi_feat"])
        visit_bins.append(feat["visit_bin"])
        poi_tr.append(feat["poi_transition"])
        cat_tr.append(feat["cat_transition"])
        if update:
            tables.update_sample(data, sample_index)
    if not hist_poi:
        empty = dict(
            hist_poi_feat=torch.zeros(0, max_length, COUNT_FEAT_DIM),
            hist_cat_feat=torch.zeros(0, max_length, COUNT_FEAT_DIM),
            cand_poi_feat=torch.zeros(0, width, COUNT_FEAT_DIM),
            visit_bin=torch.zeros(0, dtype=torch.long),
            poi_transition=torch.zeros(0, width, max_length),
            cat_transition=torch.zeros(0, width, max_length),
        )
        return empty
    return dict(
        hist_poi_feat=torch.stack(hist_poi, 0),
        hist_cat_feat=torch.stack(hist_cat, 0),
        cand_poi_feat=torch.stack(cand_feat, 0),
        visit_bin=torch.tensor(visit_bins, dtype=torch.long),
        poi_transition=torch.stack(poi_tr, 0),
        cat_transition=torch.stack(cat_tr, 0),
    )
