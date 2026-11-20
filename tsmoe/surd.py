"""Offline four-source SURD, followed by frozen PyTorch lookup.

Input q_states has one (P,C,T,L) quantized history per real prediction event.
target_rows indexes poi_ids, not the numerical magnitude of an external POI ID.
Fixed per-source feature extraction and quantization happen BEFORE this module.
All 15 marginals derive from one sparse five-way joint count table.

This is a CPU/float64 statistical reference, not a differentiable estimator.
Exact ties use fixed source/subset order; near ties are reported, never averaged.
Run this file to execute the seven analytical examples and persistence checks.
"""

from itertools import combinations, product
from pathlib import Path
import json
import math
import tempfile

import torch
from torch import nn
from tqdm.auto import tqdm

from .runtime import check_interrupt


SOURCE_ORDER = ("P", "C", "T", "L")
SUBSETS = tuple(a for k in range(1, 5) for a in combinations(range(4), k))
MULTIS = SUBSETS[4:]
ATOMS = (
    tuple(("U", a) for a in SUBSETS[:4])
    + tuple(("R", a) for a in MULTIS)
    + tuple(("S", a) for a in MULTIS)
)
SUBSET_ORDER = tuple("".join(SOURCE_ORDER[j] for j in a) for a in SUBSETS)
ATOM_ORDER = tuple(f"{kind}_{''.join(SOURCE_ORDER[j] for j in a)}"
                   for kind, a in ATOMS)
EXPERT_ORDER = ATOM_ORDER[:4] + ATOM_ORDER[15:]
ATOM_INDEX = {key: j for j, key in enumerate(ATOMS)}
DTYPE = torch.float64
INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _long_cpu(value, name):
    tensor = torch.as_tensor(value)
    if tensor.dtype not in INTEGER_DTYPES:
        raise TypeError(f"{name} must contain integer IDs, not floats/bools")
    return tensor.detach().to(device="cpu", dtype=torch.long)


def _sum_by_index(index, value, size):
    return torch.zeros(size, dtype=DTYPE).scatter_add_(0, index, value)


@torch.no_grad()
def surd_event_torch(specific, atol=1e-10, rtol=1e-10):
    """Allocate one target event's 15 specific-information values to 26 atoms."""
    values = torch.as_tensor(specific, dtype=DTYPE, device="cpu").clone()
    if values.shape != (15,) or not torch.isfinite(values).all():
        raise ValueError("specific must be 15 finite values")
    tol = atol + rtol * max(float(values.abs().max()), 1.0)
    if float(values.min()) < -tol:
        raise ValueError("materially negative specific information")
    values.clamp_(min=0.0)  # Only roundoff, after validation.
    info = dict(zip(SUBSETS, values.tolist()))
    for a in SUBSETS:
        for b in SUBSETS:
            if set(a) < set(b) and info[a] > info[b] + tol:
                raise ValueError("subset monotonicity failed")

    atom = torch.zeros(26, dtype=DTYPE)
    order = sorted(range(4), key=lambda m: (info[(m,)], m))
    previous = 0.0
    for rank, m in enumerate(order):
        key = ("U", (m,)) if rank == 3 else ("R", tuple(sorted(order[rank:])))
        atom[ATOM_INDEX[key]] += info[(m,)] - previous
        previous = info[(m,)]
    for degree in range(2, 5):
        baseline = max(info[a] for a in SUBSETS if len(a) == degree - 1)
        group = sorted((a for a in SUBSETS if len(a) == degree),
                       key=lambda a: (info[a], a))
        previous = 0.0
        for a in group:
            atom[ATOM_INDEX[("S", a)]] = max(0.0, info[a] - max(baseline, previous))
            previous = info[a]
        total_degree = sum(float(atom[ATOM_INDEX[("S", a)]]) for a in group)
        expected_degree = max(info[a] for a in group) - baseline
        if abs(total_degree - expected_degree) > tol:
            raise ArithmeticError("synergy-by-degree conservation failed")

    if float(atom.min()) < -tol or abs(float(atom.sum()) - info[SUBSETS[-1]]) > tol:
        raise ArithmeticError("atom nonnegativity/total conservation failed")
    for m in range(4):
        rebuilt = float(atom[ATOM_INDEX[("U", (m,))]]) + math.fsum(
            float(atom[ATOM_INDEX[("R", a)]]) for a in MULTIS if m in a)
        if abs(rebuilt - info[(m,)]) > tol:
            raise ArithmeticError("single-source reconstruction failed")
    return atom


@torch.no_grad()
def build_offline_surd(q_states, target_rows, poi_ids, *, sample_weight=None,
                       weight_semantics="event_count", kappa_n=20.0, kappa_i=0.05,
                       atol=1e-10, rtol=1e-10, manifest=None):
    """Build and validate an offline table.

    q_states: [n,4] integer discrete states in P,C,T,L order, each >=0.
    target_rows: [n] integers in [0,V); poi_ids fixes the shared catalog row order.
    Default: one row = one distinct real target event, weight 1.
    Explicit event_count weights may aggregate identical events and must be
    nonnegative integers; duplicated augmented windows must already be deduped.
    probability_mass weights are allowed only for analytic examples: sample-size
    shrinkage is disabled and confidence is ZERO, because mass is not a count.
    """
    q = _long_cpu(q_states, "q_states")
    y = _long_cpu(target_rows, "target_rows")
    poi_ids = list(poi_ids)
    if (not poi_ids or any(type(v) not in (str, int) for v in poi_ids)
            or len(set(poi_ids)) != len(poi_ids)):
        raise ValueError("poi_ids must be a nonempty list of unique int/string IDs")
    V = len(poi_ids)
    if q.ndim != 2 or q.shape[1] != 4 or y.shape != (q.shape[0],) or not len(y):
        raise ValueError("expected q_states[n,4] and target_rows[n], n>0")
    if (q < 0).any() or (y < 0).any() or (y >= V).any():
        raise ValueError("source states must be >=0 and targets must index poi_ids")
    if kappa_n <= 0 or kappa_i <= 0:
        raise ValueError("reliability scales must be positive")
    if weight_semantics not in ("event_count", "probability_mass"):
        raise ValueError("unknown weight semantics")
    w = (torch.ones(len(y), dtype=DTYPE) if sample_weight is None else
         torch.as_tensor(sample_weight, dtype=DTYPE, device="cpu").detach())
    if w.shape != y.shape or not torch.isfinite(w).all() or (w < 0).any():
        raise ValueError("sample_weight must be finite, nonnegative and length n")
    if weight_semantics == "event_count" and not torch.equal(w, w.round()):
        raise ValueError("event_count weights must be integer counts")
    keep = w > 0
    q, y, w = q[keep], y[keep], w[keep]
    if not len(y):
        raise ValueError("positive total weight required")
    safe_manifest = json.loads(json.dumps(manifest or {}, allow_nan=False))

    # First construct exactly ONE sparse joint table (Y,Q_P,Q_C,Q_T,Q_L).
    joint, inverse = torch.unique(torch.cat([y[:, None], q], dim=1),
                                  dim=0, sorted=True, return_inverse=True)
    joint_count = _sum_by_index(inverse, w, len(joint))
    total_weight = joint_count.sum()
    if not torch.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("positive finite total weight required")
    target_count = _sum_by_index(joint[:, 0], joint_count, V)
    target_probability = target_count / total_weight
    observed = target_count > 0
    specific = torch.zeros(V, 15, dtype=DTYPE)

    # Every subset marginal is projected/aggregated from that same joint table.
    for column, subset in enumerate(SUBSETS):
        columns = [0] + [1 + j for j in subset]
        marginal, inv = torch.unique(joint[:, columns], dim=0,
                                     sorted=True, return_inverse=True)
        count_yq = _sum_by_index(inv, joint_count, len(marginal))
        states, source_inv = torch.unique(marginal[:, 1:], dim=0,
                                          sorted=True, return_inverse=True)
        count_q = _sum_by_index(source_inv, count_yq, len(states))
        row = marginal[:, 0]
        count_y = target_count[row]
        log_ratio = (count_yq.log2() + total_weight.log2()
                     - count_y.log2() - count_q[source_inv].log2())
        specific[:, column].scatter_add_(0, row, (count_yq / count_y) * log_ratio)

    raw = torch.zeros(V, 26, dtype=DTYPE)
    near_tie = torch.zeros(V, dtype=torch.bool)
    zero_tol = atol + rtol * specific.abs().amax(dim=1).clamp_min(1.0)
    observed_rows = torch.nonzero(observed, as_tuple=False).flatten().tolist()
    for row in tqdm(observed_rows, desc="SURD per POI", leave=False, dynamic_ncols=True):
        check_interrupt()
        raw[row] = surd_event_torch(specific[row], atol, rtol)
        if specific[row].max() > -target_probability[row].log2() + zero_tol[row]:
            raise ArithmeticError("event-information upper bound failed")
        for degree in range(1, 5):
            columns = [j for j, a in enumerate(SUBSETS) if len(a) == degree]
            ordered = specific[row, columns].sort().values
            gap = ordered[1:] - ordered[:-1]
            near_tie[row] |= ((gap > 0) & (gap <= zero_tol[row])).any()
    total_information = raw.sum(dim=1)
    has_information = observed & (total_information > zero_tol)
    normalized = torch.zeros_like(raw)
    normalized[has_information] = (
        raw[has_information] / total_information[has_information, None])
    global_raw = target_probability @ raw
    global_mi = target_probability @ specific
    global_total = global_raw.sum()
    global_tol = atol + rtol * max(float(global_raw.abs().max()), 1.0)
    global_prior = (global_raw / global_total if global_total > global_tol
                    else torch.zeros(26, dtype=DTYPE))

    if weight_semantics == "event_count":
        count_factor = target_count / (target_count + kappa_n)
    else:
        count_factor = observed.to(DTYPE)  # No arbitrary probability-mass shrinkage.
    routing_prior = global_prior[None].expand(V, -1).clone()
    if global_total > global_tol:
        c = count_factor[has_information, None]
        routing_prior[has_information] = (
            c * normalized[has_information] + (1 - c) * global_prior)
    else:
        routing_prior[has_information] = normalized[has_information]
    confidence = (has_information.to(DTYPE) * count_factor
                  * total_information / (total_information + kappa_i))
    if weight_semantics == "probability_mass":
        confidence.zero_()

    # Estimate H(Y) and H(Y|Q_P,Q_C,Q_T,Q_L) directly from the same counts.
    py = target_probability[observed]
    H_Y = -(py * py.log2()).sum()
    full_states, full_inv = torch.unique(joint[:, 1:], dim=0,
                                         sorted=True, return_inverse=True)
    count_full_q = _sum_by_index(full_inv, joint_count, len(full_states))
    H_cond = -((joint_count / total_weight)
               * (joint_count.log2() - count_full_q[full_inv].log2())).sum()
    H_cond_via_mi = H_Y - target_probability @ total_information
    torch.testing.assert_close(H_cond, H_cond_via_mi, atol=atol, rtol=rtol)
    torch.testing.assert_close(global_raw.sum(), global_mi[-1], atol=atol, rtol=rtol)
    nonzero = routing_prior.sum(dim=1) > 0
    torch.testing.assert_close(routing_prior[nonzero].sum(dim=1),
                               torch.ones(int(nonzero.sum()), dtype=DTYPE),
                               atol=atol, rtol=rtol)
    if routing_prior.min() < -atol:
        raise ArithmeticError("negative routing prior")
    return {
        "schema_version": "PCTL_TORCH_SURD_v2",
        "source_order": list(SOURCE_ORDER), "subset_order": list(SUBSET_ORDER),
        "atom_order": list(ATOM_ORDER), "expert_order": list(EXPERT_ORDER),
        "poi_ids": poi_ids, "specific_information": specific, "raw_surd": raw,
        "normalized_surd": normalized, "routing_prior": routing_prior,
        "total_information": total_information, "target_count": target_count,
        "target_probability": target_probability, "observed": observed,
        "has_information": has_information, "near_tie": near_tie,
        "confidence": confidence, "count_factor": count_factor,
        "global_raw": global_raw, "global_prior": global_prior, "global_mi": global_mi,
        "H_Y": H_Y, "H_cond": H_cond, "H_cond_via_mi": H_cond_via_mi,
        "joint_states": joint, "joint_count": joint_count,
        "total_weight": total_weight,
        "config": {"kappa_n": kappa_n, "kappa_i": kappa_i,
                   "atol": atol, "rtol": rtol, "weight_semantics": weight_semantics},
        "manifest": safe_manifest,
    }


def save_surd_table(stats, path):
    """Save primitive metadata and detached CPU tensors; raw remains float64."""
    payload = {key: value.detach().cpu() if torch.is_tensor(value) else value
               for key, value in stats.items()}
    torch.save(payload, Path(path))


class FrozenSURDTable(nn.Module):
    """Runtime candidate-row lookup; -1 is unknown, other invalid IDs raise.

    Train / val / test share one train-all table (prepare writes `final.surd.pt`).
    Candidate padding still requires a separate candidate-valid mask.
    """

    def __init__(self, stats, dtype=torch.float32):
        super().__init__()
        for name, expected in (("source_order", SOURCE_ORDER),
                               ("subset_order", SUBSET_ORDER),
                               ("atom_order", ATOM_ORDER),
                               ("expert_order", EXPERT_ORDER)):
            if tuple(stats[name]) != expected:
                raise ValueError(f"incompatible {name}")
        if stats["schema_version"] != "PCTL_TORCH_SURD_v2":
            raise ValueError("unexpected schema version")
        self.poi_ids = list(stats["poi_ids"])
        self.manifest = stats["manifest"]
        self.row_of = {poi: row for row, poi in enumerate(self.poi_ids)}
        V = len(self.poi_ids)
        if stats["routing_prior"].shape != (V, 26) or stats["confidence"].shape != (V,):
            raise ValueError("invalid prior/confidence shape")
        if stats["specific_information"].shape != (V, 15):
            raise ValueError("invalid specific_information shape")
        for name in ("routing_prior", "confidence", "global_prior", "raw_surd",
                     "normalized_surd", "total_information", "target_count",
                     "specific_information", "global_mi"):
            self.register_buffer(name, stats[name].detach().to(dtype=dtype).clone())
        for name in ("observed", "has_information", "near_tie"):
            self.register_buffer(name, stats[name].detach().to(torch.bool).clone())

    @classmethod
    def from_file(cls, path, dtype=torch.float32):
        stats = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(stats, dtype=dtype)

    def _safe_rows(self, candidate_rows):
        if candidate_rows.dtype not in INTEGER_DTYPES:
            raise TypeError("candidate_rows must be integer catalog row indices")
        rows = candidate_rows.to(device=self.routing_prior.device, dtype=torch.long)
        if ((rows < -1) | (rows >= len(self.poi_ids))).any():
            raise IndexError("only -1 or valid catalog rows are allowed")
        unknown = rows == -1
        safe_rows = rows.clamp_min(0)  # -1 must never silently index the last row.
        return rows, unknown, safe_rows

    def forward(self, candidate_rows):
        rows, unknown, safe_rows = self._safe_rows(candidate_rows)
        prior = self.routing_prior[safe_rows]
        weight = self.confidence[safe_rows]
        prior = torch.where(unknown[..., None], self.global_prior, prior)
        weight = torch.where(unknown, torch.zeros_like(weight), weight)
        return prior, weight

    def specific(self, candidate_rows):
        """Per-candidate 15-D Specific MI (SUBSET_ORDER); unknown → global_mi."""
        _, unknown, safe_rows = self._safe_rows(candidate_rows)
        smi = self.specific_information[safe_rows]
        return torch.where(unknown[..., None], self.global_mi, smi)


def gather_fold_priors(tables, fold_ids, candidate_rows):
    """Batch rows may come from different folds; each gets its own table."""
    if not len(tables):
        raise ValueError("at least one fold table is required")
    if fold_ids.dtype not in INTEGER_DTYPES or candidate_rows.ndim != 2:
        raise TypeError("expected integer fold_ids[B] and candidate_rows[B,M]")
    if fold_ids.shape != (candidate_rows.shape[0],):
        raise ValueError("fold_ids must have one entry per batch sample")
    if any(table.poi_ids != tables[0].poi_ids for table in tables):
        raise ValueError("all fold tables must share the candidate catalog order")
    device = tables[0].routing_prior.device
    folds = fold_ids.to(device=device, dtype=torch.long)
    rows = candidate_rows.to(device)
    if ((folds < 0) | (folds >= len(tables))).any():
        raise IndexError("invalid fold index")
    prior = tables[0].routing_prior.new_empty(*rows.shape, 26)
    confidence = tables[0].confidence.new_empty(rows.shape)
    for fold in folds.unique().tolist():
        selected = folds == fold
        prior[selected], confidence[selected] = tables[fold](rows[selected])
    return prior, confidence


def _self_test():
    def expect(joint_dict, expected, name, known=(0, 1), probability=False):
        keys, weights = zip(*joint_dict.items())
        catalog = list(known) + [99]
        row_of = {poi: row for row, poi in enumerate(catalog)}
        stats = build_offline_surd(
            torch.tensor([key[1:] for key in keys], dtype=torch.long),
            torch.tensor([row_of[key[0]] for key in keys], dtype=torch.long),
            catalog, sample_weight=weights,
            weight_semantics="probability_mass" if probability else "event_count")
        expected_raw = torch.tensor([expected.get(a, 0) for a in ATOM_ORDER],
                                    dtype=DTYPE).expand(len(known), -1)
        torch.testing.assert_close(stats["raw_surd"][:-1], expected_raw, atol=1e-12, rtol=1e-12)
        assert not stats["observed"][-1] and not stats["raw_surd"][-1].any()
        active = {key: round(float(value), 10) for key, value in
                  zip(ATOM_ORDER, stats["raw_surd"][0]) if value > 1e-12}
        print(f"{name}: atoms={active}; H(Y)={float(stats['H_Y']):.10f}; "
              f"H(Y|Q)={float(stats['H_cond']):.10f}")
        return stats

    copies = expect({(y, y, y, y, y): 1 for y in (0, 1)}, {"R_PCTL": 1}, "copies")
    unique = expect({(p, p, c, t, l): 1 for p, c, t, l in product((0, 1), repeat=4)},
                    {"U_P": 1}, "P only")
    expect({(p ^ c, p, c, t, l): 1 for p, c, t, l in product((0, 1), repeat=4)},
           {"S_PC": 1}, "P-C XOR")
    expect({(p ^ c ^ t ^ l, p, c, t, l): 1
            for p, c, t, l in product((0, 1), repeat=4)}, {"S_PCTL": 1}, "four-way XOR")
    noise = expect({(y, p, c, t, l): 1 for y, p, c, t, l in product((0, 1), repeat=5)},
                   {}, "independent noise")
    assert not noise["has_information"].any() and not noise["confidence"].any()

    reveal = (.4, .3, .1, .2)
    erasure = {}
    for y in (0, 1):
        for visible in product((0, 1), repeat=4):
            key = (y,) + tuple(y if flag else 2 for flag in visible)
            erasure[key] = .5 * math.prod(p if flag else 1 - p
                                         for p, flag in zip(reveal, visible))
    erased = expect(erasure,
                    {"U_P": .1, "R_PC": .1, "R_PCL": .1, "R_PCTL": .1,
                     "S_PC": .06, "S_PT": .02, "S_PL": .06, "S_CL": .04,
                     "S_PCT": .042, "S_PCL": .042, "S_PCTL": .0336},
                    "independent erasures", probability=True)
    torch.testing.assert_close(erased["H_cond"], torch.tensor(.3024, dtype=DTYPE),
                               atol=1e-12, rtol=1e-12)
    assert not erased["confidence"].any()  # Probability mass is not a sample count.
    for column, subset in enumerate(SUBSETS):
        exact = 1 - math.prod(1 - reveal[m] for m in subset)
        torch.testing.assert_close(erased["specific_information"][:2, column],
                                   torch.full((2,), exact, dtype=DTYPE), atol=1e-12, rtol=1e-12)
    expect({(p, p, p % 2, t, p // 2): 1 for p in range(4) for t in (0, 1)},
           {"U_P": 1, "R_PCL": 1}, "deterministic C/L", known=range(4))

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "surd.pt"
        save_surd_table(unique, path)
        table = FrozenSURDTable.from_file(path)
        prior, confidence = table(torch.tensor([[0, 1, 2, -1]]))
        assert prior.shape == (1, 4, 26) and confidence.shape == (1, 4)
        assert not prior.requires_grad and not confidence.requires_grad
        assert not list(table.parameters())
        assert confidence[0, 2] == 0 and confidence[0, 3] == 0
        torch.testing.assert_close(prior[0, -1], table.global_prior)
        torch.testing.assert_close(prior[0, 2], table.global_prior)
        for bad in (-2, len(table.poi_ids)):
            try:
                table(torch.tensor([[bad]]))
            except IndexError:
                pass
            else:
                raise AssertionError("invalid candidate row was not rejected")
        tables = nn.ModuleList([FrozenSURDTable(unique), FrozenSURDTable(copies)])
        folds = torch.tensor([0, 1])
        rows = torch.tensor([[0, -1], [1, 2]])
        mixed_prior, mixed_weight = gather_fold_priors(tables, folds, rows)
        assert mixed_prior.shape == (2, 2, 26) and mixed_weight.shape == (2, 2)
        for batch_row, fold in enumerate(folds.tolist()):
            p, w = tables[fold](rows[batch_row])
            torch.testing.assert_close(mixed_prior[batch_row], p)
            torch.testing.assert_close(mixed_weight[batch_row], w)
    print("All seven analytic cases, entropy identities, save/load and lookup checks passed.")
    print(f"torch={torch.__version__}; 15 marginals; 26 stored atoms; 15 expert coordinates.")


if __name__ == "__main__":
    _self_test()
