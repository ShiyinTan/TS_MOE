"""Basic, graph-free P/C/T/L history summaries and offline quantization.

P/C: integer [B,N], PAD=-1 only in padding, UNK=-2 at missing real events.
T/L: finite numeric [B,N,d] on real events; upstream implements Section 11's
time/location schema, including finite fills and validity flags (default 6/8).
valid_mask: shared bool [B,N], right-padded and nonempty for every history.
No user ID, target label, trainable embedding, or graph enters this module.
"""
import hashlib
import math
from pathlib import Path

import torch

SOURCES = ("P", "C", "T", "L")
INT_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
SCHEMA = "BASIC_SOURCE_QUANTIZER_v1"


class BasicSourceQuantizer:
    """Fit once on permitted reference histories; freeze for all later use."""

    def __init__(self, n_clusters=4, hash_dim=64, max_length=50, seed=0,
                 tau_pos=10.0, max_iter=100, tol=1e-6):
        for name, value in (("n_clusters", n_clusters), ("hash_dim", hash_dim),
                            ("max_length", max_length), ("max_iter", max_iter)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63 - 4:
            raise ValueError("seed must be an integer in [0, 2**63-4)")
        if not math.isfinite(tau_pos) or tau_pos <= 0 or not math.isfinite(tol) or tol < 0:
            raise ValueError("tau_pos must be positive; tol must be nonnegative and finite")
        self._params = dict(n_clusters=n_clusters, hash_dim=hash_dim,
                            max_length=max_length, seed=seed, tau_pos=float(tau_pos),
                            max_iter=max_iter, tol=float(tol))
        self._salts = {s: f"{s}/basic_fixed_features_v1" for s in SOURCES[:2]}
        self._state = {}
        self._input_dims = {}

    @property
    def effective_clusters(self):
        return {s: int(v["centers"].shape[0]) for s, v in self._state.items()}

    @property
    def config(self):
        """Serializable manifest; persist with the SURD table for reproducibility."""
        return {**self._params, "feature_method": "basic_fixed_features_v1",
                "hash_algorithm": "SHAKE-256: byte parity -> +/-1/sqrt(hash_dim)",
                "source_salts": dict(self._salts), "source_order": list(SOURCES),
                "pad_id": -1, "unk_id": -2, "standardize_epsilon": 1e-12,
                "source_input_dims": dict(self._input_dims),
                "summary_dims": {s: 3 * d + 1 for s, d in self._input_dims.items()},
                "effective_clusters": self.effective_clusters}

    def _checked(self, history, valid_mask):
        if set(history) != set(SOURCES):
            raise ValueError("history must have exactly P, C, T, L keys")
        mask = torch.as_tensor(valid_mask).detach().cpu()
        if mask.dtype != torch.bool or mask.ndim != 2 or min(mask.shape) == 0:
            raise ValueError("valid_mask must be a nonempty bool [B,N] tensor")
        if mask.shape[1] > self._params["max_length"]:
            raise ValueError("history exceeds configured max_length")
        if not mask.any(dim=1).all() or ((~mask[:, :-1]) & mask[:, 1:]).any():
            raise ValueError("each history must be nonempty and right-padded")
        values = {}
        for source in SOURCES:
            x = torch.as_tensor(history[source]).detach().cpu()
            if x.shape[:2] != mask.shape:
                raise ValueError(f"{source} must share valid_mask's [B,N]")
            if source in SOURCES[:2]:
                if x.ndim != 2 or x.dtype not in INT_DTYPES:
                    raise TypeError(f"{source} must be integer [B,N]")
                x = x.to(torch.long)
                if ((x[mask] < 0) & (x[mask] != -2)).any() or (x[~mask] != -1).any():
                    raise ValueError(f"{source}: real IDs >=0 or UNK=-2; padding must be PAD=-1")
            else:
                if x.ndim != 3 or x.shape[2] == 0 or x.is_complex() or x.dtype == torch.bool:
                    raise TypeError(f"{source} must be numeric [B,N,d], d>0")
                x = x.to(torch.float64)
                if not torch.isfinite(x[mask]).all():
                    raise ValueError(f"{source}: encode real-event missingness with finite fills and flags")
                x = torch.where(mask[..., None], x, 0.0)
            values[source] = x
        return values, mask

    def _hashed(self, ids, mask, source):
        unique, inverse = torch.unique(ids[mask], sorted=True, return_inverse=True)
        dim, seed = self._params["hash_dim"], self._params["seed"]
        rows = []
        for identity in unique.tolist():
            key = f"{seed}|{self._salts[source]}|{identity}".encode("utf-8")
            raw = hashlib.shake_256(key).digest(dim)
            rows.append([(1.0 if byte & 1 else -1.0) / math.sqrt(dim) for byte in raw])
        out = torch.zeros(*ids.shape, dim, dtype=torch.float64)
        out[mask] = torch.tensor(rows, dtype=torch.float64)[inverse]
        return out

    @torch.no_grad()
    def summarize(self, history, valid_mask):
        """Source-isolated deterministic [last, decayed mean, mean, length ratio]."""
        values, mask = self._checked(history, valid_mask)
        length = mask.sum(dim=1)
        position = torch.arange(mask.shape[1], dtype=torch.float64)[None, :]
        age = (length[:, None] - 1 - position).clamp_min(0)
        weight = torch.exp(-age / self._params["tau_pos"]) * mask
        weight /= weight.sum(dim=1, keepdim=True)
        ratio = length[:, None].to(torch.float64) / self._params["max_length"]
        summaries = {}
        for source in SOURCES:
            x = self._hashed(values[source], mask, source) if source in SOURCES[:2] else values[source]
            last = x[torch.arange(mask.shape[0]), length - 1]
            decay = (x * weight[..., None]).sum(dim=1)
            mean = x.sum(dim=1) / length[:, None]
            out = torch.cat((last, decay, mean, ratio), dim=1)
            if not torch.isfinite(out).all():
                raise ValueError(f"{source}: summary overflow; check numeric input units")
            summaries[source] = out
        return summaries

    def _kmeans(self, x, source_index):
        unique = torch.unique(x, dim=0, sorted=True)
        k = min(self._params["n_clusters"], unique.shape[0])
        generator = torch.Generator().manual_seed(self._params["seed"] + source_index)
        centers = unique[torch.randperm(unique.shape[0], generator=generator)[:k]].clone()
        for _ in range(self._params["max_iter"]):
            labels = torch.cdist(x, centers).argmin(dim=1)
            new = centers.clone()
            for j in range(k):
                selected = labels == j
                if selected.any():
                    new[j] = x[selected].mean(dim=0)
            shift = (new - centers).abs().max().item()
            centers = new
            if shift <= self._params["tol"]:
                break
        # Remove unused clusters; no special UNK cluster is invented.
        occupied = torch.unique(torch.cdist(x, centers).argmin(dim=1), sorted=True)
        return centers[occupied]

    @torch.no_grad()
    def fit(self, history, valid_mask):
        """No labels accepted. Fit only on histories permitted by the outer split."""
        return self.fit_summaries(self.summarize(history, valid_mask))

    @torch.no_grad()
    def fit_summaries(self, summaries):
        """Fit on rows selected from batched, label-independent summaries."""
        state, dims = {}, {}
        for index, source in enumerate(SOURCES):
            x = summaries[source]
            mean, scale = x.mean(dim=0), x.std(dim=0, correction=0)
            scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
            standardized = (x - mean) / scale
            if not torch.isfinite(standardized).all():
                raise ValueError(f"{source}: invalid standardized values")
            state[source] = {"mean": mean, "scale": scale,
                             "centers": self._kmeans(standardized, index)}
            dims[source] = (x.shape[1] - 1) // 3
        self._state, self._input_dims = state, dims
        return self

    @torch.no_grad()
    def transform(self, history, valid_mask):
        """Return CPU long [B,4], ordered P,C,T,L; each column starts at 0."""
        if set(self._state) != set(SOURCES):
            raise RuntimeError("fit or load the quantizer before transform")
        return self.transform_summaries(self.summarize(history, valid_mask))

    @torch.no_grad()
    def transform_summaries(self, summaries):
        states = []
        for source in SOURCES:
            x, fitted = summaries[source], self._state[source]
            if x.shape[1] != fitted["mean"].numel():
                raise ValueError(f"{source}: feature dimension differs from the fitted schema")
            z = (x - fitted["mean"]) / fitted["scale"]
            if not torch.isfinite(z).all():
                raise ValueError(f"{source}: standardized values overflow")
            states.append(torch.cdist(z, fitted["centers"]).argmin(dim=1))
        return torch.stack(states, dim=1).to(torch.long)

    def save(self, path):
        if set(self._state) != set(SOURCES):
            raise RuntimeError("cannot save an unfitted quantizer")
        torch.save({"schema_version": SCHEMA, "constructor": self._params,
                    "config": self.config, "state": self._state}, Path(path))

    @classmethod
    def load(cls, path):
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if payload["schema_version"] != SCHEMA:
            raise ValueError("unsupported quantizer schema")
        obj = cls(**payload["constructor"])
        obj._state = payload["state"]
        obj._input_dims = payload["config"]["source_input_dims"]
        if set(obj._state) != set(SOURCES) or obj.config != payload["config"]:
            raise ValueError("quantizer manifest mismatch")
        for source, state in obj._state.items():
            d = 3 * obj._input_dims[source] + 1
            if (state["mean"].shape != (d,) or state["scale"].shape != (d,)
                    or state["centers"].shape != (obj.effective_clusters[source], d)
                    or obj.effective_clusters[source] < 1):
                raise ValueError("invalid quantizer tensor dimensions")
            if any(not torch.isfinite(x).all() for x in state.values()) or (state["scale"] <= 0).any():
                raise ValueError("invalid quantizer tensor values")
        return obj
