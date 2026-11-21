"""Target-conditioned Transformer, sparse U/S experts and SURD-guided objective."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .counts import (
    COUNT_FEAT_DIM, DEFAULT_VISIT_BINS, TRANSITION_ATTN_MODES, TRANSITION_PHI_DIM,
)
from .data import L_DIM, T_DIM
from .features import (
    GEO_FEAT_DIM, LAST_SLOT_DIM, SESSION_SCALAR_DIM, TRAJ_LENGTH_BINS, USER_HOUR_DIM,
)
from .surd import SOURCE_ORDER, SUBSETS

ROUTER_MODES = ("hard_prior", "surd_mlp", "hybrid")
CANDIDATE_CONTEXT_MODES = ("off", "on")
CANDIDATE_CONTEXT_TIME_MODES = ("off", "last_slot", "causal_slot")
ON_OFF = ("off", "on")
EXPERT_CROSS_ATTN_MODES = ("off", "hybrid", "full")
# Return-vs-explore gate: mix revisit vs spatial explore by predicted P(return).
RETURN_GATE_MODES = ("off", "on")
EXPLORE_SPATIAL_MODES = ("off", "gauss", "geo", "radiation")
# Per-candidate gate from SURD routing prior / Specific MI (see docs/causal-counts.md).
SURD_BIAS_GATE_MODES = ("off", "unique_p", "smi_p", "smi_pc", "u_mass", "confidence")
# How the gate modulates transition-score-bias / transition-attn λ.
# off = unchanged; mul = term·g; residual = term·(1+g) (SURD amplifies).
SURD_GATE_APPLY_MODES = ("off", "mul", "residual")


def compute_surd_bias_gate(mode, prior26, confidence, specific15=None):
    """Nonnegative per-candidate gate from SURD / Specific MI (SUBSET_ORDER).

    ``unique_p`` / ``u_mass`` use routing-prior atoms; ``smi_p`` / ``smi_pc`` use
    Specific MI (normalized by Σ specific); ``confidence`` uses SURD confidence.
    Returns ``None`` when ``mode == "off"``.
    """
    if mode in (None, "off"):
        return None
    if mode not in SURD_BIAS_GATE_MODES:
        raise ValueError(f"surd_bias_gate must be one of {SURD_BIAS_GATE_MODES}")
    w = confidence
    if not torch.is_tensor(w):
        raise TypeError("confidence must be a tensor")
    if mode == "confidence":
        return w.clamp_min(0)
    if mode == "unique_p":
        return (w * prior26[..., 0].clamp_min(0)).clamp_min(0)
    if mode == "u_mass":
        return (w * prior26[..., :4].sum(-1).clamp_min(0)).clamp_min(0)
    if mode in ("smi_p", "smi_pc"):
        if specific15 is None:
            raise ValueError(f"surd_bias_gate={mode} requires specific_information")
        # SUBSET_ORDER: P=0, C=1, T=2, L=3, PC=4, …
        idx = 0 if mode == "smi_p" else 4
        total = specific15.sum(-1).clamp_min(1e-8)
        frac = specific15[..., idx].clamp_min(0) / total
        return (w * frac).clamp_min(0)
    raise ValueError(f"unknown surd_bias_gate {mode}")


def apply_surd_gate(term, gate, mode):
    """Apply ``off|mul|residual`` gate to a score/attn term (broadcasts on gate)."""
    if mode in (None, "off") or gate is None:
        return term
    if mode == "mul":
        return term * gate
    if mode == "residual":
        return term * (1.0 + gate)
    raise ValueError(f"surd gate apply mode must be one of {SURD_GATE_APPLY_MODES}")


@dataclass(frozen=True)
class ModelConfig:
    dim: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = .1
    top_k: int = 4
    temperature: float = 1.
    lambda_prior: float = .1
    use_surd: bool = True
    # hard_prior (default): π from candidate U/S SURD mass only — no sequence in router.
    # surd_mlp: π = softmax(MLP(SURD feats) / T); SURD-only, no sequence.
    # hybrid: logits = λ_prior · surd_bias + λ_hybrid · learned(seq×cand×SURD).
    router_mode: str = "hard_prior"
    # ε on the sequence×candidate learned term in hybrid (default 0.1 = small).
    hybrid_lambda: float = .1
    # Causal online count tables + transition-aware cross-attn (runtime; no prepare change).
    causal_counts: bool = True
    # bias (default) | gate | rel_key — see docs/causal-counts.md.
    transition_attn: str = "bias"
    transition_lambda: float = 1.
    # Additive Acc@1 prior: λ·log1p(last→cand POI transition) on scores (0 = off).
    transition_score_bias: float = 0.
    # Hypergraph-style Acc@1 prior: λ·log1p(last_cat→cand_poi) via cat_transition (0 = off).
    category_score_bias: float = 0.
    # Popularity Acc@1 prior: λ·log1p(global POI visits) from causal cand feats (0 = off).
    pop_score_bias: float = 0.
    # Proximity Acc@1 prior: −λ·log1p(haversine km) when delta_geo is present (0 = off).
    # Needs --candidate-context on (WIN already sets it).
    geo_score_bias: float = 0.
    # Gravity-style distance exponent on geo prior: −λ·log1p(km^α). α=1 ≡ default log1p(km).
    geo_score_power: float = 1.
    # Gaussian spatial kernel Acc@1 prior: λ·exp(−km²/(2σ²)); 0 = off. Prefer replace geo.
    gauss_geo_bias: float = 0.
    gauss_sigma_km: float = 0.5
    # Modulate geo / gauss by last→cand transition mass: off | sparse (more spatial when tsb thin).
    geo_adapt: str = "off"
    geo_adapt_temp: float = 1.0
    # Acc@1 revisit prior: λ·log1p(user→cand visits) from cand_poi_feat[...,0] (0 = off).
    revisit_score_bias: float = 0.
    # Orthogonal revisit features (keep revisit_score_bias locked; do not raise it).
    # λ·1[cand ∈ last-K history tips].
    history_hit_bias: float = 0.
    history_hit_k: int = 5
    # λ·exp(−age/τ) where age = steps from tip to most recent hist visit of cand.
    revisit_recency_bias: float = 0.
    revisit_recency_tau: float = 3.0
    # Simini radiation Acc@1 prior (replace flat geo): λ·log P_rad(j|i) over pool.
    # log P ∝ log m_j − log(m_i+s_ij) − log(m_i+m_j+s_ij); s_ij = intervening visit mass.
    radiation_score_bias: float = 0.
    # hard = strict disk intervening among candidates; soft = sigmoid-smoothed mass.
    radiation_kernel: str = "hard"
    radiation_soft_km: float = 0.25
    # Empirical jump-length Acc@1 prior: λ·log P(Δr) with Δr = haversine km (replace geo).
    jump_length_bias: float = 0.
    jump_dist: str = "lognormal"  # lognormal | trunc_power
    # Lognormal params on log(km+eps); trunc_power uses jump_beta / jump_d0.
    jump_mu: float = 0.0
    jump_sigma: float = 1.0
    jump_beta: float = 1.5
    jump_d0: float = 0.1
    # Return-vs-explore gate: π·(λ_ret·revisit) + (1−π)·(λ_exp·spatial). Default off.
    return_gate: str = "off"
    explore_spatial_bias: str = "off"  # off|gauss|geo|radiation
    explore_score_bias: float = 0.0
    # Familiar vs unfamiliar area soft blend (#44): no MLP — scalar f from local hist density.
    # score += (1−f)·λ_exp·gauss; locked revisit λ stays. Default off.
    fam_area_blend: str = "off"  # off|on
    fam_radius_km: float = 0.5
    fam_explore_bias: float = 0.0
    fam_return_bias: float = 0.0
    # Multi-hop tsb: blend last K history→cand transitions (1 = last-only default).
    tsb_history_k: int = 1
    tsb_history_decay: float = 0.5  # w_o ∝ exp(−o/κ); o=0 is last tip
    # Δt-bin tsb: hour edges e.g. (1,6,24) → bins <1h / 1–6 / 6–24 / >24; empty = off.
    # Uses bin-specific last→cand counts with pooled fallback when bin sparse.
    tsb_dt_edges: tuple = ()
    # Category-stage Acc@1 prior: λ·log1p(last_cat→cand_cat); soft (no hard filter).
    cat_stage_bias: float = 0.
    # Multiply transition-score-bias by exp(−Δt_hours/τ); 0 = off (no time decay).
    tsb_decay_hours: float = 0.
    # PRME-style session-gap gate: keep full tsb only when Δt(last→now) ≤ τ hours; else 0.
    # Distinct from continuous Hawkes decay / Δt-bins (both Acc Discard). 0 = off.
    tsb_session_gap_hours: float = 0.
    visit_bins: int = DEFAULT_VISIT_BINS
    # Optional overlays (all default off — Acc@1 path unchanged unless set).
    candidate_context: str = "off"
    candidate_context_time: str = "off"
    session_summary: str = "off"
    user_pref: str = "off"
    # Learned embedding of train user_id → base/context (default off; not hard_prior).
    user_id_emb: str = "off"
    transition_pmi: str = "off"
    expert_cross_attn: str = "off"
    # When expert_cross_attn ≠ off: only run expert-local attn for router top-k (E3).
    expert_cross_attn_sparse: str = "on"
    # SURD / Specific-MI controlled bias weights (all default off / 0).
    surd_bias_gate: str = "off"
    # Additive score term: λ · gate (needs surd_bias_gate ≠ off when λ > 0).
    surd_score_bias: float = 0.
    # How gate modulates transition-score-bias / category-score-bias (off = flat λ).
    tsb_gate_mode: str = "off"
    # How gate modulates transition-attn λ in bias mode (broadcast over history).
    surd_attn_gate: str = "off"

    def __post_init__(self):
        if self.dim < 1 or self.heads < 1 or self.dim % self.heads or self.layers < 1:
            raise ValueError("positive hidden dimension must be divisible by heads; layers must be positive")
        if not 1 <= self.top_k <= 15 or self.temperature <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("invalid top_k, temperature or dropout")
        if self.router_mode not in ROUTER_MODES:
            raise ValueError(f"router_mode must be one of {ROUTER_MODES}")
        if self.hybrid_lambda < 0:
            raise ValueError("hybrid_lambda must be nonnegative")
        if self.transition_attn not in TRANSITION_ATTN_MODES:
            raise ValueError(f"transition_attn must be one of {TRANSITION_ATTN_MODES}")
        if self.transition_lambda < 0:
            raise ValueError("transition_lambda must be nonnegative")
        if self.transition_score_bias < 0:
            raise ValueError("transition_score_bias must be nonnegative")
        if self.category_score_bias < 0:
            raise ValueError("category_score_bias must be nonnegative")
        if self.pop_score_bias < 0:
            raise ValueError("pop_score_bias must be nonnegative")
        if self.geo_score_bias < 0:
            raise ValueError("geo_score_bias must be nonnegative")
        if self.geo_score_bias > 0 and self.candidate_context == "off":
            raise ValueError("geo_score_bias > 0 requires --candidate-context on")
        if self.geo_score_power <= 0:
            raise ValueError("geo_score_power must be positive")
        if self.gauss_geo_bias < 0:
            raise ValueError("gauss_geo_bias must be nonnegative")
        if self.gauss_geo_bias > 0 and self.gauss_sigma_km <= 0:
            raise ValueError("gauss_sigma_km must be positive when gauss_geo_bias > 0")
        if self.gauss_geo_bias > 0 and self.candidate_context == "off":
            raise ValueError("gauss_geo_bias > 0 requires --candidate-context on")
        if self.geo_adapt not in ("off", "sparse"):
            raise ValueError("geo_adapt must be off|sparse")
        if self.geo_adapt_temp < 0:
            raise ValueError("geo_adapt_temp must be nonnegative")
        if self.revisit_score_bias < 0:
            raise ValueError("revisit_score_bias must be nonnegative")
        if self.history_hit_bias < 0:
            raise ValueError("history_hit_bias must be nonnegative")
        if self.history_hit_k < 1:
            raise ValueError("history_hit_k must be >= 1")
        if self.revisit_recency_bias < 0:
            raise ValueError("revisit_recency_bias must be nonnegative")
        if self.revisit_recency_tau <= 0:
            raise ValueError("revisit_recency_tau must be positive")
        if self.radiation_score_bias < 0:
            raise ValueError("radiation_score_bias must be nonnegative")
        if self.radiation_score_bias > 0 and self.candidate_context == "off":
            raise ValueError("radiation_score_bias > 0 requires --candidate-context on")
        if self.radiation_kernel not in ("hard", "soft"):
            raise ValueError("radiation_kernel must be hard|soft")
        if self.radiation_soft_km <= 0:
            raise ValueError("radiation_soft_km must be positive")
        if self.jump_length_bias < 0:
            raise ValueError("jump_length_bias must be nonnegative")
        if self.jump_length_bias > 0 and self.candidate_context == "off":
            raise ValueError("jump_length_bias > 0 requires --candidate-context on")
        if self.jump_dist not in ("lognormal", "trunc_power"):
            raise ValueError("jump_dist must be lognormal|trunc_power")
        if self.jump_sigma <= 0:
            raise ValueError("jump_sigma must be positive")
        if self.jump_beta <= 0:
            raise ValueError("jump_beta must be positive")
        if self.jump_d0 < 0:
            raise ValueError("jump_d0 must be nonnegative")
        if self.return_gate not in RETURN_GATE_MODES:
            raise ValueError("return_gate must be off|on")
        if self.explore_spatial_bias not in EXPLORE_SPATIAL_MODES:
            raise ValueError("explore_spatial_bias must be off|gauss|geo|radiation")
        if self.explore_score_bias < 0:
            raise ValueError("explore_score_bias must be nonnegative")
        if self.tsb_history_k < 1:
            raise ValueError("tsb_history_k must be >= 1")
        if self.tsb_history_decay <= 0:
            raise ValueError("tsb_history_decay must be positive")
        edges = tuple(float(x) for x in (self.tsb_dt_edges or ()))
        if any(e <= 0 for e in edges):
            raise ValueError("tsb_dt_edges must be positive hours")
        if edges != tuple(sorted(edges)):
            raise ValueError("tsb_dt_edges must be sorted ascending")
        if self.cat_stage_bias < 0:
            raise ValueError("cat_stage_bias must be nonnegative")
        if self.return_gate == "on":
            if self.revisit_score_bias <= 0:
                raise ValueError("return_gate on requires --revisit-score-bias > 0")
            if self.explore_spatial_bias == "off" or self.explore_score_bias <= 0:
                raise ValueError(
                    "return_gate on requires --explore-spatial-bias ≠ off and "
                    "--explore-score-bias > 0")
            if self.explore_spatial_bias in ("gauss", "geo", "radiation") and (
                    self.candidate_context == "off"):
                raise ValueError("return-gate explore spatial requires --candidate-context on")
            if self.explore_spatial_bias == "radiation" and not self.causal_counts:
                raise ValueError("explore radiation requires causal counts")
        if self.fam_area_blend not in ("off", "on"):
            raise ValueError("fam_area_blend must be off|on")
        if self.fam_area_blend == "on":
            if self.fam_radius_km <= 0:
                raise ValueError("fam_radius_km must be positive when fam_area_blend on")
            if self.fam_explore_bias < 0 or self.fam_return_bias < 0:
                raise ValueError("fam explore/return biases must be nonnegative")
            if self.fam_explore_bias <= 0 and self.fam_return_bias <= 0:
                raise ValueError("fam_area_blend on requires a positive fam bias")
            if self.candidate_context == "off":
                raise ValueError("fam_area_blend requires --candidate-context on")
        if self.tsb_decay_hours < 0:
            raise ValueError("tsb_decay_hours must be nonnegative")
        if self.tsb_session_gap_hours < 0:
            raise ValueError("tsb_session_gap_hours must be nonnegative")
        if self.visit_bins < 1:
            raise ValueError("visit_bins must be positive")
        if self.candidate_context not in CANDIDATE_CONTEXT_MODES:
            raise ValueError(f"candidate_context must be one of {CANDIDATE_CONTEXT_MODES}")
        if self.candidate_context_time not in CANDIDATE_CONTEXT_TIME_MODES:
            raise ValueError(
                f"candidate_context_time must be one of {CANDIDATE_CONTEXT_TIME_MODES}")
        for name, value in (
                ("session_summary", self.session_summary),
                ("user_pref", self.user_pref),
                ("user_id_emb", self.user_id_emb),
                ("transition_pmi", self.transition_pmi),
                ("expert_cross_attn_sparse", self.expert_cross_attn_sparse)):
            if value not in ON_OFF:
                raise ValueError(f"{name} must be one of {ON_OFF}")
        if self.expert_cross_attn not in EXPERT_CROSS_ATTN_MODES:
            raise ValueError(f"expert_cross_attn must be one of {EXPERT_CROSS_ATTN_MODES}")
        if self.surd_bias_gate not in SURD_BIAS_GATE_MODES:
            raise ValueError(f"surd_bias_gate must be one of {SURD_BIAS_GATE_MODES}")
        if self.surd_score_bias < 0:
            raise ValueError("surd_score_bias must be nonnegative")
        if self.tsb_gate_mode not in SURD_GATE_APPLY_MODES:
            raise ValueError(f"tsb_gate_mode must be one of {SURD_GATE_APPLY_MODES}")
        if self.surd_attn_gate not in SURD_GATE_APPLY_MODES:
            raise ValueError(f"surd_attn_gate must be one of {SURD_GATE_APPLY_MODES}")
        needs_gate = (
            self.surd_score_bias > 0
            or self.tsb_gate_mode != "off"
            or self.surd_attn_gate != "off")
        if needs_gate and self.surd_bias_gate == "off":
            raise ValueError(
                "surd_score_bias / tsb_gate_mode / surd_attn_gate require "
                "--surd-bias-gate ≠ off")
        if not self.use_surd and self.surd_bias_gate != "off":
            raise ValueError("--surd-bias-gate requires SURD (omit --no-surd)")


def mlp(inputs, hidden, outputs, dropout=0.):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, outputs))


class SynergyExpert(nn.Module):
    def __init__(self, dim, degree, dropout):
        super().__init__()
        self.interaction = nn.Linear(dim, dim, bias=False)
        self.residual = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(degree))
        self.norm = nn.LayerNorm(dim)
        self.ffn = mlp(dim, 2 * dim, dim, dropout)

    def forward(self, members, projected):
        combined = self.interaction(torch.stack(projected).prod(0))
        residual = sum(layer(z) for layer, z in zip(self.residual, members))
        return self.ffn(self.norm(combined + residual / math.sqrt(len(members))))


class TransitionCrossAttention(nn.Module):
    """Candidate←history cross-attn with transition counts (bias / gate / rel_key).

    Modes (``docs/causal-counts.md``):
      * **bias** — add ``λ · log1p(count)`` to attention logits (default).
      * **gate** — sigmoid MLP on ``[history; candidate; φ(count)]`` scales weights.
      * **rel_key** — add ``MLP(φ(count))`` into keys (candidate-dependent).

    When ``use_pmi`` is True, transition tensors are treated as already-logged PMI
    values (may be negative) and are **not** passed through ``log1p``.

    P-source counts are POI→POI; C-source counts are category→POI.
    """

    def __init__(self, dim, heads, dropout, mode, transition_lambda, use_pmi=False):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.mode = mode
        self.transition_lambda = float(transition_lambda)
        self.use_pmi = bool(use_pmi)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        scale = self.head_dim ** -0.5
        self.register_buffer("scale", torch.tensor(scale), persistent=False)
        if mode == "gate":
            # [history; candidate; φ(count)] → sigmoid gate per (cand, hist) pair.
            self.gate_mlp = mlp(2 * dim + TRANSITION_PHI_DIM, dim, 1, dropout)
        elif mode == "rel_key":
            self.rel_key = mlp(TRANSITION_PHI_DIM, dim, dim, dropout)
        else:
            self.gate_mlp = None
            self.rel_key = None

    def forward(self, query, key, value, key_padding_mask, transition_counts,
                cand_gate=None, gate_apply="off"):
        """
        query: [B, C, D] candidates · key/value: [B, L, D] history
        key_padding_mask: [B, L] True = **ignore** (same as nn.MultiheadAttention)
        transition_counts: [B, C, L] raw counts (≥0) or PMI (any real) when use_pmi
        cand_gate: optional [B, C] SURD/SMI gate; only scales **bias**-mode λ·φ
        gate_apply: off | mul | residual (see ``apply_surd_gate``)
        """
        b, c, _ = query.shape
        length = key.shape[1]
        h, d = self.heads, self.head_dim
        q = self.q_proj(query).view(b, c, h, d).transpose(1, 2)          # [B,H,C,d]
        if self.use_pmi:
            phi = transition_counts.to(query.dtype)
        else:
            phi = torch.log1p(transition_counts.clamp_min(0)).to(query.dtype)  # [B,C,L]
        if self.mode == "rel_key":
            # Candidate-dependent keys: K[b,c,j] += MLP(φ(count[b,c,j])).
            rel = self.rel_key(phi[..., None])  # [B,C,L,D]
            key = key[:, None, :, :] + rel      # [B,C,L,D]
            k = self.k_proj(key).view(b, c, length, h, d).permute(0, 3, 1, 2, 4)  # B,H,C,L,d
            v = self.v_proj(value)[:, None, :, :].expand(-1, c, -1, -1)
            v = v.reshape(b, c, length, h, d).permute(0, 3, 1, 2, 4)
            logits = (q.unsqueeze(3) * k).sum(-1) * self.scale  # [B,H,C,L]
        else:
            k = self.k_proj(key).view(b, length, h, d).transpose(1, 2)   # [B,H,L,d]
            v = self.v_proj(value).view(b, length, h, d).transpose(1, 2)
            logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B,H,C,L]
            if self.mode == "bias":
                bias_term = self.transition_lambda * phi  # [B,C,L]
                if cand_gate is not None and gate_apply not in (None, "off"):
                    g = cand_gate.to(bias_term.dtype)[:, :, None]
                    bias_term = apply_surd_gate(bias_term, g, gate_apply)
                logits = logits + bias_term[:, None, :, :]
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        if self.mode == "gate":
            # Broadcast history/candidate to [B,C,L,*] for the gate MLP.
            hist = key[:, None, :, :].expand(-1, c, -1, -1)
            cand = query[:, :, None, :].expand(-1, -1, length, -1)
            gate = self.gate_mlp(torch.cat([hist, cand, phi[..., None]], -1)).sigmoid()
            gate = gate.squeeze(-1)[:, None, :, :]  # [B,1,C,L]
            weights = weights * gate
            # Renormalize over history (keep mass on valid keys).
            denom = weights.sum(-1, keepdim=True).clamp_min(1e-8)
            weights = weights / denom
        weights = self.dropout(weights)
        if self.mode == "rel_key":
            out = (weights.unsqueeze(-1) * v).sum(3)  # [B,H,C,d]
        else:
            out = torch.matmul(weights, v)  # [B,H,C,d]
        out = out.transpose(1, 2).contiguous().view(b, c, self.dim)
        return self.out_proj(out)


class TSMoE(nn.Module):
    def __init__(self, n_pois, n_categories, max_length, config, n_users=0):
        super().__init__()
        self.cfg = config
        d = config.dim
        self.n_pois = n_pois
        self.n_categories = n_categories
        # Train-user vocabulary size (0 OK when user_id_emb off); tokens use pad=0, UNK=1, row+2.
        self.n_users = max(0, int(n_users))
        self.max_length = max_length
        self.projections = nn.ModuleDict({
            "P": nn.Embedding(n_pois + 2, d, padding_idx=0),
            "C": nn.Embedding(n_categories + 2, d, padding_idx=0),
            "T": mlp(T_DIM, d, d), "L": mlp(L_DIM, d, d)})
        self.positions = nn.Embedding(max_length, d)
        self.encoders = nn.ModuleDict({s: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, config.heads, 4 * d, config.dropout, activation="gelu",
                                       batch_first=True, norm_first=True),
            config.layers, norm=nn.LayerNorm(d), enable_nested_tensor=False) for s in SOURCE_ORDER})
        # TransformerEncoder clones its prototype layer; initialize distinct weights at every depth.
        for encoder in self.encoders.values():
            for layer in encoder.layers:
                nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
                for module in layer.modules():
                    if isinstance(module, nn.Linear):
                        module.reset_parameters()
        # Standard MHA kept for T/L (and for P/C when causal_counts is off).
        self.attention = nn.ModuleDict({s: nn.MultiheadAttention(d, config.heads, batch_first=True)
                                        for s in SOURCE_ORDER})
        self.context = mlp(5 * d, 2 * d, d, config.dropout)
        self.base = nn.Sequential(mlp(5 * d, 2 * d, d, config.dropout), nn.Linear(d, 1))
        self.prior = mlp(26, 64, 64)
        # Hybrid router: sequence context + candidate + SURD embedding (66 = 64+ρ+α).
        self.router = mlp(2 * d + 66, 2 * d, 15, config.dropout)
        self.gate = mlp(2 * d + 66, d, 1, config.dropout)
        # SURD-only router / gate (hard_prior uses fixed bias; surd_mlp uses surd_router).
        self.surd_router = mlp(66, 64, 15, config.dropout)
        self.surd_gate = mlp(66, d, 1, config.dropout)
        self.shared = nn.ModuleList(nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d)) for _ in SOURCE_ORDER)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), mlp(d, 2 * d, d, config.dropout)) if len(a) == 1
            else SynergyExpert(d, len(a), config.dropout) for a in SUBSETS)
        self.moe_head = nn.Linear(d, 1)
        # Causal-count modules (always constructed so checkpoints are stable; unused when off).
        self.poi_count_mlp = mlp(COUNT_FEAT_DIM, d, d, config.dropout)
        self.cat_count_mlp = mlp(COUNT_FEAT_DIM, d, d, config.dropout)
        self.visit_bin_emb = nn.Embedding(config.visit_bins, d)
        use_pmi = config.transition_pmi == "on"
        self.transition_attn = nn.ModuleDict({
            "P": TransitionCrossAttention(
                d, config.heads, config.dropout, config.transition_attn,
                config.transition_lambda, use_pmi=use_pmi),
            "C": TransitionCrossAttention(
                d, config.heads, config.dropout, config.transition_attn,
                config.transition_lambda, use_pmi=use_pmi),
        })
        # --- A: candidate context (cat + Δgeo + optional time) ---
        # Fuse input: e_P + e_C + geo(+time/slot) → D. Always built for ckpt stability.
        fuse_in = 3 * d + GEO_FEAT_DIM  # will project geo/time into D first
        self.cand_geo_mlp = mlp(GEO_FEAT_DIM, d, d, config.dropout)
        self.cand_time_mlp = mlp(LAST_SLOT_DIM, d, d, config.dropout)
        self.cand_slot_mlp = mlp(COUNT_FEAT_DIM, d, d, config.dropout)
        # Fuse [e_P; e_C; geo_d; time_d] → D (time_d may be zero when time off).
        self.cand_fuse = mlp(4 * d, 2 * d, d, config.dropout)
        # --- B: session summary ---
        self.session_length_emb = nn.Embedding(TRAJ_LENGTH_BINS, d)
        self.session_cat_proj = nn.Linear(n_categories, d)
        self.session_scalar_mlp = mlp(SESSION_SCALAR_DIM, d, d, config.dropout)
        self.session_mlp = mlp(3 * d, 2 * d, d, config.dropout)
        self.session_to_full = nn.Linear(d, 5 * d)
        # --- C: user preference ---
        self.user_cat_proj = nn.Linear(n_categories, d)
        self.user_hour_proj = nn.Linear(USER_HOUR_DIM, d)
        self.user_pref_mlp = mlp(2 * d, 2 * d, d, config.dropout)
        self.user_pref_to_full = nn.Linear(d, 5 * d)
        # --- user_id embedding (identity; separate from causal user-pref counts) ---
        self.user_emb = nn.Embedding(self.n_users + 2, d, padding_idx=0)
        self.user_id_to_full = nn.Linear(d, 5 * d)
        # --- E: expert-local cross-attn (shared QKV per source; transition stays on shared P/C) ---
        self.expert_attn = nn.ModuleDict({
            s: nn.MultiheadAttention(d, config.heads, batch_first=True) for s in SOURCE_ORDER
        })
        self.expert_query_proj = nn.ModuleDict({
            s: nn.Linear(d, d) for s in SOURCE_ORDER
        })
        # Return-vs-explore gate: π from last P emb + user-visit richness scalar.
        # Always constructed for checkpoint stability; unused when return_gate=off.
        self.return_gate_mlp = mlp(d + 1, max(d // 2, 8), 1, config.dropout)
        # Next-category MTL aux head (#45): CE on tip P/C encoding → cat; inference unused.
        self.cat_aux_head = nn.Linear(d, n_categories)

    def _flag_on(self, name):
        return getattr(self.cfg, name) == "on"

    def _project_history(self, history, count_ctx=None):
        """Token / numeric projections, optionally + causal count features / visit bin."""
        projected = {}
        projected["P"] = self.projections["P"](history["P"])
        projected["C"] = self.projections["C"](history["C"])
        projected["T"] = self.projections["T"](history["T"])
        projected["L"] = self.projections["L"](history["L"])
        if count_ctx is not None and self.cfg.causal_counts:
            if "hist_poi_feat" in count_ctx:
                projected["P"] = projected["P"] + self.poi_count_mlp(count_ctx["hist_poi_feat"])
            if "hist_cat_feat" in count_ctx:
                projected["C"] = projected["C"] + self.cat_count_mlp(count_ctx["hist_cat_feat"])
            # Equal-frequency bin of user cumulative visits (not raw counts) → T and L.
            if "visit_bin" in count_ctx:
                bins = count_ctx["visit_bin"].clamp(0, self.cfg.visit_bins - 1)
                visit = self.visit_bin_emb(bins)[:, None, :]
                projected["T"] = projected["T"] + visit
                projected["L"] = projected["L"] + visit
        return projected

    def encode(self, history, valid, count_ctx=None):
        position = self.positions(torch.arange(valid.shape[1], device=valid.device))
        projected = self._project_history(history, count_ctx)
        return {s: self.encoders[s](projected[s] + position,
                                    src_key_padding_mask=~valid) for s in SOURCE_ORDER}

    def _build_candidate_vector(self, rows, mask, count_ctx=None):
        """P-embed (+ optional counts / A-context fuse) → candidate query [B,M,D]."""
        safe = rows.masked_fill(~mask, -1) + 2
        e_p = self.projections["P"](safe)
        if count_ctx is not None and self.cfg.causal_counts and "cand_poi_feat" in count_ctx:
            feats = count_ctx["cand_poi_feat"][:, :rows.shape[1]].to(e_p.device)
            e_p = e_p + self.poi_count_mlp(feats)
        if not self._flag_on("candidate_context") and self.cfg.candidate_context_time == "off":
            return e_p, {"P": e_p, "C": e_p, "T": e_p, "L": e_p}
        # Typed pieces for A fuse and for E typed queries.
        b, m = rows.shape
        device = e_p.device
        if count_ctx is not None and "cand_cat_rows" in count_ctx:
            cat_rows = count_ctx["cand_cat_rows"][:, :m].to(device)
            cat_tok = cat_rows.masked_fill(cat_rows < 0, -1) + 2
            cat_tok = cat_tok.masked_fill(~mask, 0)
            e_c = self.projections["C"](cat_tok)
        else:
            e_c = torch.zeros_like(e_p)
        if count_ctx is not None and "delta_geo" in count_ctx and self._flag_on("candidate_context"):
            geo = self.cand_geo_mlp(count_ctx["delta_geo"][:, :m].to(device))
        else:
            geo = torch.zeros_like(e_p)
        time_d = torch.zeros_like(e_p)
        if self.cfg.candidate_context_time == "last_slot" and count_ctx is not None and "last_slot" in count_ctx:
            slot = self.cand_time_mlp(count_ctx["last_slot"].to(device))  # [B,D]
            time_d = slot[:, None, :].expand(-1, m, -1)
        elif self.cfg.candidate_context_time == "causal_slot" and count_ctx is not None:
            if "last_slot" in count_ctx:
                slot = self.cand_time_mlp(count_ctx["last_slot"].to(device))
                time_d = time_d + slot[:, None, :].expand(-1, m, -1)
            if "cand_slot_feat" in count_ctx:
                time_d = time_d + self.cand_slot_mlp(count_ctx["cand_slot_feat"][:, :m].to(device))
        if self._flag_on("candidate_context") or self.cfg.candidate_context_time != "off":
            fused = self.cand_fuse(torch.cat([e_p, e_c, geo, time_d], -1))
            # Keep padding candidates at zero contribution via mask (score masks later).
            fused = fused * mask.unsqueeze(-1).to(fused.dtype)
        else:
            fused = e_p
        typed = {"P": e_p, "C": e_c if e_c.abs().sum() > 0 else fused,
                 "T": time_d if time_d.abs().sum() > 0 else fused,
                 "L": geo if geo.abs().sum() > 0 else fused}
        return fused, typed

    def _session_addon(self, count_ctx, batch, device):
        if not self._flag_on("session_summary") or count_ctx is None:
            return None
        if "session_scalars" not in count_ctx:
            return None
        scalars = self.session_scalar_mlp(count_ctx["session_scalars"].to(device))
        length = count_ctx["session_length_bin"].to(device).clamp(0, TRAJ_LENGTH_BINS - 1)
        length_e = self.session_length_emb(length)
        cat_h = self.session_cat_proj(count_ctx["session_cat_hist"].to(device))
        sess = self.session_mlp(torch.cat([scalars, length_e, cat_h], -1))
        return self.session_to_full(sess)  # [B, 5D]

    def _user_pref_addon(self, count_ctx, device):
        if not self._flag_on("user_pref") or count_ctx is None:
            return None
        if "user_cat_pref" not in count_ctx or "user_hour_pref" not in count_ctx:
            return None
        cat_p = self.user_cat_proj(count_ctx["user_cat_pref"].to(device))
        hour_p = self.user_hour_proj(count_ctx["user_hour_pref"].to(device))
        pref = self.user_pref_mlp(torch.cat([cat_p, hour_p], -1))
        return self.user_pref_to_full(pref)  # [B, 5D]

    def _user_id_addon(self, count_ctx, device):
        """Learned user_id embedding → shared full (base/context); never hard_prior."""
        if not self._flag_on("user_id_emb") or count_ctx is None:
            return None
        if "user_token" not in count_ctx:
            return None
        tok = count_ctx["user_token"].to(device).long().clamp(0, self.n_users + 1)
        return self.user_id_to_full(self.user_emb(tok))  # [B, 5D]

    def _cross_attend(self, encoded, valid, candidate, count_ctx=None, cand_gate=None):
        """Per-source candidate←history attention; P/C use transition modes when enabled."""
        pad_mask = ~valid
        use = (count_ctx is not None and self.cfg.causal_counts
               and "poi_transition" in count_ctx and "cat_transition" in count_ctx)
        attn_gate_mode = getattr(self.cfg, "surd_attn_gate", "off")
        z = []
        for s in SOURCE_ORDER:
            if use and s in self.transition_attn:
                counts = count_ctx["poi_transition" if s == "P" else "cat_transition"]
                # Align width to this candidate chunk.
                c = candidate.shape[1]
                counts = counts[:, :c, :encoded[s].shape[1]].to(candidate.device)
                gate = None
                if (cand_gate is not None and attn_gate_mode not in (None, "off")
                        and self.cfg.transition_attn == "bias"):
                    gate = cand_gate[:, :c]
                z.append(self.transition_attn[s](
                    candidate, encoded[s], encoded[s], pad_mask, counts,
                    cand_gate=gate, gate_apply=attn_gate_mode))
            else:
                z.append(self.attention[s](
                    candidate, encoded[s], encoded[s],
                    key_padding_mask=pad_mask, need_weights=False)[0])
        return z

    def _expert_local_attend(self, encoded, valid, typed_queries, sources=None):
        """Plain MHA per source with typed queries (no transition — stays on shared P/C)."""
        pad_mask = ~valid
        sources = sources or list(SOURCE_ORDER)
        out = {}
        for s in sources:
            q = self.expert_query_proj[s](typed_queries[s])
            out[s] = self.expert_attn[s](
                q, encoded[s], encoded[s],
                key_padding_mask=pad_mask, need_weights=False)[0]
        return out

    def dispatch(self, z, indices, weights, mask, details, *, expert_z=None):
        """Group valid (batch,candidate,slot) assignments by expert before each FFN.

        When ``expert_z`` is provided (E1/E2), unique/synergy experts read from
        those typed streams instead of the shared ``z`` list.
        """
        d, k = self.cfg.dim, indices.shape[-1]
        source_z = expert_z if expert_z is not None else z
        flat_z = [v.reshape(-1, d) for v in source_z]
        projected = [layer(v) for layer, v in zip(self.shared, flat_z)]
        index = indices.reshape(-1, k)
        weight = weights.reshape(-1, k)
        assignments = mask.reshape(-1, 1)
        score = weight.new_zeros(index.shape[0])
        projected_scores = torch.zeros_like(weight) if details else None
        for e, members in enumerate(SUBSETS):
            pair, slot = torch.where((index == e) & assignments)
            if pair.numel() == 0:
                continue
            if len(members) == 1:
                out = self.experts[e](flat_z[members[0]][pair])
            else:
                out = self.experts[e]([flat_z[m][pair] for m in members], [projected[m][pair] for m in members])
            # A linear output head commutes with the weighted expert sum. Add its bias exactly once.
            projection = F.linear(out, self.moe_head.weight).squeeze(-1)
            score = score.index_add(0, pair, weight[pair, slot] * projection)
            if details:
                projected_scores[pair, slot] = projection
        return score.reshape_as(mask) + self.moe_head.bias, (
            projected_scores.reshape_as(weights) if details else None)

    @staticmethod
    def _last_hop_log1p(transition, rows, valid, device):
        """Gather last-history column of [B,M,L] transition tensor → λ·log1p scale input."""
        return TSMoE._multi_hop_log1p(transition, rows, valid, device, k=1, decay=0.5)

    @staticmethod
    def _multi_hop_log1p(transition, rows, valid, device, k=1, decay=0.5):
        """Blend last-K history→cand log1p counts with w_o∝exp(−o/κ) (o=0 = last tip)."""
        tr = transition[:, :rows.shape[1], :valid.shape[1]].to(device)
        lengths = valid.long().sum(-1).clamp_min(1)  # [B]
        k = max(1, int(k))
        decay = max(float(decay), 1e-6)
        acc = None
        w_sum = 0.0
        for o in range(k):
            w = math.exp(-o / decay)
            w_sum += w
            idx = (lengths - 1 - o).clamp_min(0)
            hop_ok = (lengths > o).to(tr.dtype)[:, None]
            col = tr.gather(2, idx.view(-1, 1, 1).expand(-1, tr.shape[1], 1)).squeeze(-1)
            term = torch.log1p(col.clamp_min(0)) * hop_ok
            acc = term * w if acc is None else acc + term * w
        return acc / w_sum

    @staticmethod
    def _radiation_log_score(km, mass_j, mass_i, kernel="hard", soft_km=0.25, eps=1e-6):
        """Simini radiation log-score over a candidate pool (ranking-invariant constants ok).

        ``km`` / ``mass_j``: [B, M]; ``mass_i``: [B] origin visit mass.
        ``hard``: intervening = sum of masses strictly closer than j.
        ``soft``: intervening = Σ_k m_k · σ((d_j − d_k) / soft_km).
        """
        m_j = mass_j.clamp_min(0)
        m_i = mass_i.clamp_min(0)[:, None]
        if kernel == "soft":
            # [B,M,M]: contrib of k to j's intervening ball
            diff = (km[:, :, None] - km[:, None, :]) / float(soft_km)
            w = torch.sigmoid(diff)
            # Exclude self (k==j) from intervening mass.
            eye = torch.eye(km.shape[1], device=km.device, dtype=km.dtype)[None]
            w = w * (1.0 - eye)
            s_ij = (w * m_j[:, None, :]).sum(-1)
        else:
            order = km.argsort(dim=1)
            m_sorted = m_j.gather(1, order)
            cum = torch.cumsum(m_sorted, dim=1)
            s_sorted = torch.zeros_like(cum)
            s_sorted[:, 1:] = cum[:, :-1]
            inv = order.argsort(dim=1)
            s_ij = s_sorted.gather(1, inv)
        return (
            torch.log(m_j.clamp_min(eps))
            - torch.log((m_i + s_ij).clamp_min(eps))
            - torch.log((m_i + m_j + s_ij).clamp_min(eps))
        )

    @staticmethod
    def _jump_log_density(km, dist="lognormal", mu=0.0, sigma=1.0, beta=1.5, d0=0.1, eps=1e-6):
        """log P(Δr) for empirical jump-length priors (unnormalized ok for ranking)."""
        d = km.clamp_min(0)
        if dist == "trunc_power":
            return -float(beta) * torch.log(d + float(d0) + eps)
        # lognormal on (d + eps)
        x = torch.log(d + eps)
        inv_s = 1.0 / float(sigma)
        return -math.log(float(sigma) * math.sqrt(2.0 * math.pi)) - x - 0.5 * ((x - float(mu)) * inv_s) ** 2

    def _sources_for_topk(self, top):
        """Source indices needed by the selected top-k experts (for E3 sparse)."""
        needed = set()
        # top: [B,M,K]
        flat = top.reshape(-1).tolist()
        for e in set(flat):
            for m in SUBSETS[e]:
                needed.add(m)
        return [SOURCE_ORDER[i] for i in sorted(needed)]

    def score(self, encoded, valid, rows, prior26, confidence, *, candidate_mask=None,
              details=False, count_ctx=None, surd_specific=None):
        mask = torch.ones_like(rows, dtype=torch.bool) if candidate_mask is None else candidate_mask
        if mask.shape != rows.shape:
            raise ValueError("candidate mask and candidate rows must have identical shapes")
        prior, w = prior26.detach(), confidence.detach()
        if not self.cfg.use_surd:
            w = torch.zeros_like(w)
        # Optional SURD/SMI gate (default off — flat transition / no additive SURD score).
        gate_mode = getattr(self.cfg, "surd_bias_gate", "off")
        smi = None if surd_specific is None else surd_specific.detach()
        cand_gate = compute_surd_bias_gate(gate_mode, prior, w, smi)
        candidate, typed = self._build_candidate_vector(rows, mask, count_ctx)
        eca = self.cfg.expert_cross_attn
        # Shared cross-attn (E0 / E1). E2 (full) skips shared attn for MoE; still builds base z.
        if eca == "full":
            # Base uses typed expert-local attn over all sources (no shared transition path).
            local = self._expert_local_attend(encoded, valid, typed)
            z = [local[s] for s in SOURCE_ORDER]
        else:
            z = self._cross_attend(encoded, valid, candidate, count_ctx, cand_gate=cand_gate)
        full = torch.cat([*z, candidate], -1)
        # B/C/user_id addons → base/context only (never hard_prior logits).
        sess = self._session_addon(count_ctx, rows.shape[0], candidate.device)
        if sess is not None:
            full = full + sess[:, None, :]
        pref = self._user_pref_addon(count_ctx, candidate.device)
        if pref is not None:
            full = full + pref[:, None, :]
        uid = self._user_id_addon(count_ctx, candidate.device)
        if uid is not None:
            full = full + uid[:, None, :]
        base = self.base(full).squeeze(-1)
        # U/S mass over the 15 experts (atoms 0..3 Unique, 15..25 Synergy); R is gate context only.
        mass = torch.cat((prior[..., :4], prior[..., 15:]), -1)
        alpha, rho = mass.sum(-1), prior[..., 4:15].sum(-1)
        prior_feat = self.prior(w[..., None] * prior)
        surd_inputs = torch.cat([prior_feat, (w * rho)[..., None], (w * alpha)[..., None]], -1)
        # Confidence-scaled log1p of U/S prior atoms (same 15-D expert order as EXPERT_ORDER).
        surd_bias = w[..., None] * torch.log1p(15 * mass)
        mode = self.cfg.router_mode
        if mode == "hard_prior":
            learned = surd_bias.new_zeros(surd_bias.shape)
            bias = surd_bias
            logits = surd_bias
            gate_logits = self.surd_gate(surd_inputs).squeeze(-1)
        elif mode == "surd_mlp":
            learned = self.surd_router(surd_inputs)
            bias = surd_bias.new_zeros(surd_bias.shape)
            logits = learned
            gate_logits = self.surd_gate(surd_inputs).squeeze(-1)
        else:  # hybrid
            inputs = torch.cat([self.context(full), candidate, prior_feat,
                                (w * rho)[..., None], (w * alpha)[..., None]], -1)
            learned = self.router(inputs)
            bias = self.cfg.lambda_prior * surd_bias
            logits = bias + self.cfg.hybrid_lambda * learned
            gate_logits = self.gate(inputs).squeeze(-1)
        log_pi = F.log_softmax(logits / self.cfg.temperature, -1)
        top = torch.argsort(log_pi, dim=-1, descending=True, stable=True)[..., :self.cfg.top_k]
        weights = log_pi.gather(-1, top).softmax(-1)
        gate = gate_logits.sigmoid()
        # E1: keep shared z for base; MoE residual uses expert-local typed attn.
        expert_z_list = None
        if eca == "hybrid":
            if self.cfg.expert_cross_attn_sparse == "on":
                sources = self._sources_for_topk(top)
            else:
                sources = list(SOURCE_ORDER)
            local = self._expert_local_attend(encoded, valid, typed, sources=sources)
            # Fill missing sources from shared z so dispatch indexing stays valid.
            expert_z_list = []
            for i, s in enumerate(SOURCE_ORDER):
                expert_z_list.append(local[s] if s in local else z[i])
        elif eca == "full":
            expert_z_list = z  # already expert-local
        residual, projected = self.dispatch(
            z, top, weights, mask, details, expert_z=expert_z_list)
        score = base + gate * residual
        gate_mode = getattr(self.cfg, "tsb_gate_mode", "off")
        # Light graph prior: last-K history → candidate POI transition (default K=1).
        # Optional Δt-bins: prefer bin-specific last→cand count, else pooled fallback.
        tb = float(self.cfg.transition_score_bias or 0.0)
        if (tb > 0 and count_ctx is not None and self.cfg.causal_counts
                and "poi_transition" in count_ctx):
            dt_edges = tuple(getattr(self.cfg, "tsb_dt_edges", ()) or ())
            if (dt_edges and "poi_transition_dt" in count_ctx
                    and "target_gap_hours" in count_ctx):
                # Pooled last-hop raw count (expm1 of log1p mass).
                pooled_l1p = self._last_hop_log1p(
                    count_ctx["poi_transition"], rows, valid, score.device)
                pooled = torch.expm1(pooled_l1p).clamp_min(0)
                dt = count_ctx["poi_transition_dt"][:, :rows.shape[1], :].to(score.device)
                gap = count_ctx["target_gap_hours"].to(score.device).clamp_min(0)
                edges_t = torch.tensor(dt_edges, device=score.device, dtype=gap.dtype)
                bid = torch.bucketize(gap, edges_t).clamp(0, dt.shape[-1] - 1)  # [B]
                bin_count = dt.gather(
                    2, bid.view(-1, 1, 1).expand(-1, dt.shape[1], 1)).squeeze(-1)
                use = torch.where(bin_count > 0, bin_count, pooled)
                tsb_term = tb * torch.log1p(use.clamp_min(0))
            else:
                tsb_term = tb * self._multi_hop_log1p(
                    count_ctx["poi_transition"], rows, valid, score.device,
                    k=int(getattr(self.cfg, "tsb_history_k", 1) or 1),
                    decay=float(getattr(self.cfg, "tsb_history_decay", 0.5) or 0.5))
            # Hawkes-lite: down-weight transition prior when last→target gap is large.
            tau = float(getattr(self.cfg, "tsb_decay_hours", 0.0) or 0.0)
            if tau > 0 and "target_gap_hours" in count_ctx:
                gap = count_ctx["target_gap_hours"].to(score.device).clamp_min(0)
                tsb_term = tsb_term * torch.exp(-gap / tau)[:, None]
            # Session-gap gate (PRME-style): zero tsb across long breaks (not continuous decay).
            gap_tau = float(getattr(self.cfg, "tsb_session_gap_hours", 0.0) or 0.0)
            if gap_tau > 0 and "target_gap_hours" in count_ctx:
                gap = count_ctx["target_gap_hours"].to(score.device).clamp_min(0)
                tsb_term = tsb_term * (gap <= gap_tau).to(score.dtype)[:, None]
            tsb_term = apply_surd_gate(tsb_term, cand_gate, gate_mode)
            score = score + tsb_term
        # Hypergraph prior: last category → candidate POI (cat_transition).
        csb = float(getattr(self.cfg, "category_score_bias", 0.0) or 0.0)
        if (csb > 0 and count_ctx is not None and self.cfg.causal_counts
                and "cat_transition" in count_ctx):
            csb_term = csb * self._last_hop_log1p(
                count_ctx["cat_transition"], rows, valid, score.device)
            csb_term = apply_surd_gate(csb_term, cand_gate, gate_mode)
            score = score + csb_term
        # Popularity prior: causal global POI log1p visits (cand_poi_feat dim 1).
        psb = float(getattr(self.cfg, "pop_score_bias", 0.0) or 0.0)
        if (psb > 0 and count_ctx is not None and self.cfg.causal_counts
                and "cand_poi_feat" in count_ctx):
            pop = count_ctx["cand_poi_feat"][:, :rows.shape[1], 1].to(score.device)
            score = score + psb * pop.clamp_min(0)
        # Spatial scale: 1 when geo_adapt=off; shrinks when last→cand transition is dense.
        spatial_scale = None
        if (getattr(self.cfg, "geo_adapt", "off") == "sparse"
                and count_ctx is not None and self.cfg.causal_counts
                and "poi_transition" in count_ctx):
            tsb_mass = self._last_hop_log1p(
                count_ctx["poi_transition"], rows, valid, score.device)  # already log1p
            # Convert back to count-ish: expm1(log1p(c)) = c.
            count_l1p = tsb_mass
            at = float(getattr(self.cfg, "geo_adapt_temp", 1.0) or 1.0)
            spatial_scale = 1.0 / (1.0 + at * count_l1p.clamp_min(0))
        # Proximity prior: closer candidates score higher (−λ · log1p(km^α)).
        # delta_geo[...,0] stores log1p(km); α=1 recovers −λ·log1p(km).
        gsb = float(getattr(self.cfg, "geo_score_bias", 0.0) or 0.0)
        if gsb > 0 and count_ctx is not None and "delta_geo" in count_ctx:
            hav = count_ctx["delta_geo"][:, :rows.shape[1], 0].to(score.device).clamp_min(0)
            geo_pow = float(getattr(self.cfg, "geo_score_power", 1.0) or 1.0)
            if abs(geo_pow - 1.0) < 1e-8:
                geo_pen = hav
            else:
                km = torch.expm1(hav).clamp_min(0)
                geo_pen = torch.log1p(km.pow(geo_pow))
            if spatial_scale is not None:
                geo_pen = geo_pen * spatial_scale
            score = score - gsb * geo_pen
        # Gaussian spatial kernel: λ·exp(−km²/(2σ²)); prefer replace flat geo.
        ggb = float(getattr(self.cfg, "gauss_geo_bias", 0.0) or 0.0)
        if ggb > 0 and count_ctx is not None and "delta_geo" in count_ctx:
            hav = count_ctx["delta_geo"][:, :rows.shape[1], 0].to(score.device).clamp_min(0)
            km = torch.expm1(hav).clamp_min(0)
            sigma = float(getattr(self.cfg, "gauss_sigma_km", 0.5) or 0.5)
            gauss = torch.exp(-(km * km) / (2.0 * sigma * sigma))
            if spatial_scale is not None:
                gauss = gauss * spatial_scale
            score = score + ggb * gauss
        # Radiation (intervening-opportunity) prior; prefer replace flat geo.
        rad_b = float(getattr(self.cfg, "radiation_score_bias", 0.0) or 0.0)
        if (rad_b > 0 and count_ctx is not None and "delta_geo" in count_ctx
                and "cand_poi_feat" in count_ctx):
            hav = count_ctx["delta_geo"][:, :rows.shape[1], 0].to(score.device).clamp_min(0)
            km = torch.expm1(hav).clamp_min(0)
            # cand_poi_feat dim1 = log1p(global visits) → mass m_j
            m_j = torch.expm1(
                count_ctx["cand_poi_feat"][:, :rows.shape[1], 1].to(score.device).clamp_min(0)
            ).clamp_min(0)
            # Origin mass m_i from last history tip global visits (fallback: batch median).
            if "hist_poi_feat" in count_ctx:
                last = valid.long().sum(-1).clamp_min(1) - 1
                hist_g = count_ctx["hist_poi_feat"][:, :valid.shape[1], 1].to(score.device)
                m_i_log = hist_g.gather(1, last.view(-1, 1)).squeeze(1).clamp_min(0)
                m_i = torch.expm1(m_i_log).clamp_min(0)
            else:
                m_i = m_j.median(dim=1).values.clamp_min(0)
            rad = self._radiation_log_score(
                km, m_j, m_i,
                kernel=getattr(self.cfg, "radiation_kernel", "hard"),
                soft_km=float(getattr(self.cfg, "radiation_soft_km", 0.25) or 0.25))
            if spatial_scale is not None:
                rad = rad * spatial_scale
            score = score + rad_b * rad
        # Jump-length prior: λ·log P(Δr); prefer replace flat geo.
        jlb = float(getattr(self.cfg, "jump_length_bias", 0.0) or 0.0)
        if jlb > 0 and count_ctx is not None and "delta_geo" in count_ctx:
            hav = count_ctx["delta_geo"][:, :rows.shape[1], 0].to(score.device).clamp_min(0)
            km = torch.expm1(hav).clamp_min(0)
            jump = self._jump_log_density(
                km,
                dist=getattr(self.cfg, "jump_dist", "lognormal"),
                mu=float(getattr(self.cfg, "jump_mu", 0.0) or 0.0),
                sigma=float(getattr(self.cfg, "jump_sigma", 1.0) or 1.0),
                beta=float(getattr(self.cfg, "jump_beta", 1.5) or 1.5),
                d0=float(getattr(self.cfg, "jump_d0", 0.1) or 0.1))
            if spatial_scale is not None:
                jump = jump * spatial_scale
            score = score + jlb * jump
        # Category-stage: λ·log1p(last_cat→cand_cat) soft prior (needs cat_stage_count).
        csb_stage = float(getattr(self.cfg, "cat_stage_bias", 0.0) or 0.0)
        if csb_stage > 0 and count_ctx is not None and "cat_stage_count" in count_ctx:
            stage = count_ctx["cat_stage_count"][:, :rows.shape[1]].to(score.device).clamp_min(0)
            score = score + csb_stage * torch.log1p(stage)
        # Revisit prior: user→cand causal visit mass (cand_poi_feat dim 0 = log1p(user)).
        # When return_gate=on: mix π·(λ_ret·revisit) + (1−π)·(λ_exp·explore spatial).
        rsb = float(getattr(self.cfg, "revisit_score_bias", 0.0) or 0.0)
        use_return_gate = getattr(self.cfg, "return_gate", "off") == "on"
        if (rsb > 0 and count_ctx is not None and self.cfg.causal_counts
                and "cand_poi_feat" in count_ctx):
            u_vis = count_ctx["cand_poi_feat"][:, :rows.shape[1], 0].to(score.device).clamp_min(0)
            if use_return_gate:
                last = valid.long().sum(-1).clamp_min(1) - 1
                last_emb = encoded["P"].gather(
                    1, last.view(-1, 1, 1).expand(-1, 1, encoded["P"].shape[-1])
                ).squeeze(1)
                # User visit richness: mean log1p(user→hist) over valid tips.
                if "hist_poi_feat" in count_ctx:
                    hist_u = count_ctx["hist_poi_feat"][:, :valid.shape[1], 0].to(score.device)
                    hist_u = hist_u.masked_fill(~valid, 0)
                    denom = valid.float().sum(-1).clamp_min(1.0)
                    richness = (hist_u.sum(-1) / denom).unsqueeze(-1)
                else:
                    richness = u_vis.mean(dim=1, keepdim=True)
                pi = torch.sigmoid(self.return_gate_mlp(
                    torch.cat([last_emb, richness], dim=-1))).squeeze(-1)  # [B]
                ret_term = rsb * u_vis
                exp_b = float(getattr(self.cfg, "explore_score_bias", 0.0) or 0.0)
                exp_mode = getattr(self.cfg, "explore_spatial_bias", "off")
                exp_term = score.new_zeros(score.shape)
                if exp_b > 0 and exp_mode != "off" and "delta_geo" in count_ctx:
                    hav = count_ctx["delta_geo"][:, :rows.shape[1], 0].to(score.device).clamp_min(0)
                    km = torch.expm1(hav).clamp_min(0)
                    if exp_mode == "gauss":
                        sigma = float(getattr(self.cfg, "gauss_sigma_km", 0.5) or 0.5)
                        spatial = torch.exp(-(km * km) / (2.0 * sigma * sigma))
                    elif exp_mode == "geo":
                        spatial = -hav  # −log1p(km); closer → higher
                    else:  # radiation
                        m_j = torch.expm1(
                            count_ctx["cand_poi_feat"][:, :rows.shape[1], 1]
                            .to(score.device).clamp_min(0)).clamp_min(0)
                        if "hist_poi_feat" in count_ctx:
                            hist_g = count_ctx["hist_poi_feat"][:, :valid.shape[1], 1].to(score.device)
                            m_i_log = hist_g.gather(1, last.view(-1, 1)).squeeze(1).clamp_min(0)
                            m_i = torch.expm1(m_i_log).clamp_min(0)
                        else:
                            m_i = m_j.median(dim=1).values.clamp_min(0)
                        spatial = self._radiation_log_score(
                            km, m_j, m_i,
                            kernel=getattr(self.cfg, "radiation_kernel", "hard"),
                            soft_km=float(getattr(self.cfg, "radiation_soft_km", 0.25) or 0.25))
                    if spatial_scale is not None:
                        spatial = spatial * spatial_scale
                    exp_term = exp_b * spatial
                score = score + pi[:, None] * ret_term + (1.0 - pi)[:, None] * exp_term
            else:
                score = score + rsb * u_vis
        # Familiar-area soft blend (#44): f = fraction of hist tips within fam_radius of last.
        # score += (1−f)·λ_exp·gauss + f·λ_ret·revisit_mass (revisit λ locked separately).
        if (getattr(self.cfg, "fam_area_blend", "off") == "on"
                and count_ctx is not None and "delta_geo" in count_ctx):
            # Familiarity proxy: mean user→hist visit mass (squashed). Optional tip-km
            # filter when hist_tip_km_to_last is supplied by the feature gatherer.
            hist_mask = valid
            if "hist_poi_feat" in count_ctx:
                hist_u = count_ctx["hist_poi_feat"][:, :valid.shape[1], 0].to(score.device)
                hist_u = hist_u.masked_fill(~hist_mask, 0)
            else:
                hist_u = score.new_zeros(valid.shape)
            tip_km = count_ctx.get("hist_tip_km_to_last")
            if tip_km is not None:
                tip_km = tip_km[:, :valid.shape[1]].to(score.device)
                local = hist_mask & (tip_km <= float(self.cfg.fam_radius_km))
            else:
                local = hist_mask
            local_n = local.float().sum(-1).clamp_min(1.0)
            local_mass = (hist_u * local.float()).sum(-1) / local_n
            f = (1.0 - torch.exp(-local_mass)).clamp(0, 1).unsqueeze(-1)  # [B,1]
            hav = count_ctx["delta_geo"][:, :rows.shape[1], 0].to(score.device).clamp_min(0)
            km = torch.expm1(hav).clamp_min(0)
            sigma = float(getattr(self.cfg, "gauss_sigma_km", 0.5) or 0.5)
            gauss = torch.exp(-(km * km) / (2.0 * sigma * sigma))
            feb = float(getattr(self.cfg, "fam_explore_bias", 0.0) or 0.0)
            frb = float(getattr(self.cfg, "fam_return_bias", 0.0) or 0.0)
            if feb > 0:
                score = score + (1.0 - f) * feb * gauss
            if frb > 0 and "cand_poi_feat" in count_ctx:
                u_vis = count_ctx["cand_poi_feat"][:, :rows.shape[1], 0].to(score.device).clamp_min(0)
                score = score + f * frb * u_vis
        # History-hit / revisit-recency: orthogonal to flat revisit λ (needs hist_poi_rows).
        hhb = float(getattr(self.cfg, "history_hit_bias", 0.0) or 0.0)
        rrb = float(getattr(self.cfg, "revisit_recency_bias", 0.0) or 0.0)
        if ((hhb > 0 or rrb > 0) and count_ctx is not None
                and "hist_poi_rows" in count_ctx):
            hist_rows = count_ctx["hist_poi_rows"][:, :valid.shape[1]].to(score.device)
            hist_ok = valid & (hist_rows >= 0)
            if hhb > 0:
                k_hit = max(1, int(getattr(self.cfg, "history_hit_k", 5) or 5))
                lengths = valid.long().sum(-1)
                hit = torch.zeros_like(score, dtype=torch.bool)
                for o in range(k_hit):
                    tip_idx = (lengths - 1 - o).clamp_min(0)
                    tip = hist_rows.gather(1, tip_idx.view(-1, 1)).squeeze(1)
                    tip_ok = (lengths > o) & (tip >= 0)
                    hit = hit | ((rows == tip[:, None]) & tip_ok[:, None])
                score = score + hhb * hit.to(score.dtype)
            if rrb > 0:
                # age = steps from tip to most recent hist match; never → no boost.
                tau = float(getattr(self.cfg, "revisit_recency_tau", 3.0) or 3.0)
                bsz, n_hist = hist_rows.shape
                n_cand = rows.shape[1]
                pos = torch.arange(n_hist, device=score.device).view(1, n_hist, 1)
                tip = (valid.long().sum(-1) - 1).clamp_min(0).view(-1, 1, 1)
                match = (
                    (hist_rows.unsqueeze(2) == rows.unsqueeze(1))
                    & hist_ok.unsqueeze(2))
                age = (tip - pos).to(score.dtype).masked_fill(~match, 1e4)
                min_age = age.min(dim=1).values.clamp_min(0)
                seen = min_age < 1e3
                recency = torch.exp(-min_age / tau) * seen.to(score.dtype)
                score = score + rrb * recency
        # Direct SURD/SMI score contribution (default λ=0).
        ssb = float(getattr(self.cfg, "surd_score_bias", 0.0) or 0.0)
        if ssb > 0 and cand_gate is not None:
            score = score + ssb * cand_gate
        score = score.masked_fill(~mask, -torch.inf)
        result = dict(score=score, log_pi=log_pi, top=top, weights=weights, gate_logits=gate_logits,
                      gate=gate, mass=mass, alpha=alpha, confidence=w, mask=mask,
                      router_mode=mode, surd_gate=cand_gate)
        if details:
            result.update(base=base, learned=learned, bias=bias, logits=logits, projected=projected,
                          contributions=gate[..., None] * weights * projected,
                          moe_bias=gate * self.moe_head.bias)
        return result

    def cat_aux_logits(self, encoded, valid):
        """Next-category logits from last-tip mean of P/C encodings (#45)."""
        lengths = valid.long().sum(-1).clamp_min(1)
        tip = (lengths - 1).view(-1, 1, 1)
        d = encoded["P"].size(-1)
        zp = encoded["P"].gather(1, tip.expand(-1, 1, d)).squeeze(1)
        zc = encoded["C"].gather(1, tip.expand(-1, 1, d)).squeeze(1)
        return self.cat_aux_head(0.5 * (zp + zc))

    def forward(self, history, valid, rows, prior, confidence, **kwargs):
        count_ctx = kwargs.pop("count_ctx", None)
        surd_specific = kwargs.pop("surd_specific", None)
        # Catalog rows for history tips (token−2); used by history-hit / recency biases.
        if count_ctx is not None and "P" in history:
            count_ctx = dict(count_ctx)
            count_ctx["hist_poi_rows"] = history["P"].to(valid.device) - 2
        encoded = self.encode(history, valid, count_ctx)
        result = self.score(encoded, valid, rows, prior, confidence,
                            count_ctx=count_ctx, surd_specific=surd_specific, **kwargs)
        result["cat_aux_logits"] = self.cat_aux_logits(encoded, valid)
        return result


def soft_ema_targets(labels, mask, teacher_scores, beta):
    """Build ``(1−β)·onehot(y) + β·softmax(teacher)`` over masked candidates (#39)."""
    beta = float(beta or 0.0)
    if beta < 0 or beta > 1:
        raise ValueError("soft_target_beta must be in [0, 1]")
    bsz, n_cand = mask.shape
    onehot = F.one_hot(labels, num_classes=n_cand).to(dtype=torch.float32)
    if beta <= 0:
        return onehot
    logits = teacher_scores.to(dtype=torch.float32).masked_fill(~mask, float("-inf"))
    teacher_p = torch.softmax(logits, dim=1)
    teacher_p = teacher_p.masked_fill(~mask, 0.0)
    z = teacher_p.sum(dim=1, keepdim=True).clamp_min(1e-8)
    teacher_p = teacher_p / z
    mix = (1.0 - beta) * onehot + beta * teacher_p
    mix = mix.masked_fill(~mask, 0.0)
    return mix / mix.sum(dim=1, keepdim=True).clamp_min(1e-8)


def soft_prior_targets(labels, mask, count_ctx, rows, valid, priors, beta):
    """Build ``(1−β)·onehot(y) + β·normalize(Σ prior masses)`` over masked cands.

    ``priors`` is an iterable of ``\"tsb\"`` / ``\"revisit\"`` (Acc-positive only).
    ``\"ema\"`` is handled separately via ``soft_ema_targets``.
    """
    beta = float(beta or 0.0)
    if beta < 0 or beta > 1:
        raise ValueError("soft_target_beta must be in [0, 1]")
    device = labels.device
    bsz, n_cand = mask.shape
    onehot = F.one_hot(labels, num_classes=n_cand).to(dtype=torch.float32)
    if beta <= 0 or not priors:
        return onehot
    prior_mass = torch.zeros(bsz, n_cand, device=device, dtype=torch.float32)
    names = {str(p).strip().lower() for p in priors if str(p).strip()}
    names.discard("ema")  # EMA softmax is mixed outside this helper.
    unknown = names - {"tsb", "revisit"}
    if unknown:
        raise ValueError(f"unsupported soft-target priors: {sorted(unknown)}")
    if not names:
        return onehot
    if count_ctx is None:
        raise ValueError("soft-target priors require count_ctx")
    if "tsb" in names:
        if "poi_transition" not in count_ctx:
            raise ValueError("soft-target tsb requires poi_transition in count_ctx")
        # log1p counts → raw counts via expm1
        l1p = TSMoE._last_hop_log1p(count_ctx["poi_transition"], rows, valid, device)
        prior_mass = prior_mass + torch.expm1(l1p.clamp_min(0)).clamp_min(0)
    if "revisit" in names:
        if "cand_poi_feat" not in count_ctx:
            raise ValueError("soft-target revisit requires cand_poi_feat in count_ctx")
        u = count_ctx["cand_poi_feat"][:, :n_cand, 0].to(device).clamp_min(0)
        prior_mass = prior_mass + torch.expm1(u).clamp_min(0)
    prior_mass = prior_mass.masked_fill(~mask, 0.0)
    z = prior_mass.sum(dim=1, keepdim=True)
    # Rows with empty prior fall back to one-hot (no soft mix).
    has = (z.squeeze(1) > 0)
    prior = torch.zeros_like(prior_mass)
    prior[has] = prior_mass[has] / z[has].clamp_min(1e-8)
    mix = (1.0 - beta) * onehot + beta * prior
    # Keep label mass when prior was empty.
    mix = torch.where(has[:, None], mix, onehot)
    # Renormalize after mask (numerical safety).
    mix = mix.masked_fill(~mask, 0.0)
    mix = mix / mix.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return mix


def blend_soft_target_mixes(mixes, mask):
    """Average already-normalized soft target tensors, then renormalize on mask."""
    if not mixes:
        raise ValueError("blend_soft_target_mixes requires ≥1 mix")
    if len(mixes) == 1:
        return mixes[0]
    stacked = torch.stack([m.to(dtype=torch.float32) for m in mixes], dim=0).mean(dim=0)
    stacked = stacked.masked_fill(~mask, 0.0)
    return stacked / stacked.sum(dim=1, keepdim=True).clamp_min(1e-8)


def objective(result, labels, lambda_surd=.05, lambda_balance=.001, lambda_bpr=0.,
              bpr_margin=0., ce_focal_gamma=0., bpr_mode="max",
              soft_targets=None):
    """Ranking objective: CE (+ optional SURD aux/balance) + optional hard-negative BPR.

    ``lambda_bpr > 0`` adds a hard-neg aux averaged over rows with ≥1 valid
    negative. ``bpr_mode`` selects the competitor aggregation:

    * ``max`` (default): ``softplus(-(s_pos - max_neg - margin))`` — historical.
    * ``bpr_max``: GRU4Rec+ BPR-max — softplus against the softmax-weighted
      mean of negative scores.
    * ``simce``: binary CE between ``[s_pos, max_neg]`` (hardest competitor).

    ``bpr_margin`` (default 0) applies to ``max`` / ``bpr_max`` only.
    ``ce_focal_gamma > 0`` replaces plain CE with focal
    ``(1 − p_t)^γ · (−log p_t)`` (γ=0 keeps mean CE).
    ``soft_targets`` if set replaces hard CE with ``−Σ soft · log_softmax(score)``.
    Do not combine soft targets with focal γ>0.
    """
    mask, w = result["mask"], result["confidence"]
    log_pi, mass, gate = result["log_pi"], result["mass"], result["gate"]
    if not mask.gather(1, labels[:, None]).all():
        raise ValueError("every positive must be in the valid candidate set")
    scores = result["score"]
    gamma = float(ce_focal_gamma or 0.0)
    if gamma < 0:
        raise ValueError("ce_focal_gamma must be nonnegative")
    mode = str(bpr_mode or "max")
    if mode not in ("max", "bpr_max", "simce"):
        raise ValueError("bpr_mode must be max|bpr_max|simce")
    if soft_targets is not None and gamma > 0:
        raise ValueError("soft_targets cannot combine with ce_focal_gamma > 0")
    if soft_targets is not None:
        log_p = F.log_softmax(scores, dim=1)
        rec = -(soft_targets.to(scores.dtype) * log_p).sum(dim=1).mean()
    else:
        ce = F.cross_entropy(scores, labels, reduction="none")
        if gamma > 0:
            pt = (-ce).exp().clamp(max=1.0)
            rec = ((1.0 - pt).pow(gamma) * ce).mean()
        else:
            rec = ce.mean()
    weights = w * mask / mask.sum()
    bce = F.binary_cross_entropy_with_logits(result["gate_logits"], result["alpha"], reduction="none")
    aux = (weights * (bce - (mass * log_pi).sum(-1))).sum()
    difference = gate[..., None] * log_pi.exp() - mass
    balance = (weights[..., None] * difference).sum((0, 1)).square().sum()
    bpr = scores.new_zeros(())
    if float(lambda_bpr) > 0:
        s_pos = scores.gather(1, labels[:, None]).squeeze(1)
        neg_mask = mask.clone()
        neg_mask.scatter_(1, labels[:, None], False)
        rows = neg_mask.any(dim=1)
        if bool(rows.any()):
            margin = float(bpr_margin or 0.0)
            neg_logits = scores.masked_fill(~neg_mask, float("-inf"))
            if mode == "simce":
                s_neg = neg_logits.max(dim=1).values
                pair = torch.stack([s_pos[rows], s_neg[rows]], dim=1)
                bpr = F.cross_entropy(pair, torch.zeros(pair.size(0), dtype=torch.long,
                                                        device=pair.device))
            elif mode == "bpr_max":
                # Softmax over negatives only; masked cols already -inf → weight 0.
                neg_w = torch.softmax(neg_logits, dim=1)
                s_neg = (neg_w * scores.masked_fill(~neg_mask, 0.0)).sum(dim=1)
                finite = rows & torch.isfinite(s_neg)
                if bool(finite.any()):
                    bpr = F.softplus(-(s_pos[finite] - s_neg[finite] - margin)).mean()
            else:
                s_neg = neg_logits.max(dim=1).values
                finite = rows & torch.isfinite(s_neg)
                if bool(finite.any()):
                    bpr = F.softplus(-(s_pos[finite] - s_neg[finite] - margin)).mean()
    total = rec + lambda_surd * aux + lambda_balance * balance + float(lambda_bpr) * bpr
    return total, (rec.detach(), aux.detach(), balance.detach(), bpr.detach())


def top_k_kd_loss(student_scores, teacher_scores, mask, top_k=20, temp=2.0):
    """KL on teacher top-K logits (EKD4Rec-style) with temperature scaling (#41)."""
    top_k = int(top_k)
    temp = float(temp)
    if top_k < 1:
        raise ValueError("kd-top-k must be >= 1")
    if temp <= 0:
        raise ValueError("kd-temp must be positive")
    t_logits = teacher_scores.masked_fill(~mask, float("-inf"))
    n_valid = mask.sum(dim=1)
    k = min(top_k, int(n_valid.min().item()) if bool(n_valid.numel()) else top_k)
    if k < 1:
        return student_scores.new_zeros(())
    topv, topi = t_logits.topk(k, dim=1)
    s = student_scores.gather(1, topi) / temp
    tt = topv / temp
    teacher_p = torch.softmax(tt, dim=1)
    student_log_p = F.log_softmax(s, dim=1)
    return -(teacher_p * student_log_p).sum(dim=1).mean() * (temp * temp)


def poi_geo_laplace_loss(poi_emb, edge_src, edge_dst, edge_w, lambda_laplace):
    """Graph Laplacian smoothness on POI embeddings (#40): λ Σ w‖e_i−e_j‖²."""
    lam = float(lambda_laplace or 0.0)
    if lam <= 0 or edge_src.numel() == 0:
        return poi_emb.new_zeros(())
    diff = poi_emb[edge_src] - poi_emb[edge_dst]
    return lam * (edge_w * diff.pow(2).sum(dim=-1)).mean()
