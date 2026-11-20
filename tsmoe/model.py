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

    def forward(self, query, key, value, key_padding_mask, transition_counts):
        """
        query: [B, C, D] candidates · key/value: [B, L, D] history
        key_padding_mask: [B, L] True = **ignore** (same as nn.MultiheadAttention)
        transition_counts: [B, C, L] raw counts (≥0) or PMI (any real) when use_pmi
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
                logits = logits + self.transition_lambda * phi[:, None, :, :]
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

    def _cross_attend(self, encoded, valid, candidate, count_ctx=None):
        """Per-source candidate←history attention; P/C use transition modes when enabled."""
        pad_mask = ~valid
        use = (count_ctx is not None and self.cfg.causal_counts
               and "poi_transition" in count_ctx and "cat_transition" in count_ctx)
        z = []
        for s in SOURCE_ORDER:
            if use and s in self.transition_attn:
                counts = count_ctx["poi_transition" if s == "P" else "cat_transition"]
                # Align width to this candidate chunk.
                c = candidate.shape[1]
                counts = counts[:, :c, :encoded[s].shape[1]].to(candidate.device)
                z.append(self.transition_attn[s](
                    candidate, encoded[s], encoded[s], pad_mask, counts))
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
              details=False, count_ctx=None):
        mask = torch.ones_like(rows, dtype=torch.bool) if candidate_mask is None else candidate_mask
        if mask.shape != rows.shape:
            raise ValueError("candidate mask and candidate rows must have identical shapes")
        candidate, typed = self._build_candidate_vector(rows, mask, count_ctx)
        eca = self.cfg.expert_cross_attn
        # Shared cross-attn (E0 / E1). E2 (full) skips shared attn for MoE; still builds base z.
        if eca == "full":
            # Base uses typed expert-local attn over all sources (no shared transition path).
            local = self._expert_local_attend(encoded, valid, typed)
            z = [local[s] for s in SOURCE_ORDER]
        else:
            z = self._cross_attend(encoded, valid, candidate, count_ctx)
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
        prior, w = prior26.detach(), confidence.detach()
        if not self.cfg.use_surd:
            w = torch.zeros_like(w)
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
        score = (base + gate * residual).masked_fill(~mask, -torch.inf)
        result = dict(score=score, log_pi=log_pi, top=top, weights=weights, gate_logits=gate_logits,
                      gate=gate, mass=mass, alpha=alpha, confidence=w, mask=mask,
                      router_mode=mode)
        if details:
            result.update(base=base, learned=learned, bias=bias, logits=logits, projected=projected,
                          contributions=gate[..., None] * weights * projected,
                          moe_bias=gate * self.moe_head.bias)
        return result

    def forward(self, history, valid, rows, prior, confidence, **kwargs):
        count_ctx = kwargs.pop("count_ctx", None)
        return self.score(self.encode(history, valid, count_ctx), valid, rows, prior, confidence,
                          count_ctx=count_ctx, **kwargs)


def objective(result, labels, lambda_surd=.05, lambda_balance=.001):
    mask, w = result["mask"], result["confidence"]
    log_pi, mass, gate = result["log_pi"], result["mass"], result["gate"]
    if not mask.gather(1, labels[:, None]).all():
        raise ValueError("every positive must be in the valid candidate set")
    rec = F.cross_entropy(result["score"], labels)
    weights = w * mask / mask.sum()
    bce = F.binary_cross_entropy_with_logits(result["gate_logits"], result["alpha"], reduction="none")
    aux = (weights * (bce - (mass * log_pi).sum(-1))).sum()
    difference = gate[..., None] * log_pi.exp() - mass
    balance = (weights[..., None] * difference).sum((0, 1)).square().sum()
    return rec + lambda_surd * aux + lambda_balance * balance, (rec.detach(), aux.detach(), balance.detach())
