"""Target-conditioned Transformer, sparse U/S experts and SURD-guided objective."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .data import L_DIM, T_DIM
from .surd import SOURCE_ORDER, SUBSETS

ROUTER_MODES = ("hard_prior", "surd_mlp", "hybrid")


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

    def __post_init__(self):
        if self.dim < 1 or self.heads < 1 or self.dim % self.heads or self.layers < 1:
            raise ValueError("positive hidden dimension must be divisible by heads; layers must be positive")
        if not 1 <= self.top_k <= 15 or self.temperature <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("invalid top_k, temperature or dropout")
        if self.router_mode not in ROUTER_MODES:
            raise ValueError(f"router_mode must be one of {ROUTER_MODES}")
        if self.hybrid_lambda < 0:
            raise ValueError("hybrid_lambda must be nonnegative")


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


class TSMoE(nn.Module):
    def __init__(self, n_pois, n_categories, max_length, config):
        super().__init__()
        self.cfg = config
        d = config.dim
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

    def encode(self, history, valid):
        position = self.positions(torch.arange(valid.shape[1], device=valid.device))
        return {s: self.encoders[s](self.projections[s](history[s]) + position,
                                    src_key_padding_mask=~valid) for s in SOURCE_ORDER}

    def dispatch(self, z, indices, weights, mask, details):
        """Group valid (batch,candidate,slot) assignments by expert before each FFN."""
        d, k = self.cfg.dim, indices.shape[-1]
        flat_z = [v.reshape(-1, d) for v in z]
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

    def score(self, encoded, valid, rows, prior26, confidence, *, candidate_mask=None, details=False):
        mask = torch.ones_like(rows, dtype=torch.bool) if candidate_mask is None else candidate_mask
        if mask.shape != rows.shape:
            raise ValueError("candidate mask and candidate rows must have identical shapes")
        candidate = self.projections["P"](rows.masked_fill(~mask, -1) + 2)
        z = [self.attention[s](candidate, encoded[s], encoded[s], key_padding_mask=~valid,
                               need_weights=False)[0] for s in SOURCE_ORDER]
        full = torch.cat([*z, candidate], -1)
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
        residual, projected = self.dispatch(z, top, weights, mask, details)
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
        return self.score(self.encode(history, valid), valid, rows, prior, confidence, **kwargs)


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
