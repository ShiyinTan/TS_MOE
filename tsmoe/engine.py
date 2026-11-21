"""Training, exact catalog ranking, and candidate-level case study output."""
import copy
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .candidates import (
    format_pool_size_line, haversine_km, mix_adaptive_negatives,
    pools_for_split, sample_candidates, shrink_pools_spatial_prior,
    summarize_pool_sizes,
)
from .counts import (
    DEFAULT_VISIT_BINS, CausalCountTables, chronological_indices, gather_batch_context,
    parse_dt_bin_edges,
)
from .data import FEATURE_METHOD, Categories, build_split, file_digest, make_batch, read_split
from .features import (
    build_poi_cat_table, gather_side_features, merge_feature_context, slice_side_ctx,
)
from .model import (
    ModelConfig, TSMoE, blend_soft_target_mixes, objective, poi_geo_laplace_loss,
    soft_ema_targets, soft_prior_targets, top_k_kd_loss,
)
from .prepare import digest_json, dump_json, verify_files
from .runtime import check_interrupt, install_interrupt_handler
from .surd import ATOM_ORDER, EXPERT_ORDER, SOURCE_ORDER, FrozenSURDTable


def apply_eval_bias_overrides(model, args):
    """Replace frozen score biases at evaluate-time (locked-bias ε-mix / #38).

    ``None`` keeps the checkpoint value. Biases are additive outside the neural net,
    so overriding ``geo_score_bias`` / ``revisit_score_bias`` equals residual micro-mix
    on frozen logits without retraining.
    """
    updates = {}
    geo = getattr(args, "geo_score_bias", None)
    if geo is not None:
        geo = float(geo)
        if geo < 0:
            raise ValueError("--geo-score-bias override must be nonnegative")
        updates["geo_score_bias"] = geo
    rev = getattr(args, "revisit_score_bias", None)
    if rev is not None:
        rev = float(rev)
        if rev < 0:
            raise ValueError("--revisit-score-bias override must be nonnegative")
        updates["revisit_score_bias"] = rev
    if not updates:
        return model.cfg
    new_cfg = replace(model.cfg, **updates)
    model.cfg = new_cfg
    print(
        "eval_bias_override "
        + " ".join(f"{k}={v}" for k, v in sorted(updates.items())),
        flush=True,
    )
    return new_cfg


def parse_tta_modes(raw):
    """Parse ``--tta`` CSV into modes; empty / off → ()."""
    text = str(raw or "").strip().lower()
    if text in ("", "off", "none"):
        return ()
    modes = tuple(p.strip() for p in text.split(",") if p.strip())
    allowed = {"mask", "substitute"}
    unknown = set(modes) - allowed
    if unknown:
        raise ValueError(f"--tta modes must be mask|substitute; got {sorted(unknown)}")
    return modes


def make_tta_history_views(history, valid, modes, *, keep_last=True, n_views=1, seed=0):
    """Build Mask / Substitute history views (last tip fixed). Yields (history, valid)."""
    if not modes:
        return
    bsz, seq = valid.shape
    device = valid.device
    lengths = valid.long().sum(dim=1)
    for view_i in range(max(1, int(n_views))):
        for mode in modes:
            g = torch.Generator(device=device)
            g.manual_seed(int(seed) + 17 * view_i + (0 if mode == "mask" else 101))
            h = {k: v.clone() for k, v in history.items()}
            v = valid.clone()
            for b in range(bsz):
                L = int(lengths[b].item())
                if L <= 1:
                    continue
                last = L - 1
                editable = list(range(0 if not keep_last else 0, last))
                if not editable:
                    continue
                if mode == "mask":
                    # Drop ~40% of non-last tips (at least one when L>=3).
                    n_drop = max(1, int(round(0.4 * len(editable)))) if L >= 3 else 1
                    n_drop = min(n_drop, len(editable))
                    pick = torch.randperm(len(editable), generator=g, device=device)[:n_drop]
                    for j in pick.tolist():
                        v[b, editable[j]] = False
                elif mode == "substitute":
                    # Replace ~40% non-last P tokens with another tip from same trail.
                    n_sub = max(1, int(round(0.4 * len(editable)))) if L >= 3 else 1
                    n_sub = min(n_sub, len(editable))
                    pick = torch.randperm(len(editable), generator=g, device=device)[:n_sub]
                    src = torch.randint(0, L, (n_sub,), generator=g, device=device)
                    for k, j in enumerate(pick.tolist()):
                        src_i = int(src[k].item())
                        if src_i == editable[j]:
                            src_i = last
                        h["P"][b, editable[j]] = history["P"][b, src_i]
            # Ensure last tip remains valid.
            if keep_last:
                for b in range(bsz):
                    L = int(lengths[b].item())
                    if L >= 1:
                        v[b, L - 1] = True
            yield h, v


def apply_score_calib(scores, pool_rows, pool_mask, count_ctx, mode, calib_lambda):
    """Acc-aware non-order-preserving score calib (#34).

    ``gauss_pop``: Gaussian-CDF of centered scores minus λ·log1p(global pop).
    ``gauss_rev``: same CDF plus λ·log1p(user revisit). Univariate CDF alone is
    Acc-neutral; the feature term can change top-1.
    """
    mode = str(mode or "off").lower()
    if mode in ("", "off", "none"):
        return scores
    if mode not in ("gauss_pop", "gauss_rev"):
        raise ValueError("--score-calib must be off|gauss_pop|gauss_rev")
    lam = float(calib_lambda or 0.0)
    if abs(lam) < 1e-12:
        return scores
    if count_ctx is None or "cand_poi_feat" not in count_ctx:
        raise ValueError(f"--score-calib {mode} requires causal cand_poi_feat")
    feat_dim = 1 if mode == "gauss_pop" else 0
    feat = count_ctx["cand_poi_feat"][..., feat_dim].to(scores.device).to(scores.dtype)
    # Align feature width to padded pool scores.
    bsz, width = scores.shape
    mask = pool_mask.to(device=scores.device, dtype=torch.bool)
    if feat.shape != scores.shape:
        padded = torch.zeros(bsz, width, dtype=scores.dtype, device=scores.device)
        n = min(feat.shape[1], width)
        padded[:, :n] = feat[:, :n]
        feat = padded
    # Median over valid pool slots.
    meds = []
    for b in range(bsz):
        vals = scores[b][mask[b]]
        if vals.numel() == 0:
            meds.append(scores.new_zeros(()))
        else:
            meds.append(vals.median())
    med = torch.stack(meds).view(bsz, 1)
    centered = (scores - med).masked_fill(~mask, 0.0)
    denom = mask.to(scores.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    var = centered.pow(2).sum(dim=1, keepdim=True) / denom
    scale = var.sqrt().clamp_min(1e-3)
    # Φ approx via logistic; order-preserving alone, then ±λ·feat flips Acc.
    cdf = torch.sigmoid(1.702 * centered / scale)
    if mode == "gauss_pop":
        out = cdf - lam * feat
    else:
        out = cdf + lam * feat
    return out.masked_fill(~mask, float("-inf"))


def model_config_from_checkpoint(raw):
    """Build ModelConfig from a checkpoint; legacy saves without router_mode → hybrid."""
    cfg = dict(raw)
    if "router_mode" not in cfg:
        # Pre-router-mode checkpoints used sequence×candidate router + λ_prior·SURD bias
        # with an implicit learned scale of 1 (logits = learned + λ_prior · bias).
        cfg["router_mode"] = "hybrid"
        cfg.setdefault("hybrid_lambda", 1.0)
    # Pre-causal-count checkpoints lack the new modules → keep the feature off.
    if "causal_counts" not in cfg:
        cfg["causal_counts"] = False
    cfg.setdefault("transition_attn", "bias")
    cfg.setdefault("transition_lambda", 1.0)
    cfg.setdefault("transition_score_bias", 0.0)
    cfg.setdefault("category_score_bias", 0.0)
    cfg.setdefault("pop_score_bias", 0.0)
    cfg.setdefault("geo_score_bias", 0.0)
    cfg.setdefault("geo_score_power", 1.0)
    cfg.setdefault("gauss_geo_bias", 0.0)
    cfg.setdefault("gauss_sigma_km", 0.5)
    cfg.setdefault("geo_adapt", "off")
    cfg.setdefault("geo_adapt_temp", 1.0)
    cfg.setdefault("revisit_score_bias", 0.0)
    cfg.setdefault("history_hit_bias", 0.0)
    cfg.setdefault("history_hit_k", 5)
    cfg.setdefault("revisit_recency_bias", 0.0)
    cfg.setdefault("revisit_recency_tau", 3.0)
    cfg.setdefault("radiation_score_bias", 0.0)
    cfg.setdefault("radiation_kernel", "hard")
    cfg.setdefault("radiation_soft_km", 0.25)
    cfg.setdefault("jump_length_bias", 0.0)
    cfg.setdefault("jump_dist", "lognormal")
    cfg.setdefault("jump_mu", 0.0)
    cfg.setdefault("jump_sigma", 1.0)
    cfg.setdefault("jump_beta", 1.5)
    cfg.setdefault("jump_d0", 0.1)
    cfg.setdefault("return_gate", "off")
    cfg.setdefault("explore_spatial_bias", "off")
    cfg.setdefault("explore_score_bias", 0.0)
    cfg.setdefault("fam_area_blend", "off")
    cfg.setdefault("fam_radius_km", 0.5)
    cfg.setdefault("fam_explore_bias", 0.0)
    cfg.setdefault("fam_return_bias", 0.0)
    cfg.setdefault("tsb_history_k", 1)
    cfg.setdefault("tsb_history_decay", 0.5)
    cfg.setdefault("tsb_dt_edges", ())
    cfg.setdefault("cat_stage_bias", 0.0)
    cfg.setdefault("transition_2hop_bias", 0.0)
    cfg.setdefault("hour_pop_score_bias", 0.0)
    cfg.setdefault("tsb_decay_hours", 0.0)
    cfg.setdefault("tsb_session_gap_hours", 0.0)
    cfg.setdefault("visit_bins", DEFAULT_VISIT_BINS)
    # New overlays default off for legacy checkpoints.
    cfg.setdefault("candidate_context", "off")
    cfg.setdefault("candidate_context_time", "off")
    cfg.setdefault("session_summary", "off")
    cfg.setdefault("user_pref", "off")
    cfg.setdefault("user_id_emb", "off")
    cfg.setdefault("transition_pmi", "off")
    cfg.setdefault("expert_cross_attn", "off")
    cfg.setdefault("expert_cross_attn_sparse", "on")
    # SURD/SMI bias gates default off for legacy checkpoints.
    cfg.setdefault("surd_bias_gate", "off")
    cfg.setdefault("surd_score_bias", 0.0)
    cfg.setdefault("tsb_gate_mode", "off")
    cfg.setdefault("surd_attn_gate", "off")
    return ModelConfig(**cfg)


def needs_surd_specific(cfg):
    """True when score/attn path needs Specific MI from the SURD table."""
    mode = getattr(cfg, "surd_bias_gate", "off")
    return mode in ("smi_p", "smi_pc")


def lookup_surd_specific(table, rows, cfg):
    """Return Specific MI tensor for ``rows`` when the gate mode needs it."""
    if not needs_surd_specific(cfg):
        return None
    return table.specific(rows)


_COUNT_WEIGHT_PREFIXES = (
    "poi_count_mlp.", "cat_count_mlp.", "visit_bin_emb.", "transition_attn.",
    "cand_geo_mlp.", "cand_time_mlp.", "cand_slot_mlp.", "cand_fuse.",
    "session_length_emb.", "session_cat_proj.", "session_scalar_mlp.",
    "session_mlp.", "session_to_full.",
    "user_cat_proj.", "user_hour_proj.", "user_pref_mlp.", "user_pref_to_full.",
    "user_emb.", "user_id_to_full.",
    "expert_attn.", "expert_query_proj.",
    "return_gate_mlp.", "cat_aux_head.",
)


def load_weights(model, weights):
    """Load weights; allow missing surd_* / causal-count modules on legacy checkpoints."""
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected:
        raise ValueError(f"checkpoint has unexpected keys: {sorted(unexpected)[:8]}")
    allowed = [
        k for k in missing
        if k.startswith("surd_router.") or k.startswith("surd_gate.")
        or any(k.startswith(p) for p in _COUNT_WEIGHT_PREFIXES)
    ]
    if len(allowed) != len(missing):
        raise ValueError(f"checkpoint missing required keys: "
                         f"{sorted(set(missing) - set(allowed))[:8]}")
    return missing


def count_ctx_to_device(ctx, device):
    if ctx is None:
        return None
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ctx.items()}


def attach_target_gap_hours(ctx, data, indices):
    """Hours between last history tip and target event (for tsb time-decay)."""
    idxs = indices.tolist() if torch.is_tensor(indices) else list(indices)
    gaps = []
    for sample_index in idxs:
        end = int(data["end"][sample_index])
        start = int(data["start"][sample_index])
        if end <= start:
            gaps.append(0.0)
            continue
        tip_us = float(data["us"][end - 1])
        tgt_us = float(data["us"][end])
        gaps.append(max(0.0, (tgt_us - tip_us) / 3_600_000_000.0))  # us → hours
    gap_t = torch.tensor(gaps, dtype=torch.float32)
    if ctx is None:
        return dict(target_gap_hours=gap_t)
    out = dict(ctx)
    out["target_gap_hours"] = gap_t
    return out


def attach_cat_stage_counts(ctx, tables, data, indices):
    """last_cat→cand_cat raw counts [B,M] for --cat-stage-bias (needs cand_cat_rows)."""
    from .counts import history_catalog_rows
    if ctx is None or "cand_cat_rows" not in ctx or tables is None:
        return ctx
    cat_rows = ctx["cand_cat_rows"]
    b, m = cat_rows.shape
    out = torch.zeros(b, m, dtype=torch.float32)
    idxs = indices.tolist() if torch.is_tensor(indices) else list(indices)
    for bi, sample_index in enumerate(idxs):
        _hp, hc = history_catalog_rows(data, sample_index)
        if not hc:
            continue
        last_c = int(hc[-1])
        for j in range(m):
            cc = int(cat_rows[bi, j].item())
            if cc >= 0 and last_c >= 0:
                out[bi, j] = tables.cat_cat_count(last_c, cc)
    merged = dict(ctx)
    merged["cat_stage_count"] = out
    return merged


def build_transition_a2_log1p(train_data, n_pois):
    """Train-only dense ``log1p(A²)`` for 2-hop transition bias (#4)."""
    n = int(n_pois)
    A = torch.zeros(n, n, dtype=torch.float32)
    trail = train_data["trail"]
    poi = train_data["poi_token"]
    for i in range(len(poi) - 1):
        if int(trail[i]) != int(trail[i + 1]):
            continue
        a = int(poi[i]) - 2
        b = int(poi[i + 1]) - 2
        if 0 <= a < n and 0 <= b < n:
            A[a, b] += 1.0
    # A² via matmul; log1p for score-scale stability.
    return torch.log1p(A @ A)


def attach_transition_2hop(ctx, a2_log1p, data, indices, candidate_rows_list):
    """Gather ``log1p(A²)[last, cand]`` → [B,M] (#4)."""
    from .counts import history_catalog_rows
    if a2_log1p is None:
        return ctx
    idxs = indices.tolist() if torch.is_tensor(indices) else list(indices)
    width = max((len(r) for r in candidate_rows_list), default=0)
    out = torch.zeros(len(idxs), width, dtype=torch.float32)
    n = a2_log1p.shape[0]
    for bi, sample_index in enumerate(idxs):
        hp, _hc = history_catalog_rows(data, sample_index)
        if not hp:
            continue
        last = int(hp[-1])
        if not (0 <= last < n):
            continue
        rows = candidate_rows_list[bi]
        if not isinstance(rows, list):
            rows = list(rows) if not torch.is_tensor(rows) else rows.tolist()
        for j, r in enumerate(rows):
            rr = int(r)
            if 0 <= rr < n:
                out[bi, j] = float(a2_log1p[last, rr])
    merged = {} if ctx is None else dict(ctx)
    merged["transition_2hop_log1p"] = out
    return merged


def attach_hour_pop(ctx, tables, data, indices, candidate_rows_list):
    """Gather ``log1p(global_poi_hour[cand, tip_hour])`` → [B,M] (#8)."""
    from .features import us_to_hour_weekday
    if tables is None:
        return ctx
    idxs = indices.tolist() if torch.is_tensor(indices) else list(indices)
    width = max((len(r) for r in candidate_rows_list), default=0)
    out = torch.zeros(len(idxs), width, dtype=torch.float32)
    gph = tables.global_poi_hour  # [n_pois, 24]
    n = int(gph.shape[0])
    for bi, sample_index in enumerate(idxs):
        tip = None
        start = int(data["start"][sample_index])
        end = int(data["end"][sample_index])
        if end > start:
            tip = end - 1
        if tip is None:
            continue
        hour, _wd = us_to_hour_weekday(int(data["us"][tip]))
        h = int(hour) % 24
        rows = candidate_rows_list[bi]
        if not isinstance(rows, list):
            rows = list(rows) if not torch.is_tensor(rows) else rows.tolist()
        for j, r in enumerate(rows):
            rr = int(r)
            if 0 <= rr < n:
                out[bi, j] = float(torch.log1p(gph[rr, h].clamp_min(0)))
    merged = {} if ctx is None else dict(ctx)
    merged["hour_pop_log1p"] = out
    return merged


def slice_count_ctx(ctx, start, end):
    """Slice candidate-aligned count / side features for a scoring chunk."""
    if ctx is None:
        return None
    # Prefer the general side-aware slicer (keeps hist_* / prefs / session intact).
    return slice_side_ctx(ctx, start, end)


def needs_side_features(cfg):
    return (
        cfg.candidate_context == "on"
        or cfg.candidate_context_time != "off"
        or cfg.session_summary == "on"
        or cfg.user_pref == "on"
        or getattr(cfg, "user_id_emb", "off") == "on"
    )


def ensure_poi_cat_table(cache, train_data=None):
    """Mode category per POI from train (cached on the Cache object)."""
    if getattr(cache, "poi_cat_table", None) is not None:
        return cache.poi_cat_table
    data = train_data if train_data is not None else cache.split("train")
    cache.poi_cat_table = build_poi_cat_table(data, len(cache.pois))
    return cache.poi_cat_table


def build_batch_feature_context(
    cfg, cache, data, indices, candidate_rows_list, seq_len, *,
    count_tables=None, update=True, train_data=None,
):
    """Gather causal counts (optional) + A/B/C side features into one ctx dict."""
    ctx = None
    use_counts = count_tables is not None and cfg.causal_counts
    transition_pmi = cfg.transition_pmi == "on"
    if use_counts:
        ctx = gather_batch_context(
            count_tables, data, indices, candidate_rows_list, seq_len,
            update=update, transition_pmi=transition_pmi)
    elif needs_side_features(cfg) and cfg.user_pref == "on":
        # user-pref requires causal tables; skip silently if unavailable.
        pass
    if needs_side_features(cfg):
        poi_cat = ensure_poi_cat_table(cache, train_data=train_data)
        # When counts already updated above, pass tables for causal_slot / user_pref reads
        # that must see the pre-update state — gather_side does not update.
        # If we already updated in gather_batch_context, user_pref/causal_slot would see
        # post-update tables. Fix: gather side features BEFORE update when both needed.
        # Caller should use gather_features_causal for the joint path.
        side = gather_side_features(
            data, indices, candidate_rows_list,
            poi_coords=cache.poi_coords,
            poi_cat_table=poi_cat,
            n_categories=len(cache.meta["cat_vocab"]),
            max_length=seq_len,
            tables=count_tables,
            candidate_context=cfg.candidate_context == "on",
            candidate_context_time=cfg.candidate_context_time,
            session_summary=cfg.session_summary == "on",
            user_pref=cfg.user_pref == "on" and count_tables is not None,
            user_id_emb=getattr(cfg, "user_id_emb", "off") == "on",
            user_vocab=cache.user_vocab() if getattr(cfg, "user_id_emb", "off") == "on" else None,
            transition_pmi=transition_pmi,
        )
        ctx = merge_feature_context(ctx, side)
    need_gap = (
        float(getattr(cfg, "tsb_decay_hours", 0.0) or 0.0) > 0
        or bool(getattr(cfg, "tsb_dt_edges", ()) or ()))
    if need_gap:
        ctx = attach_target_gap_hours(ctx, data, indices)
    if float(getattr(cfg, "cat_stage_bias", 0.0) or 0.0) > 0 and count_tables is not None:
        ctx = attach_cat_stage_counts(ctx, count_tables, data, indices)
    return ctx


def gather_features_causal(
    cfg, cache, data, indices, candidate_rows_list, seq_len, count_tables, *,
    update=True, train_data=None,
):
    """Strict causal: read counts+side at current table state, then optional update."""
    transition_pmi = cfg.transition_pmi == "on"
    ctx = None
    if cfg.causal_counts and count_tables is not None:
        ctx = gather_batch_context(
            count_tables, data, indices, candidate_rows_list, seq_len,
            update=False, transition_pmi=transition_pmi)
    if needs_side_features(cfg):
        poi_cat = ensure_poi_cat_table(cache, train_data=train_data)
        side = gather_side_features(
            data, indices, candidate_rows_list,
            poi_coords=cache.poi_coords,
            poi_cat_table=poi_cat,
            n_categories=len(cache.meta["cat_vocab"]),
            max_length=seq_len,
            tables=count_tables if cfg.causal_counts else None,
            candidate_context=cfg.candidate_context == "on",
            candidate_context_time=cfg.candidate_context_time,
            session_summary=cfg.session_summary == "on",
            user_pref=cfg.user_pref == "on" and count_tables is not None,
            user_id_emb=getattr(cfg, "user_id_emb", "off") == "on",
            user_vocab=cache.user_vocab() if getattr(cfg, "user_id_emb", "off") == "on" else None,
            transition_pmi=transition_pmi,
        )
        ctx = merge_feature_context(ctx, side)
    need_gap = (
        float(getattr(cfg, "tsb_decay_hours", 0.0) or 0.0) > 0
        or bool(getattr(cfg, "tsb_dt_edges", ()) or ()))
    if need_gap:
        ctx = attach_target_gap_hours(ctx, data, indices)
    if float(getattr(cfg, "cat_stage_bias", 0.0) or 0.0) > 0:
        ctx = attach_cat_stage_counts(ctx, count_tables, data, indices)
    a2 = getattr(cfg, "_transition_a2_log1p", None)
    if float(getattr(cfg, "transition_2hop_bias", 0.0) or 0.0) > 0 and a2 is not None:
        ctx = attach_transition_2hop(ctx, a2, data, indices, candidate_rows_list)
    if (float(getattr(cfg, "hour_pop_score_bias", 0.0) or 0.0) > 0
            and count_tables is not None):
        ctx = attach_hour_pop(ctx, count_tables, data, indices, candidate_rows_list)
    if update and cfg.causal_counts and count_tables is not None:
        for sample_index in indices.tolist():
            count_tables.update_sample(data, sample_index)
    return ctx


def masked_candidate_lists(rows, mask):
    """Per-sample candidate rows with ``-1`` where masked (aligns with padded rows)."""
    out = []
    for row, m in zip(rows.tolist(), mask.tolist()):
        out.append([r if keep else -1 for r, keep in zip(row, m)])
    return out


def new_count_tables(cache, cfg, visit_bin_edges=None):
    return CausalCountTables(
        len(cache.pois), len(cache.meta["cat_vocab"]),
        visit_bins=cfg.visit_bins, visit_bin_edges=visit_bin_edges,
        dt_bin_edges=getattr(cfg, "tsb_dt_edges", ()) or ())


def replay_train_tables(cache, cfg, train_data, visit_bin_edges=None):
    """Rebuild train-end tables by chronological score-less updates."""
    tables = new_count_tables(cache, cfg, visit_bin_edges=visit_bin_edges)
    if visit_bin_edges is None:
        tables.fit_visit_bins(train_data, cfg.visit_bins)
    tables.replay(train_data)
    return tables

# Cutoffs used for Acc/Hit/Recall/NDCG/MRR/MAP tables (next-POI papers often include @20).
METRIC_KS = (1, 5, 10, 20)

# Primary headline metrics for terminal / report.txt (Acc@1 is the default focus).
PRIMARY_METRICS = ("Acc@1", "NDCG@1", "MRR")

# Human-readable labels for evaluate() group keys (JSON keys stay machine-stable).
# poi/* bins use SURD table target_count = # of times that POI is a train next-POI target.
GROUP_LABELS = {
    "poi/outside_catalog": "poi/outside_catalog — target not in train catalog",
    "poi/no_train_target": "poi/no_train_target — in catalog, 0 train targets",
    "poi/tail_1_2": "poi/tail — 1–2 train targets (long-tail)",
    "poi/mid_3_9": "poi/mid — 3–9 train targets",
    "poi/head_10_plus": "poi/head — ≥10 train targets (popular)",
    "history/1": "history/len=1 — prefix length 1",
    "history/2_5": "history/len=2–5 — prefix length 2–5",
    "history/6_plus": "history/len≥6 — prefix length ≥6",
    "last_category/0": "last_category/freq=0 — last-event category unseen in train",
    "last_category/1_9": "last_category/freq=1–9 — last-event category rare in train",
    "last_category/10_plus": "last_category/freq≥10 — last-event category common in train",
}

GROUP_NOTES = (
    "groups slice each eval sample by (1) poi/* = train-target popularity of the "
    "true next POI via SURD target_count: outside_catalog / no_train_target / "
    "tail_1_2 / mid_3_9 / head_10_plus; (2) history/* = observed prefix length "
    "bins 1 / 2_5 / 6_plus; (3) last_category/* = train frequency of the last "
    "history event's category bins 0 / 1_9 / 10_plus. Headline group columns are "
    "Acc@1, NDCG@1, full MRR."
)


def load_file(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def build_geo_knn_edges(coords: torch.Tensor, knn: int):
    """Undirected kNN edges on catalog coords with inverse-distance weights (#40)."""
    n = int(coords.shape[0])
    k = max(1, min(int(knn), max(1, n - 1)))
    if n < 2:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty, torch.zeros(0, dtype=torch.float32)
    lat, lon = coords[:, 0], coords[:, 1]
    # Pairwise haversine via broadcasting (catalog sizes are modest).
    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    dist = dist.fill_diagonal_(float("inf"))
    top = torch.topk(dist, k=k, largest=False)
    src_list, dst_list, w_list = [], [], []
    seen = set()
    for i in range(n):
        for j_idx, d in zip(top.indices[i].tolist(), top.values[i].tolist()):
            a, b = (i, j_idx) if i < j_idx else (j_idx, i)
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            src_list.append(a)
            dst_list.append(b)
            w_list.append(1.0 / (1.0 + float(d)))
    return (torch.tensor(src_list, dtype=torch.long),
            torch.tensor(dst_list, dtype=torch.long),
            torch.tensor(w_list, dtype=torch.float32))


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "manifest.json").read_text())
        self.pois = self.meta["poi_ids"]
        self.cfg = self.meta["config"]
        if self.cfg["version"] != "tsmoe.massivesteps.v1" or digest_json(self.cfg) != self.meta["fingerprint"]:
            raise ValueError("incompatible cache or manifest fingerprint")
        self.id = file_digest(self.path / "manifest.json")
        # Candidate construction is fixed at prepare time so train/eval share one pool rule.
        construction = self.meta.get("candidate_construction")
        if construction is None:
            # Pre-spatial caches: full catalog (legacy).
            self.candidate_construction = dict(
                method="full_catalog", radius_statistic="full", radius_km=None,
                empty_union_fallback="full_catalog", train_eval_shared=True,
                eval_uses_full_pool=True)
            self.radius_km = None
            self.popular_radius_km = None
            self.popular_top_k = 0
            self.popularity = None
            self.user_frequent = {}
            self.random_negatives = 0
            self.random_negatives_seed = 0
            self.poi_coords = torch.full((len(self.pois), 2), float("nan"), dtype=torch.float64)
        else:
            self.candidate_construction = construction
            self.radius_km = construction.get("radius_km")
            self.popular_radius_km = construction.get("popular_radius_km")
            self.popular_top_k = int(construction.get("popular_top_k") or 0)
            self.random_negatives = int(construction.get("random_negatives") or 0)
            self.random_negatives_seed = int(
                construction.get("random_negatives_seed")
                if construction.get("random_negatives_seed") is not None else 0)
            coords = self.meta.get("poi_coordinates")
            if coords is None or len(coords) != len(self.pois):
                raise ValueError("cache missing poi_coordinates aligned with poi_ids; re-run prepare")
            self.poi_coords = torch.tensor(coords, dtype=torch.float64)
            popular_by = construction.get("popular_by") or "visit"
            visit = construction.get("poi_visit_counts")
            target = construction.get("poi_target_counts")
            if popular_by == "target" and target is not None and len(target) == len(self.pois):
                scores = target
            elif visit is not None and len(visit) == len(self.pois):
                scores = visit
            else:
                scores = [0] * len(self.pois)
            self.popularity = torch.tensor(scores, dtype=torch.float64)
            if "user_frequent.pt" in self.meta.get("files", {}):
                verify_files(self.path, {"user_frequent.pt": self.meta["files"]["user_frequent.pt"]})
                raw = load_file(self.path / "user_frequent.pt")
                self.user_frequent = {str(k): list(v) for k, v in raw.items()}
            else:
                self.user_frequent = {}

    def load(self, name):
        verify_files(self.path, {name: self.meta["files"][name]})
        return load_file(self.path / name)

    def split(self, name):
        return self.load(f"{name}.pt")

    def table(self, name, device):
        stats = self.load(f"{name}.surd.pt")
        info = stats["manifest"]
        if (stats["poi_ids"] != self.pois or info["cache_id"] != self.meta["fingerprint"]
                or info["feature_method"] != FEATURE_METHOD
                or info["preprocessing_sha256"] != digest_json(self.meta["preprocessing"])):
            raise ValueError("SURD prior, catalog or preprocessing mismatch")
        pair = info["quantizer_manifest"]
        if pair:
            verify_files(self.path, {pair["file"]: pair["sha256"]})
        return FrozenSURDTable(stats).to(device)

    def user_vocab(self):
        """Stable map train ``user_id`` → row in ``[0, n_users)`` (for ``--user-id-emb``)."""
        cached = getattr(self, "_user_vocab", None)
        if cached is not None:
            return cached
        train = self.split("train")
        users = sorted({str(user) for user, _trail in train["trails"]})
        self._user_vocab = {user: i for i, user in enumerate(users)}
        return self._user_vocab

    def model(self, cfg):
        n_users = len(self.user_vocab()) if getattr(cfg, "user_id_emb", "off") == "on" else 0
        return TSMoE(
            len(self.pois), len(self.meta["cat_vocab"]), self.cfg["max_length"], cfg,
            n_users=n_users)

    def uses_full_catalog_pools(self):
        """True when every sample scores against the full training catalog."""
        method = self.candidate_construction.get("method")
        if method == "full_catalog":
            return True
        if method in ("multi_source", "random_pool"):
            return False
        # Legacy caches without an explicit method: None radius ⇒ full catalog.
        return self.radius_km is None

    def uses_random_pools(self):
        """True when prepare used ``--candidate-radius random``."""
        return self.candidate_construction.get("method") == "random_pool"

    def candidate_pools(self, data):
        """Same multi-source / random / full-catalog pools for train and evaluate.

        In-catalog ground-truth targets are always present (verified inside
        ``pools_for_split``); val/test ranking never drops the true next POI.
        """
        return pools_for_split(
            data, self.poi_coords, self.radius_km, len(self.pois),
            popular_radius_km=self.popular_radius_km,
            popularity=self.popularity,
            popular_top_k=self.popular_top_k,
            user_frequent=self.user_frequent,
            random_negatives=self.random_negatives,
            random_negatives_seed=self.random_negatives_seed,
            random_only=self.uses_random_pools(),
        )

    def pool_size_stats(self, data):
        """Min/max/mean/median candidate counts for one split (scoring definition)."""
        pools, resolutions = self.candidate_pools(data)
        return summarize_pool_sizes(pools, resolutions), pools, resolutions


def select_device(name):
    """Resolve --device. Logs selection; warns when auto falls back to CPU."""
    cuda_ok = torch.cuda.is_available()
    if name == "auto":
        if cuda_ok:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            print("WARNING: --device auto fell back to CPU because "
                  "torch.cuda.is_available() is False "
                  "(CPU-only torch build, missing NVIDIA driver, or no GPU).",
                  flush=True)
    else:
        device = torch.device(name)
        if device.type == "cuda" and not cuda_ok:
            raise RuntimeError(
                f"--device {name} requested CUDA, but torch.cuda.is_available() "
                "is False. Install a CUDA PyTorch wheel (see README) and check "
                "nvidia-smi / drivers.")
        if device.type == "mps" and not (hasattr(torch.backends, "mps")
                                         and torch.backends.mps.is_available()):
            raise RuntimeError(f"--device {name} requested MPS, but it is not available")
    parts = [f"device={device}", f"cuda_available={cuda_ok}"]
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        parts.append(f"gpu={torch.cuda.get_device_name(idx)!r}")
    print("Using " + " ".join(parts), flush=True)
    return device


def device_batch(data, indices, device, labels=True):
    history, valid, target = make_batch(data, indices, labels=labels)
    return {s: x.to(device) for s, x in history.items()}, valid.to(device), target


def candidates(targets, size, negatives, rng, pools=None):
    """Uniform or spatial negative sampling. Returns (rows, labels) for tests; train uses mask too."""
    rows, labels, _mask = sample_candidates(targets, size, negatives, rng, pools=pools)
    return rows, labels


def all_scores(model, table, encoded, valid, size, chunk, count_ctx=None,
               score_temperature=1.0):
    """Keep only scalar scores on the CPU across chunks. Encode the batch history once."""
    parts = []
    temp = float(score_temperature or 1.0)
    for start in range(0, size, chunk):
        check_interrupt()
        end = min(size, start + chunk)
        rows = torch.arange(start, end, device=valid.device).expand(len(valid), -1)
        prior, confidence = table(rows)
        scored = model.score(
            encoded, valid, rows, prior, confidence,
            count_ctx=slice_count_ctx(count_ctx, start, end),
            surd_specific=lookup_surd_specific(table, rows, model.cfg))
        part = scored["score"].cpu()
        if abs(temp - 1.0) > 1e-12:
            part = part / temp
        parts.append(part)
    scores = torch.cat(parts, 1)
    if not torch.isfinite(scores).all():
        raise FloatingPointError("nonfinite full-catalog scores")
    return scores


def score_pools(model, table, encoded, valid, pools, chunk, count_ctx=None,
                score_temperature=1.0):
    """Score variable-size candidate pools; pad with mask (shared train/eval path)."""
    width = max(int(pool.numel()) for pool in pools)
    rows = torch.zeros(len(pools), width, dtype=torch.long, device=valid.device)
    mask = torch.zeros(len(pools), width, dtype=torch.bool, device=valid.device)
    temp = float(score_temperature or 1.0)
    for i, pool in enumerate(pools):
        n = int(pool.numel())
        rows[i, :n] = pool.to(valid.device)
        mask[i, :n] = True
    parts = []
    for start in range(0, width, chunk):
        check_interrupt()
        end = min(width, start + chunk)
        sl = slice(start, end)
        chunk_rows, chunk_mask = rows[:, sl], mask[:, sl]
        safe = chunk_rows.masked_fill(~chunk_mask, -1)
        prior, confidence = table(safe)
        scored = model.score(
            encoded, valid, chunk_rows, prior, confidence, candidate_mask=chunk_mask,
            count_ctx=slice_count_ctx(count_ctx, start, end),
            surd_specific=lookup_surd_specific(table, safe, model.cfg))
        part = scored["score"].cpu()
        if abs(temp - 1.0) > 1e-12:
            part = part / temp
        part = part.masked_fill(~chunk_mask.cpu(), float("-inf"))
        parts.append(part)
    scores = torch.cat(parts, 1)
    finite = torch.isfinite(scores) | torch.isneginf(scores)
    if not finite.all():
        raise FloatingPointError("nonfinite candidate-pool scores")
    return scores, rows.cpu(), mask.cpu()


def ranks_in_pools(pool_scores, pool_rows, pool_mask, targets):
    """Rank the target inside each sample's candidate pool (1-based; 0 = unknown)."""
    ranks = []
    for scores, rows, mask, target in zip(pool_scores, pool_rows, pool_mask, targets.tolist()):
        if target < 0:
            ranks.append(0)
            continue
        valid_scores = scores[mask]
        valid_rows = rows[mask]
        hit = (valid_rows == target).nonzero(as_tuple=False)
        if hit.numel() == 0:
            ranks.append(0)
            continue
        pos = int(hit[0])
        truth = valid_scores[pos]
        earlier = torch.arange(len(valid_scores)) < pos
        rank = 1 + int((valid_scores > truth).sum()) + int(((valid_scores == truth) & earlier).sum())
        ranks.append(rank)
    return ranks


def ranks_of(scores, target):
    known = target >= 0
    row = target.clamp_min(0)
    truth = scores.gather(1, row[:, None])
    earlier = torch.arange(scores.shape[1])[None, :] < row[:, None]
    rank = 1 + (scores > truth).sum(1) + ((scores == truth) & earlier).sum(1)
    return rank.masked_fill(~known, 0)  # zero is the sentinel for out-of-catalog


def metric_row(ranks):
    """Aggregate ranks into ranking metrics.

    Single correct next-POI target ⇒ Acc@K = Hit@K = Recall@K, and MAP@K = MRR@K
    (AP of one relevant item is 1/rank when rank ≤ K). Unknown / out-of-catalog
    targets use rank sentinel 0 and contribute 0 to hit-based metrics under the
    all_targets denominator; known_only drops them first.
    """
    ranks = torch.as_tensor(ranks)
    known = ranks > 0
    n = len(ranks)
    known_ranks = ranks[known].double()
    out = dict(samples=n, known_targets=int(known.sum()),
               coverage=float(known.double().mean()) if n else None)
    if known_ranks.numel():
        out["mean_rank"] = float(known_ranks.mean())
        out["median_rank"] = float(known_ranks.median())
        # Full MRR (no K cutoff) over the same denominator as @K metrics.
        recip = torch.zeros(n, dtype=torch.float64)
        recip[known] = 1.0 / known_ranks
        out["MRR"] = float(recip.mean()) if n else None
    else:
        out["mean_rank"] = out["median_rank"] = out["MRR"] = None
    for k in METRIC_KS:
        hit = known & (ranks <= k)
        discount = ranks[hit].double()
        recall = float(hit.sum()) / n if n else None
        ndcg = ((1 / (1 + discount).log2()).sum().item() / n) if n else None
        mrr_at = ((1 / discount).sum().item() / n) if n else None
        out[f"Recall@{k}"] = recall
        out[f"Hit@{k}"] = recall
        out[f"Acc@{k}"] = recall
        out[f"NDCG@{k}"] = ndcg
        out[f"MRR@{k}"] = mrr_at
        out[f"MAP@{k}"] = mrr_at  # single relevant ⇒ MAP@K == MRR@K
    return out


def _fmt(value, width=8):
    if value is None:
        return f"{'—':>{width}}"
    if isinstance(value, float):
        return f"{value:{width}.4f}"
    return f"{value:>{width}}"


def format_primary_line(row, indent="  "):
    """One-line Acc@1 / NDCG@1 / MRR headline (default / primary metrics)."""
    return (f"{indent}primary  "
            f"Acc@1={_fmt(row.get('Acc@1')).strip()}  "
            f"NDCG@1={_fmt(row.get('NDCG@1')).strip()}  "
            f"MRR={_fmt(row.get('MRR')).strip()}")


def format_metric_block(row, title):
    """Compact human-readable table for one metric_row dict; Acc@1 leads."""
    lines = [title,
             format_primary_line(row),
             f"  samples={row['samples']}  known={row['known_targets']}  "
             f"coverage={_fmt(row['coverage']).strip()}  "
             f"mean_rank={_fmt(row['mean_rank']).strip()}  "
             f"median_rank={_fmt(row['median_rank']).strip()}"]
    header = f"  {'metric':<8}" + "".join(f"{'@' + str(k):>8}" for k in METRIC_KS)
    lines.append(header)
    for name in ("Acc", "Recall", "Hit", "NDCG", "MRR", "MAP"):
        # Acc/Hit/Recall share values; still print Acc + NDCG + MRR as the main trio,
        # and MAP once (aliases documented in README).
        if name in ("Recall", "Hit"):
            continue
        cells = "".join(_fmt(row.get(f"{name}@{k}")) for k in METRIC_KS)
        label = "Acc/Hit" if name == "Acc" else name
        lines.append(f"  {label:<8}{cells}")
    return "\n".join(lines)


def format_group_key(key):
    """Prefer the annotated label; fall back to the raw group key."""
    return GROUP_LABELS.get(key, key)


def format_eval_report(payload, group_keys=None):
    """Terminal / report.txt text: overview + main tables + short group summary."""
    lines = ["=== Evaluation ==="]
    for key in ("split", "device", "cache", "cache_id", "checkpoint", "catalog_size",
                "target_mode", "timestamp_utc"):
        value = payload.get(key)
        if value not in (None, ""):
            lines.append(f"  {key}={value}")
    pool_stats = payload.get("candidate_pool_stats")
    if pool_stats:
        lines.append(f"  {format_pool_size_line(payload.get('split') or 'split', pool_stats)}")
    # Lead the whole report with Acc@1 on all_targets (default / primary metric).
    all_row = payload["all_targets"]
    lines.append(
        f"  summary  Acc@1={_fmt(all_row.get('Acc@1')).strip()}  "
        f"NDCG@1={_fmt(all_row.get('NDCG@1')).strip()}  "
        f"MRR={_fmt(all_row.get('MRR')).strip()}  "
        f"(primary; full tables below)")
    lines.append("")
    lines.append(format_metric_block(all_row, "all_targets (OOV / unknown → miss)"))
    lines.append("")
    lines.append(format_metric_block(payload["known_only"], "known_only (in-catalog targets)"))
    groups = payload.get("groups") or {}
    if groups:
        keys = group_keys or sorted(groups)
        lines.append("")
        lines.append("groups — Acc@1 / NDCG@1 / MRR / n_known  "
                     "(poi = train-target popularity; history = prefix length; "
                     "last_category = train freq of last event's category)")
        for key in keys:
            row = groups[key]
            label = format_group_key(key)
            lines.append(
                f"  {label}")
            lines.append(
                f"    Acc@1={_fmt(row.get('Acc@1')).strip()}  "
                f"NDCG@1={_fmt(row.get('NDCG@1')).strip()}  "
                f"MRR={_fmt(row.get('MRR')).strip()}  "
                f"known={row['known_targets']}/{row['samples']}")
    return "\n".join(lines) + "\n"


def print_eval_tables(payload):
    """Print a tqdm-friendly summary (plain lines, no JSON dump)."""
    text = format_eval_report(payload)
    print(text, end="", flush=True)


def results_run_dir(path, *, cwd=None):
    """Map a train --out or checkpoint-parent path to results/<run_name>/.

    Under the working directory, strips a leading ``runs/`` segment so
    ``runs/tokyo/model`` → ``<cwd>/results/tokyo/model``. Paths outside the
    working directory (e.g. temp dirs in tests) use a sibling
    ``<parent>/results/<basename>``.
    """
    p = Path(path).expanduser().resolve()
    base = Path(cwd or Path.cwd()).resolve()
    try:
        rel = p.relative_to(base)
    except ValueError:
        return p.parent / "results" / p.name
    parts = rel.parts
    if parts and parts[0] == "runs":
        run_name = Path(*parts[1:]) if len(parts) > 1 else Path(p.name)
    else:
        run_name = rel if parts else Path(p.name)
    return base / "results" / run_name


def resolve_eval_out(args):
    """Directory for metrics.json / report.txt; default under results/."""
    if getattr(args, "out", None):
        return Path(args.out)
    # Mirror train --out under results/: runs/<city>/model → results/<city>/model/eval_<split>
    return results_run_dir(Path(args.checkpoint).resolve().parent) / f"eval_{args.split}"


def resolve_train_eval_dir(train_out, split):
    """Where train writes final eval artifacts (not checkpoints)."""
    return results_run_dir(train_out) / f"eval_{split}"


# Shared ablation table under results/ (gitignored); columns match docs/ablation_log.template.*
ABLATION_LOG_COLUMNS = ("date", "run_name", "cache", "key_flags", "Acc@1", "NDCG@1", "MRR", "notes")
ABLATION_MD_HEADER = (
    "# Ablation results log (auto-appended)\n"
    "\n"
    "Rows are appended by `train` / `evaluate` when they finish. "
    "Columns match [`docs/ablation_log.template.md`](../docs/ablation_log.template.md). "
    "Primary numbers are **test** `all_targets` Acc@1 / NDCG@1 / MRR "
    "(best-val summary may appear in `notes`).\n"
    "\n"
    "| date | run_name | cache | key flags | Acc@1 | NDCG@1 | MRR | notes |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
)


def results_root_dir(path, *, cwd=None):
    """Directory that holds ``ablation_log.*`` (the ``results/`` root for this run)."""
    run = results_run_dir(path, cwd=cwd)
    cur = run
    while cur.name != "results" and cur.parent != cur:
        cur = cur.parent
    return cur if cur.name == "results" else run.parent


def run_name_for_results(path, *, cwd=None):
    """POSIX path of the run under ``results/`` (e.g. ``tokyo/model``)."""
    run = results_run_dir(path, cwd=cwd)
    root = results_root_dir(path, cwd=cwd)
    try:
        return run.relative_to(root).as_posix()
    except ValueError:
        return run.name


def format_ablation_key_flags(cache, training_args=None, model_config=None):
    """Compact CLI-ish flag string for the ablation table (cache + train knobs)."""
    parts = []
    cc = cache.candidate_construction or {}
    radius = cc.get("radius_statistic") or cache.cfg.get("candidate_radius")
    if radius is not None:
        parts.append(f"--candidate-radius {radius}")
    if cc.get("method") == "multi_source":
        if cc.get("popular_top_k") is not None:
            parts.append(f"--popular-top-k {cc['popular_top_k']}")
        if cc.get("popular_radius") is not None:
            parts.append(f"--popular-radius {cc['popular_radius']}")
        if cc.get("user_frequent_top_k") is not None:
            parts.append(f"--user-frequent-top-k {cc['user_frequent_top_k']}")
        if cc.get("random_negatives") is not None:
            parts.append(f"--random-negatives {cc['random_negatives']}")
    elif cc.get("method") == "random_pool":
        # Random-only: neighborhood / popular / user-frequent unused.
        if cc.get("random_negatives") is not None:
            parts.append(f"--random-negatives {cc['random_negatives']}")
        if cc.get("random_negatives_seed") is not None:
            parts.append(f"--random-negatives-seed {cc['random_negatives_seed']}")
    targets = []
    if cache.cfg.get("train_targets") is not None:
        targets.append(f"--train-targets {cache.cfg['train_targets']}")
    if cache.cfg.get("eval_targets") is not None:
        targets.append(f"--eval-targets {cache.cfg['eval_targets']}")
    parts.extend(targets)
    if cache.cfg.get("filter_unknown", "off") == "on":
        parts.append("--filter-unknown on")
    cfg = model_config or {}
    args = training_args or {}
    router = args.get("router_mode", cfg.get("router_mode"))
    if router is not None:
        parts.append(f"--router-mode {router}")
    if args.get("no_surd") or cfg.get("use_surd") is False:
        parts.append("--no-surd")
    causal = args.get("causal_counts", cfg.get("causal_counts"))
    if causal is False or args.get("no_causal_counts") is True:
        parts.append("--no-causal-counts")
    else:
        parts.append("--causal-counts")
        attn = args.get("transition_attn", cfg.get("transition_attn", "bias"))
        parts.append(f"--transition-attn {attn}")
        if args.get("transition_lambda", cfg.get("transition_lambda")) is not None:
            parts.append(f"--transition-lambda {args.get('transition_lambda', cfg.get('transition_lambda'))}")
        tsb = args.get("transition_score_bias", cfg.get("transition_score_bias", 0.0))
        if tsb is not None and float(tsb) != 0.0:
            parts.append(f"--transition-score-bias {tsb}")
        for key, flag in (
                ("category_score_bias", "--category-score-bias"),
                ("pop_score_bias", "--pop-score-bias"),
                ("geo_score_bias", "--geo-score-bias"),
                ("geo_score_power", "--geo-score-power"),
        ):
            val = args.get(key, cfg.get(key, 0.0 if "bias" in key else 1.0))
            if val is None:
                continue
            # Always emit nonzero biases; emit power only when ≠ default 1.
            if "power" in key:
                if abs(float(val) - 1.0) > 1e-12:
                    parts.append(f"{flag} {val}")
            elif float(val) != 0.0:
                parts.append(f"{flag} {val}")
        if args.get("visit_bins", cfg.get("visit_bins")) is not None:
            parts.append(f"--visit-bins {args.get('visit_bins', cfg.get('visit_bins'))}")
    gate = args.get("surd_bias_gate", cfg.get("surd_bias_gate", "off"))
    if gate not in (None, "off"):
        parts.append(f"--surd-bias-gate {gate}")
        ssb = args.get("surd_score_bias", cfg.get("surd_score_bias", 0.0))
        if ssb is not None and float(ssb) != 0.0:
            parts.append(f"--surd-score-bias {ssb}")
        tsb_g = args.get("tsb_gate_mode", cfg.get("tsb_gate_mode", "off"))
        if tsb_g not in (None, "off"):
            parts.append(f"--tsb-gate-mode {tsb_g}")
        attn_g = args.get("surd_attn_gate", cfg.get("surd_attn_gate", "off"))
        if attn_g not in (None, "off"):
            parts.append(f"--surd-attn-gate {attn_g}")
    for key, flag in (
            ("candidate_context", "--candidate-context"),
            ("candidate_context_time", "--candidate-context-time"),
            ("session_summary", "--session-summary"),
            ("user_pref", "--user-pref"),
            ("user_id_emb", "--user-id-emb"),
            ("transition_pmi", "--transition-pmi"),
            ("expert_cross_attn", "--expert-cross-attn"),
            ("expert_cross_attn_sparse", "--expert-cross-attn-sparse"),
    ):
        value = args.get(key, cfg.get(key))
        if value is not None and value != "off":
            parts.append(f"{flag} {value}")
        elif key == "expert_cross_attn_sparse" and cfg.get("expert_cross_attn", "off") not in (None, "off"):
            # Always record sparse setting when expert cross-attn is active.
            parts.append(f"{flag} {value or 'on'}")
    if args.get("negatives") is not None:
        parts.append(f"--negatives {args['negatives']}")
    if args.get("max_train_candidates") is not None:
        parts.append(f"--max-train-candidates {args['max_train_candidates']}")
    # Record non-default train hyperparams / selection (hillclimb knobs).
    defaults = {
        "lr": 0.001, "weight_decay": 0.01, "dropout": 0.1, "dim": 128,
        "layers": 2, "heads": 4, "top_k": 4, "batch_size": 32,
        "checkpoint_metric": "ndcg10", "lambda_bpr": 0.0, "bpr_margin": 0.0,
        "temperature": 1.0, "epochs": 20,
    }
    for key, flag in (
            ("lr", "--lr"), ("weight_decay", "--weight-decay"),
            ("dropout", "--dropout"), ("dim", "--dim"), ("layers", "--layers"),
            ("heads", "--heads"), ("top_k", "--top-k"),
            ("batch_size", "--batch-size"),
            ("checkpoint_metric", "--checkpoint-metric"),
            ("lambda_bpr", "--lambda-bpr"), ("bpr_margin", "--bpr-margin"),
            ("temperature", "--temperature"), ("epochs", "--epochs"),
    ):
        if key not in args or args[key] is None:
            continue
        if args[key] != defaults.get(key):
            parts.append(f"{flag} {args[key]}")
    return " ".join(parts)


def _metric_cell(row, key):
    if not row or row.get(key) is None:
        return ""
    value = row[key]
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".") if value != 0 else "0"
    return str(value)


def _csv_escape(value):
    text = "" if value is None else str(value)
    if any(c in text for c in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def append_ablation_log(row, results_anchor, *, cwd=None):
    """Append one ablation row to ``results/ablation_log.md`` and ``.csv``.

    Creates both files (with header) when missing. ``results_anchor`` is a train
    ``--out`` or checkpoint-parent path used to locate the shared ``results/`` root.
    ``row`` keys match ``ABLATION_LOG_COLUMNS``.
    """
    root = results_root_dir(results_anchor, cwd=cwd)
    root.mkdir(parents=True, exist_ok=True)
    md_path, csv_path = root / "ablation_log.md", root / "ablation_log.csv"
    cells = [row.get(col, "") for col in ABLATION_LOG_COLUMNS]
    # Markdown: display "key flags" header already in template; body cells raw.
    md_line = "| " + " | ".join("" if c is None else str(c) for c in cells) + " |\n"
    if not md_path.is_file():
        md_path.write_text(ABLATION_MD_HEADER + md_line, encoding="utf-8")
    else:
        with md_path.open("a", encoding="utf-8") as f:
            f.write(md_line)
    csv_line = ",".join(_csv_escape(c) for c in cells) + "\n"
    if not csv_path.is_file():
        csv_path.write_text(",".join(ABLATION_LOG_COLUMNS) + "\n" + csv_line, encoding="utf-8")
    else:
        with csv_path.open("a", encoding="utf-8") as f:
            f.write(csv_line)
    print(f"Appended ablation row → {md_path}", flush=True)
    print(f"Appended ablation row → {csv_path}", flush=True)
    return md_path, csv_path


def ablation_row_from_eval(cache, metrics_row, *, run_name, cache_path, key_flags,
                           notes="", date=None):
    """Build an ablation_log row dict from a primary metric row."""
    return dict(
        date=date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        run_name=run_name,
        cache=str(cache_path),
        key_flags=key_flags,
        **{k: _metric_cell(metrics_row, k) for k in ("Acc@1", "NDCG@1", "MRR")},
        notes=notes,
    )


def primary_metric_snapshot(metrics):
    """Compact Acc@1 / NDCG@1 / MRR (+ NDCG@10 when present) from an evaluate() dict."""
    row = (metrics or {}).get("all_targets") or {}
    out = {k: row.get(k) for k in PRIMARY_METRICS}
    if "NDCG@10" in row:
        out["NDCG@10"] = row.get("NDCG@10")
    return out


def build_eval_payload(args, cache, metrics, device, checkpoint_path=None, checkpoint=None):
    settings = {key: value for key, value in vars(args).items()
                if isinstance(value, (str, float, int, bool))}
    ckpt_path = checkpoint_path or getattr(args, "checkpoint", None)
    ckpt = checkpoint
    if ckpt is None and ckpt_path and Path(ckpt_path).is_file():
        ckpt = load_file(ckpt_path)
    ckpt = ckpt or {}
    return dict(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        split=getattr(args, "split", None),
        catalog_size=len(cache.pois),
        target_mode=cache.cfg["eval_targets"],
        filter_unknown=cache.cfg.get("filter_unknown", "off"),
        cache=str(Path(args.cache).resolve()),
        cache_id=cache.id,
        checkpoint=str(Path(ckpt_path).resolve()) if ckpt_path else None,
        device=str(device),
        cuda_available=torch.cuda.is_available(),
        model_config=ckpt.get("config"),
        training_args=ckpt.get("training_args"),
        checkpoint_epoch=ckpt.get("epoch"),
        checkpoint_validation_ndcg10=ckpt.get("validation_ndcg10"),
        cli=settings,
        metric_ks=list(METRIC_KS),
        primary_metrics=list(PRIMARY_METRICS),
        metric_notes=("Primary headline metrics are Acc@1, NDCG@1, and full MRR. "
                      "Acc@K=Hit@K=Recall@K and MAP@K=MRR@K for single next-POI target; "
                      "rank ties break by lower catalog row index; "
                      "unknown targets use rank 0 and count as misses under all_targets. "
                      "When candidate_construction.method is multi_source, ranks are within "
                      "the per-sample constructed pool (full union at eval; training may "
                      "subsample negatives from the same pool)."),
        candidate_construction=cache.candidate_construction,
        group_notes=GROUP_NOTES,
        group_labels=dict(GROUP_LABELS),
        **metrics,
    )


def report_candidate_pool_stats(cache, splits):
    """Build and print per-split pool-size stats (same pools as train negatives / eval).

    ``splits`` is an ordered mapping of split name → split tensor dict.
    Returns ``(stats_by_split, pools_by_split)``.
    """
    radius_info = cache.candidate_construction
    print(
        f"candidates method={radius_info.get('method')} "
        f"radius_statistic={radius_info.get('radius_statistic')} "
        f"radius_km={cache.radius_km} "
        f"popular_radius={radius_info.get('popular_radius')} "
        f"popular_radius_km={cache.popular_radius_km} "
        f"popular_top_k={cache.popular_top_k} "
        f"popular_by={radius_info.get('popular_by')} "
        f"user_frequent_top_k={radius_info.get('user_frequent_top_k')} "
        f"random_negatives={cache.random_negatives} "
        f"eval_pool=full_constructed ground_truth_in_pool=always",
        flush=True,
    )
    all_stats = {}
    pools_by_split = {}
    for name, data in splits.items():
        stats, pools, _resolutions = cache.pool_size_stats(data)
        all_stats[name] = stats
        pools_by_split[name] = pools
        print(format_pool_size_line(name, stats), flush=True)
    return all_stats, pools_by_split


def save_eval_results(payload, out_dir, legacy_output=None):
    """Write metrics.json + report.txt under out_dir; optionally mirror JSON to --output."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    report_path = out / "report.txt"
    dump_json(payload, metrics_path)
    report_path.write_text(format_eval_report(payload), encoding="utf-8")
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)
    if legacy_output:
        legacy = Path(legacy_output)
        if legacy.resolve() != metrics_path.resolve():
            legacy.parent.mkdir(parents=True, exist_ok=True)
            dump_json(payload, legacy)
            print(f"Wrote {legacy}", flush=True)
    return metrics_path, report_path


@torch.inference_mode()
def evaluate(model, table, data, cache, batch_size, chunk, cat_frequency=None, desc="eval",
             count_tables=None, score_temperature=1.0, tta_modes=(), tta_keep_last=True,
             tta_views=1, tta_fuse="mean", score_calib="off", calib_lambda=0.0,
             tta_seed=0, candidate_spatial_prior="off", candidate_prior_top_m=0,
             candidate_prior_gravity_beta=1.0, mc_dropout=0, mc_keep_train_dropout=False):
    """Rank samples; with ``count_tables``, walk chronologically (score then update).

    Optional eval-only Acc levers: TTA (#36), Gaussian+feature calib (#34),
    coverage-safe spatial top-M pool shrink (#10), MC-dropout logit avg (#46).
    """
    mc_n = int(mc_dropout or 0)
    if mc_n < 0:
        raise ValueError("--mc-dropout must be nonnegative")
    if mc_n > 1 and tta_modes:
        raise ValueError("combine MC-dropout with TTA is not supported")
    model.eval()
    ranks, grouped = [], defaultdict(list)
    counts = table.target_count.cpu()
    device = table.routing_prior.device
    pools, _resolutions = cache.candidate_pools(data)
    pool_shrink_stats = None
    prior = str(candidate_spatial_prior or "off").lower()
    top_m = int(candidate_prior_top_m or 0)
    if prior not in ("", "off", "none") and top_m > 0:
        if cache.uses_full_catalog_pools():
            raise ValueError("--candidate-spatial-prior requires constructed pools "
                             "(not full-catalog cache)")
        pools, pool_shrink_stats = shrink_pools_spatial_prior(
            data, pools, cache.poi_coords,
            prior=prior, top_m=top_m,
            popularity=cache.popularity,
            gravity_beta=float(candidate_prior_gravity_beta or 1.0))
        cov = pool_shrink_stats.get("natural_gt_coverage")
        print(
            f"candidate_spatial_prior={prior} top_m={top_m} "
            f"natural_gt_coverage={cov} "
            f"size_after_mean={pool_shrink_stats.get('size_after_mean')}",
            flush=True)
        if cov is not None and cov < 0.98:
            print(
                f"WARNING: natural GT coverage {cov:.4f} < 0.98 — Acc@1 may be "
                "unfair if force-include rescues many targets",
                flush=True)
    use_counts = count_tables is not None and model.cfg.causal_counts
    use_side = needs_side_features(model.cfg)
    chronological = use_counts or (use_side and model.cfg.user_pref == "on")
    score_temp = float(score_temperature or 1.0)
    tta_modes = tuple(tta_modes or ())
    fuse = str(tta_fuse or "mean").lower()
    if fuse not in ("mean", "rrf"):
        raise ValueError("--tta-fuse must be mean|rrf")
    if chronological:
        order = chronological_indices(data)
        batches = list(order.split(batch_size))
        # Collect ranks in original sample order for stable group reporting.
        rank_by_index = [0] * len(data["end"])
    else:
        batches = list(torch.arange(len(data["end"])).split(batch_size))
        rank_by_index = None

    def _score_one(h_view, valid_view, count_ctx, batch_pools):
        encoded = model.encode(h_view, valid_view, count_ctx)
        if cache.uses_full_catalog_pools():
            return all_scores(
                model, table, encoded, valid_view, len(cache.pois), chunk,
                count_ctx=count_ctx, score_temperature=score_temp), None, None
        pool_scores, pool_rows, pool_mask = score_pools(
            model, table, encoded, valid_view, batch_pools, chunk,
            count_ctx=count_ctx, score_temperature=score_temp)
        return pool_scores, pool_rows, pool_mask

    def _fuse_score_list(score_list, pool_mask):
        if len(score_list) == 1:
            return score_list[0]
        if fuse == "mean":
            stacked = torch.stack(score_list, 0)
            return stacked.mean(0)
        # Reciprocal rank fusion over views (k=60).
        k = 60.0
        acc = torch.zeros_like(score_list[0])
        for s in score_list:
            # Dense RRF: 1/(k+rank); mask pads stay -inf → large rank ignored via mask.
            order = torch.argsort(s, dim=1, descending=True, stable=True)
            ranks_v = torch.empty_like(order)
            ranks_v.scatter_(1, order, torch.arange(1, s.shape[1] + 1, device=s.device).expand_as(s))
            acc = acc + 1.0 / (k + ranks_v.float())
        if pool_mask is not None:
            acc = acc.masked_fill(~pool_mask, float("-inf"))
        return acc

    for idx in tqdm(batches, desc=desc, leave=False, dynamic_ncols=True):
        check_interrupt()
        h, valid, target = device_batch(data, idx, device)
        count_ctx = None
        if use_counts or use_side:
            seq_len = int(valid.shape[1])
            if cache.uses_full_catalog_pools():
                cand_lists = [list(range(len(cache.pois))) for _ in idx.tolist()]
            else:
                cand_lists = [pools[i].tolist() for i in idx.tolist()]
            # Read features causally, then update tables with GT (same protocol as train).
            count_ctx = gather_features_causal(
                model.cfg, cache, data, idx, cand_lists, seq_len, count_tables,
                update=use_counts)
            count_ctx = count_ctx_to_device(count_ctx, device)
        batch_pools = None if cache.uses_full_catalog_pools() else [pools[i] for i in idx.tolist()]
        views = [(h, valid)]
        if tta_modes:
            views.extend(list(make_tta_history_views(
                h, valid, tta_modes, keep_last=tta_keep_last,
                n_views=tta_views, seed=tta_seed + int(idx[0]) * 13)))
        score_list = []
        pool_rows = pool_mask = None
        if mc_n > 1:
            # #46: stochastic forwards with dropout on; mean logits (training-free).
            was_training = model.training
            model.train()  # enable dropout for MC samples (#46)
            for _ in range(mc_n):
                scores_i, pool_rows_i, pool_mask_i = _score_one(
                    h, valid, count_ctx, batch_pools)
                score_list.append(scores_i)
                if pool_rows_i is not None:
                    pool_rows, pool_mask = pool_rows_i, pool_mask_i
            model.train(was_training)
            _ = mc_keep_train_dropout  # CLI flag retained for docs; always train-mode drop
            scores = torch.stack(score_list, 0).mean(0)
            if pool_mask is not None:
                scores = scores.masked_fill(~pool_mask, float("-inf"))
        else:
            for h_view, v_view in views:
                scores_i, pool_rows_i, pool_mask_i = _score_one(
                    h_view, v_view, count_ctx, batch_pools)
                score_list.append(scores_i)
                if pool_rows_i is not None:
                    pool_rows, pool_mask = pool_rows_i, pool_mask_i
            scores = _fuse_score_list(score_list, pool_mask)
        if cache.uses_full_catalog_pools():
            if str(score_calib or "off").lower() not in ("", "off", "none"):
                # Full-catalog calib: build a dense mask of ones.
                full_mask = torch.ones_like(scores, dtype=torch.bool)
                fake_rows = torch.arange(scores.shape[1]).expand(scores.shape[0], -1)
                scores = apply_score_calib(
                    scores, fake_rows, full_mask, count_ctx, score_calib, calib_lambda)
            batch_ranks = ranks_of(scores, target).tolist()
        else:
            scores = apply_score_calib(
                scores, pool_rows, pool_mask, count_ctx, score_calib, calib_lambda)
            batch_ranks = ranks_in_pools(scores, pool_rows, pool_mask, target.cpu())
        if rank_by_index is not None:
            for i, rank in zip(idx.tolist(), batch_ranks):
                rank_by_index[i] = rank
        else:
            ranks.extend(batch_ranks)
        if cat_frequency is not None and rank_by_index is None:
            # Slice ranks into Head/Mid/Tail-style popularity bins plus context bins.
            # count = SURD target_count (# times this POI is a train next-POI target).
            for i, row, rank in zip(idx.tolist(), target.tolist(), batch_ranks):
                end = int(data["end"][i])
                length = end - int(data["start"][i])
                count = int(counts[row]) if row >= 0 else 0
                cat_count = cat_frequency[int(data["C"][end - 1])]
                popularity = ("outside_catalog" if row < 0 else "no_train_target" if count == 0 else
                              "tail_1_2" if count <= 2 else "mid_3_9" if count < 10 else "head_10_plus")
                length_bin = "1" if length == 1 else "2_5" if length <= 5 else "6_plus"
                cat_bin = "0" if not cat_count else "1_9" if cat_count < 10 else "10_plus"
                for key in (f"poi/{popularity}", f"history/{length_bin}", f"last_category/{cat_bin}"):
                    grouped[key].append(rank)
    if rank_by_index is not None:
        ranks = rank_by_index
        if cat_frequency is not None:
            for i, rank in enumerate(ranks):
                row = int(data["poi_token"][int(data["end"][i])]) - 2
                end = int(data["end"][i])
                length = end - int(data["start"][i])
                count = int(counts[row]) if row >= 0 else 0
                cat_count = cat_frequency[int(data["C"][end - 1])]
                popularity = ("outside_catalog" if row < 0 else "no_train_target" if count == 0 else
                              "tail_1_2" if count <= 2 else "mid_3_9" if count < 10 else "head_10_plus")
                length_bin = "1" if length == 1 else "2_5" if length <= 5 else "6_plus"
                cat_bin = "0" if not cat_count else "1_9" if cat_count < 10 else "10_plus"
                for key in (f"poi/{popularity}", f"history/{length_bin}", f"last_category/{cat_bin}"):
                    grouped[key].append(rank)
    out = dict(all_targets=metric_row(ranks), known_only=metric_row([r for r in ranks if r]),
               groups={key: metric_row(rs) for key, rs in sorted(grouped.items())})
    if pool_shrink_stats is not None:
        out["candidate_spatial_prior"] = pool_shrink_stats
    return out


def save_checkpoint(payload, path):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        staged = Path(tmp.name)
    try:
        torch.save(payload, staged)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def fit_jump_lognormal(distances, eps=1e-6):
    """Fit lognormal (μ, σ) on log(km+eps) from consecutive train hop distances."""
    if not distances:
        return 0.0, 1.0
    logs = [math.log(max(float(d), 0.0) + eps) for d in distances]
    n = len(logs)
    mu = sum(logs) / n
    var = sum((x - mu) ** 2 for x in logs) / max(n - 1, 1)
    sigma = math.sqrt(max(var, 1e-8))
    return float(mu), float(sigma)


def jump_params_from_distance_stats(stats, eps=1e-6):
    """Approximate lognormal (μ, σ) from prepare distance_stats (median/mean)."""
    stats = stats or {}
    median = stats.get("median_km") or stats.get("mean_km") or 1.0
    mean = stats.get("mean_km") or median
    median = max(float(median), eps)
    mean = max(float(mean), eps)
    mu = math.log(median)
    if mean > median:
        sigma = math.sqrt(max(2.0 * (math.log(mean) - mu), 1e-4))
    else:
        sigma = 1.0
    return float(mu), float(sigma)


def train(args):
    install_interrupt_handler()
    cache = Cache(args.cache)
    causal_counts = not bool(getattr(args, "no_causal_counts", False))
    transition_attn = getattr(args, "transition_attn", "bias")
    transition_lambda = float(getattr(args, "transition_lambda", 1.0))
    transition_score_bias = float(getattr(args, "transition_score_bias", 0.0) or 0.0)
    category_score_bias = float(getattr(args, "category_score_bias", 0.0) or 0.0)
    pop_score_bias = float(getattr(args, "pop_score_bias", 0.0) or 0.0)
    geo_score_bias = float(getattr(args, "geo_score_bias", 0.0) or 0.0)
    geo_score_power = float(getattr(args, "geo_score_power", 1.0) or 1.0)
    gauss_geo_bias = float(getattr(args, "gauss_geo_bias", 0.0) or 0.0)
    gauss_sigma_km = float(getattr(args, "gauss_sigma_km", 0.5) or 0.5)
    geo_adapt = getattr(args, "geo_adapt", "off")
    geo_adapt_temp = float(getattr(args, "geo_adapt_temp", 1.0) or 1.0)
    revisit_score_bias = float(getattr(args, "revisit_score_bias", 0.0) or 0.0)
    history_hit_bias = float(getattr(args, "history_hit_bias", 0.0) or 0.0)
    history_hit_k = int(getattr(args, "history_hit_k", 5) or 5)
    revisit_recency_bias = float(getattr(args, "revisit_recency_bias", 0.0) or 0.0)
    revisit_recency_tau = float(getattr(args, "revisit_recency_tau", 3.0) or 3.0)
    radiation_score_bias = float(getattr(args, "radiation_score_bias", 0.0) or 0.0)
    radiation_kernel = getattr(args, "radiation_kernel", "hard")
    radiation_soft_km = float(getattr(args, "radiation_soft_km", 0.25) or 0.25)
    jump_length_bias = float(getattr(args, "jump_length_bias", 0.0) or 0.0)
    jump_dist = getattr(args, "jump_dist", "lognormal")
    jump_beta = float(getattr(args, "jump_beta", 1.5) or 1.5)
    jump_d0 = float(getattr(args, "jump_d0", 0.1) or 0.0)
    return_gate = getattr(args, "return_gate", "off")
    explore_spatial_bias = getattr(args, "explore_spatial_bias", "off")
    explore_score_bias = float(getattr(args, "explore_score_bias", 0.0) or 0.0)
    fam_area_blend = getattr(args, "fam_area_blend", "off")
    fam_radius_km = float(getattr(args, "fam_radius_km", 0.5) or 0.5)
    fam_explore_bias = float(getattr(args, "fam_explore_bias", 0.0) or 0.0)
    fam_return_bias = float(getattr(args, "fam_return_bias", 0.0) or 0.0)
    tsb_history_k = int(getattr(args, "tsb_history_k", 1) or 1)
    tsb_history_decay = float(getattr(args, "tsb_history_decay", 0.5) or 0.5)
    tsb_dt_edges = tuple(parse_dt_bin_edges(getattr(args, "transition_dt_bins", "off")))
    cat_stage_bias = float(getattr(args, "cat_stage_bias", 0.0) or 0.0)
    transition_2hop_bias = float(getattr(args, "transition_2hop_bias", 0.0) or 0.0)
    hour_pop_score_bias = float(getattr(args, "hour_pop_score_bias", 0.0) or 0.0)
    tsb_decay_hours = float(getattr(args, "tsb_decay_hours", 0.0) or 0.0)
    tsb_session_gap_hours = float(getattr(args, "tsb_session_gap_hours", 0.0) or 0.0)
    if transition_2hop_bias < 0:
        raise ValueError("--transition-2hop-bias must be nonnegative")
    if hour_pop_score_bias < 0:
        raise ValueError("--hour-pop-score-bias must be nonnegative")
    if geo_score_power <= 0:
        raise ValueError("--geo-score-power must be positive")
    if gauss_geo_bias < 0:
        raise ValueError("--gauss-geo-bias must be nonnegative")
    if gauss_geo_bias > 0 and gauss_sigma_km <= 0:
        raise ValueError("--gauss-sigma-km must be positive when --gauss-geo-bias > 0")
    if geo_adapt not in ("off", "sparse"):
        raise ValueError("--geo-adapt must be off|sparse")
    if geo_adapt_temp < 0:
        raise ValueError("--geo-adapt-temp must be nonnegative")
    if revisit_score_bias < 0:
        raise ValueError("--revisit-score-bias must be nonnegative")
    if history_hit_bias < 0:
        raise ValueError("--history-hit-bias must be nonnegative")
    if history_hit_k < 1:
        raise ValueError("--history-hit-k must be >= 1")
    if revisit_recency_bias < 0:
        raise ValueError("--revisit-recency-bias must be nonnegative")
    if revisit_recency_tau <= 0:
        raise ValueError("--revisit-recency-tau must be positive")
    if radiation_score_bias < 0:
        raise ValueError("--radiation-score-bias must be nonnegative")
    if radiation_kernel not in ("hard", "soft"):
        raise ValueError("--radiation-kernel must be hard|soft")
    if radiation_soft_km <= 0:
        raise ValueError("--radiation-soft-km must be positive")
    if jump_length_bias < 0:
        raise ValueError("--jump-length-bias must be nonnegative")
    if jump_dist not in ("lognormal", "trunc_power"):
        raise ValueError("--jump-dist must be lognormal|trunc_power")
    if jump_beta <= 0:
        raise ValueError("--jump-beta must be positive")
    if jump_d0 < 0:
        raise ValueError("--jump-d0 must be nonnegative")
    if return_gate not in ("off", "on"):
        raise ValueError("--return-gate must be off|on")
    if explore_spatial_bias not in ("off", "gauss", "geo", "radiation"):
        raise ValueError("--explore-spatial-bias must be off|gauss|geo|radiation")
    if explore_score_bias < 0:
        raise ValueError("--explore-score-bias must be nonnegative")
    if fam_area_blend not in ("off", "on"):
        raise ValueError("--fam-area-blend must be off|on")
    if fam_area_blend == "on":
        if fam_radius_km <= 0:
            raise ValueError("--fam-radius-km must be positive")
        if fam_explore_bias < 0 or fam_return_bias < 0:
            raise ValueError("--fam-explore-bias/--fam-return-bias must be nonnegative")
        if fam_explore_bias <= 0 and fam_return_bias <= 0:
            raise ValueError("--fam-area-blend on requires a positive fam bias")
    if tsb_history_k < 1:
        raise ValueError("--tsb-history-k must be >= 1")
    if tsb_history_decay <= 0:
        raise ValueError("--tsb-history-decay must be positive")
    if cat_stage_bias < 0:
        raise ValueError("--cat-stage-bias must be nonnegative")
    if return_gate == "on":
        if revisit_score_bias <= 0:
            raise ValueError("--return-gate on requires --revisit-score-bias > 0")
        if explore_spatial_bias == "off" or explore_score_bias <= 0:
            raise ValueError(
                "--return-gate on requires --explore-spatial-bias ≠ off and "
                "--explore-score-bias > 0")
    if tsb_decay_hours < 0:
        raise ValueError("--tsb-decay-hours must be nonnegative")
    if tsb_session_gap_hours < 0:
        raise ValueError("--tsb-session-gap must be nonnegative hours")
    ce_focal_gamma = float(getattr(args, "ce_focal_gamma", 0.0) or 0.0)
    if ce_focal_gamma < 0:
        raise ValueError("--ce-focal-gamma must be nonnegative")
    bpr_mode = str(getattr(args, "bpr_mode", "max") or "max")
    if bpr_mode not in ("max", "bpr_max", "simce"):
        raise ValueError("--bpr-mode must be max|bpr_max|simce")
    soft_target_beta = float(getattr(args, "soft_target_beta", 0.0) or 0.0)
    if soft_target_beta < 0 or soft_target_beta > 1:
        raise ValueError("--soft-target-beta must be in [0, 1]")
    soft_target_prior = str(getattr(args, "soft_target_prior", "off") or "off")
    soft_priors = ()
    if soft_target_prior not in ("", "off", "none"):
        soft_priors = tuple(
            p.strip().lower() for p in soft_target_prior.split(",") if p.strip())
        bad = set(soft_priors) - {"tsb", "revisit", "ema"}
        if bad:
            raise ValueError(
                "--soft-target-prior must be off or comma list of tsb|revisit|ema")
    if soft_target_beta > 0 and not soft_priors:
        raise ValueError("--soft-target-beta > 0 requires --soft-target-prior")
    if soft_priors and soft_target_beta <= 0:
        raise ValueError("--soft-target-prior requires --soft-target-beta > 0")
    if soft_target_beta > 0 and ce_focal_gamma > 0:
        raise ValueError("soft-target cannot combine with --ce-focal-gamma > 0")
    count_priors = tuple(p for p in soft_priors if p in ("tsb", "revisit"))
    use_ema_soft = "ema" in soft_priors
    if count_priors and not causal_counts:
        raise ValueError("soft-target tsb|revisit requires causal counts")
    if soft_target_beta > 0 and not use_ema_soft and not count_priors:
        raise ValueError("--soft-target-beta > 0 requires a soft-target prior")
    ema_decay = float(getattr(args, "ema_decay", 0.999) or 0.999)
    if use_ema_soft and not (0.0 < ema_decay < 1.0):
        raise ValueError("--ema-decay must be in (0, 1) when soft-target-prior includes ema")
    kd_teacher_path = getattr(args, "kd_teacher", None)
    if kd_teacher_path in ("", "off", "none"):
        kd_teacher_path = None
    lambda_kd = float(getattr(args, "lambda_kd", 0.0) or 0.0)
    kd_top_k = int(getattr(args, "kd_top_k", 20) or 20)
    kd_temp = float(getattr(args, "kd_temp", 2.0) or 2.0)
    if kd_teacher_path and lambda_kd <= 0:
        raise ValueError("--kd-teacher requires --lambda-kd > 0")
    if lambda_kd < 0:
        raise ValueError("--lambda-kd must be nonnegative")
    if kd_teacher_path and kd_top_k < 1:
        raise ValueError("--kd-top-k must be >= 1")
    if kd_teacher_path and kd_temp <= 0:
        raise ValueError("--kd-temp must be positive")
    poi_geo_laplace = float(getattr(args, "poi_geo_laplace", 0.0) or 0.0)
    geo_knn = int(getattr(args, "geo_knn", 8) or 8)
    if poi_geo_laplace < 0:
        raise ValueError("--poi-geo-laplace must be nonnegative")
    if poi_geo_laplace > 0 and geo_knn < 1:
        raise ValueError("--geo-knn must be >= 1 when --poi-geo-laplace > 0")
    lambda_cat_aux = float(getattr(args, "lambda_cat_aux", 0.0) or 0.0)
    if lambda_cat_aux < 0:
        raise ValueError("--lambda-cat-aux must be nonnegative")
    neg_sample = str(getattr(args, "neg_sample", "uniform") or "uniform").lower()
    if neg_sample not in ("uniform", "adaptive_mix"):
        raise ValueError("--neg-sample must be uniform|adaptive_mix")
    neg_hard_frac = float(getattr(args, "neg_hard_frac", 0.25) or 0.0)
    if not 0.0 <= neg_hard_frac <= 1.0:
        raise ValueError("--neg-hard-frac must be in [0, 1]")
    if neg_sample == "adaptive_mix" and int(getattr(args, "negatives", 0) or 0) < 1:
        raise ValueError("--neg-sample adaptive_mix requires --negatives >= 1")
    neg_probe_mult = float(getattr(args, "neg_probe_mult", 4.0) or 4.0)
    if neg_probe_mult < 1.0:
        raise ValueError("--neg-probe-mult must be >= 1")
    visit_bins = int(getattr(args, "visit_bins", DEFAULT_VISIT_BINS))
    # Optional overlays (default off).
    candidate_context = getattr(args, "candidate_context", "off")
    candidate_context_time = getattr(args, "candidate_context_time", "off")
    session_summary = getattr(args, "session_summary", "off")
    user_pref = getattr(args, "user_pref", "off")
    user_id_emb = getattr(args, "user_id_emb", "off")
    transition_pmi = getattr(args, "transition_pmi", "off")
    expert_cross_attn = getattr(args, "expert_cross_attn", "off")
    expert_cross_attn_sparse = getattr(args, "expert_cross_attn_sparse", "on")
    if user_pref == "on" and not causal_counts:
        raise ValueError("--user-pref on requires causal counts (omit --no-causal-counts)")
    if candidate_context_time == "causal_slot" and not causal_counts:
        raise ValueError("--candidate-context-time causal_slot requires causal counts")
    if transition_pmi == "on" and not causal_counts:
        raise ValueError("--transition-pmi on requires causal counts")
    if transition_score_bias > 0 and not causal_counts:
        raise ValueError("--transition-score-bias > 0 requires causal counts")
    if category_score_bias > 0 and not causal_counts:
        raise ValueError("--category-score-bias > 0 requires causal counts")
    if pop_score_bias > 0 and not causal_counts:
        raise ValueError("--pop-score-bias > 0 requires causal counts")
    if geo_score_bias > 0 and candidate_context == "off":
        raise ValueError("--geo-score-bias > 0 requires --candidate-context on")
    if fam_area_blend == "on" and candidate_context == "off":
        raise ValueError("--fam-area-blend requires --candidate-context on")
    if gauss_geo_bias > 0 and candidate_context == "off":
        raise ValueError("--gauss-geo-bias > 0 requires --candidate-context on")
    if radiation_score_bias > 0 and candidate_context == "off":
        raise ValueError("--radiation-score-bias > 0 requires --candidate-context on")
    if radiation_score_bias > 0 and not causal_counts:
        raise ValueError("--radiation-score-bias > 0 requires causal counts")
    if jump_length_bias > 0 and candidate_context == "off":
        raise ValueError("--jump-length-bias > 0 requires --candidate-context on")
    if cat_stage_bias > 0 and candidate_context == "off":
        raise ValueError("--cat-stage-bias > 0 requires --candidate-context on")
    if cat_stage_bias > 0 and not causal_counts:
        raise ValueError("--cat-stage-bias > 0 requires causal counts")
    # Jump lognormal: CLI override, else fit from cache distance_stats when bias>0.
    jump_mu_arg = getattr(args, "jump_mu", None)
    jump_sigma_arg = getattr(args, "jump_sigma", None)
    if jump_length_bias > 0 and jump_dist == "lognormal" and (
            jump_mu_arg is None or jump_sigma_arg is None):
        stats = (cache.candidate_construction or {}).get("distance_stats") or {}
        fit_mu, fit_sigma = jump_params_from_distance_stats(stats)
        jump_mu = fit_mu if jump_mu_arg is None else float(jump_mu_arg)
        jump_sigma = fit_sigma if jump_sigma_arg is None else float(jump_sigma_arg)
        print(f"jump_length fit from distance_stats: mu={jump_mu:.4f} "
              f"sigma={jump_sigma:.4f} (median_km={stats.get('median_km')} "
              f"mean_km={stats.get('mean_km')})", flush=True)
    else:
        jump_mu = 0.0 if jump_mu_arg is None else float(jump_mu_arg)
        jump_sigma = 1.0 if jump_sigma_arg is None else float(jump_sigma_arg)
    if jump_sigma <= 0:
        raise ValueError("--jump-sigma must be positive")
    surd_bias_gate = getattr(args, "surd_bias_gate", "off")
    surd_score_bias = float(getattr(args, "surd_score_bias", 0.0) or 0.0)
    tsb_gate_mode = getattr(args, "tsb_gate_mode", "off")
    surd_attn_gate = getattr(args, "surd_attn_gate", "off")
    if surd_attn_gate != "off" and not causal_counts:
        raise ValueError("--surd-attn-gate ≠ off requires causal counts")
    if surd_attn_gate != "off" and transition_attn != "bias":
        raise ValueError("--surd-attn-gate only applies to --transition-attn bias")
    cfg = ModelConfig(args.dim, args.heads, args.layers, args.dropout, args.top_k,
                      args.temperature, args.lambda_prior, use_surd=not args.no_surd,
                      router_mode=args.router_mode, hybrid_lambda=args.hybrid_lambda,
                      causal_counts=causal_counts, transition_attn=transition_attn,
                      transition_lambda=transition_lambda,
                      transition_score_bias=transition_score_bias,
                      category_score_bias=category_score_bias,
                      pop_score_bias=pop_score_bias,
                      geo_score_bias=geo_score_bias,
                      geo_score_power=geo_score_power,
                      gauss_geo_bias=gauss_geo_bias,
                      gauss_sigma_km=gauss_sigma_km,
                      geo_adapt=geo_adapt,
                      geo_adapt_temp=geo_adapt_temp,
                      revisit_score_bias=revisit_score_bias,
                      history_hit_bias=history_hit_bias,
                      history_hit_k=history_hit_k,
                      revisit_recency_bias=revisit_recency_bias,
                      revisit_recency_tau=revisit_recency_tau,
                      radiation_score_bias=radiation_score_bias,
                      radiation_kernel=radiation_kernel,
                      radiation_soft_km=radiation_soft_km,
                      jump_length_bias=jump_length_bias,
                      jump_dist=jump_dist,
                      jump_mu=jump_mu,
                      jump_sigma=jump_sigma,
                      jump_beta=jump_beta,
                      jump_d0=jump_d0,
                      return_gate=return_gate,
                      explore_spatial_bias=explore_spatial_bias,
                      explore_score_bias=explore_score_bias,
                      fam_area_blend=fam_area_blend,
                      fam_radius_km=fam_radius_km,
                      fam_explore_bias=fam_explore_bias,
                      fam_return_bias=fam_return_bias,
                      tsb_history_k=tsb_history_k,
                      tsb_history_decay=tsb_history_decay,
                      tsb_dt_edges=tsb_dt_edges,
                      cat_stage_bias=cat_stage_bias,
                      transition_2hop_bias=transition_2hop_bias,
                      hour_pop_score_bias=hour_pop_score_bias,
                      tsb_decay_hours=tsb_decay_hours,
                      tsb_session_gap_hours=tsb_session_gap_hours,
                      visit_bins=visit_bins,
                      candidate_context=candidate_context,
                      candidate_context_time=candidate_context_time,
                      session_summary=session_summary, user_pref=user_pref,
                      user_id_emb=user_id_emb,
                      transition_pmi=transition_pmi,
                      expert_cross_attn=expert_cross_attn,
                      expert_cross_attn_sparse=expert_cross_attn_sparse,
                      surd_bias_gate=surd_bias_gate,
                      surd_score_bias=surd_score_bias,
                      tsb_gate_mode=tsb_gate_mode,
                      surd_attn_gate=surd_attn_gate)
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    batches = torch.Generator().manual_seed(args.seed)
    model = cache.model(cfg).to(device)
    ema_teacher = None
    if use_ema_soft:
        ema_teacher = copy.deepcopy(model)
        ema_teacher.eval()
        for p in ema_teacher.parameters():
            p.requires_grad_(False)
        print(f"ema_soft_teacher decay={ema_decay} beta={soft_target_beta} "
              f"priors={','.join(soft_priors)}", flush=True)
    kd_teacher = None
    if kd_teacher_path:
        kd_ckpt = load_file(kd_teacher_path)
        kd_cfg_raw = kd_ckpt.get("config") or kd_ckpt.get("model_config")
        if kd_cfg_raw is None:
            raise ValueError(f"kd-teacher checkpoint missing config: {kd_teacher_path}")
        kd_cfg = model_config_from_checkpoint(kd_cfg_raw)
        kd_teacher = cache.model(kd_cfg).to(device)
        load_weights(kd_teacher, kd_ckpt["weights"])
        kd_teacher.eval()
        for p in kd_teacher.parameters():
            p.requires_grad_(False)
        print(f"kd_teacher={kd_teacher_path} lambda_kd={lambda_kd} "
              f"top_k={kd_top_k} temp={kd_temp}", flush=True)
    geo_edge_src = geo_edge_dst = geo_edge_w = None
    if poi_geo_laplace > 0:
        if getattr(cache, "poi_coords", None) is None:
            raise ValueError("--poi-geo-laplace requires cache poi_coordinates")
        coords = cache.poi_coords.detach().to(dtype=torch.float64)
        geo_edge_src, geo_edge_dst, geo_edge_w = build_geo_knn_edges(coords, geo_knn)
        geo_edge_src = geo_edge_src.to(device)
        geo_edge_dst = geo_edge_dst.to(device)
        geo_edge_w = geo_edge_w.to(device)
        print(f"poi_geo_laplace={poi_geo_laplace} geo_knn={geo_knn} "
              f"edges={int(geo_edge_src.numel())}", flush=True)
    if lambda_cat_aux > 0:
        print(f"lambda_cat_aux={lambda_cat_aux} (next-cat MTL #45)", flush=True)
    param_device = next(model.parameters()).device
    print(f"model_param_device={param_device} router_mode={cfg.router_mode} "
          f"hybrid_lambda={cfg.hybrid_lambda} lambda_prior={cfg.lambda_prior} "
          f"causal_counts={cfg.causal_counts} transition_attn={cfg.transition_attn} "
          f"transition_score_bias={cfg.transition_score_bias} "
          f"category_score_bias={cfg.category_score_bias} "
          f"pop_score_bias={cfg.pop_score_bias} geo_score_bias={cfg.geo_score_bias} "
          f"geo_score_power={cfg.geo_score_power} "
          f"gauss_geo_bias={cfg.gauss_geo_bias} gauss_sigma_km={cfg.gauss_sigma_km} "
          f"geo_adapt={cfg.geo_adapt} geo_adapt_temp={cfg.geo_adapt_temp} "
          f"revisit_score_bias={cfg.revisit_score_bias} "
          f"history_hit_bias={cfg.history_hit_bias} history_hit_k={cfg.history_hit_k} "
          f"revisit_recency_bias={cfg.revisit_recency_bias} "
          f"revisit_recency_tau={cfg.revisit_recency_tau} "
          f"radiation_score_bias={cfg.radiation_score_bias} "
          f"radiation_kernel={cfg.radiation_kernel} "
          f"jump_length_bias={cfg.jump_length_bias} jump_dist={cfg.jump_dist} "
          f"jump_mu={cfg.jump_mu} jump_sigma={cfg.jump_sigma} "
          f"return_gate={cfg.return_gate} explore_spatial_bias={cfg.explore_spatial_bias} "
          f"explore_score_bias={cfg.explore_score_bias} "
          f"tsb_history_k={cfg.tsb_history_k} tsb_history_decay={cfg.tsb_history_decay} "
          f"tsb_dt_edges={cfg.tsb_dt_edges} "
          f"cat_stage_bias={cfg.cat_stage_bias} "
          f"tsb_decay_hours={cfg.tsb_decay_hours} "
          f"tsb_session_gap_hours={cfg.tsb_session_gap_hours} "
          f"surd_bias_gate={cfg.surd_bias_gate} surd_score_bias={cfg.surd_score_bias} "
          f"tsb_gate_mode={cfg.tsb_gate_mode} surd_attn_gate={cfg.surd_attn_gate} "
          f"visit_bins={cfg.visit_bins} candidate_context={cfg.candidate_context} "
          f"session_summary={cfg.session_summary} user_pref={cfg.user_pref} "
          f"user_id_emb={cfg.user_id_emb} n_users={getattr(model, 'n_users', 0)} "
          f"expert_cross_attn={cfg.expert_cross_attn}", flush=True)
    if param_device.type != device.type or (
            device.index is not None and param_device.index not in (None, device.index)):
        raise RuntimeError(f"model landed on {param_device}, expected {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    data, val = cache.split("train"), cache.split("validation")
    if transition_2hop_bias > 0:
        print(f"building transition A² log1p (n_pois={len(cache.pois)}) …", flush=True)
        # ModelConfig is frozen; stash runtime tensor via object.__setattr__.
        object.__setattr__(
            cfg, "_transition_a2_log1p",
            build_transition_a2_log1p(data, len(cache.pois)))
        print(f"transition_2hop_bias={transition_2hop_bias}", flush=True)
    if hour_pop_score_bias > 0:
        print(f"hour_pop_score_bias={hour_pop_score_bias}", flush=True)
    # Shared train-all SURD table for train / val / test (no fold-specific priors).
    final = cache.table("final", device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best, history, best_metrics = -math.inf, [], None
    settings = {key: value for key, value in vars(args).items() if isinstance(value, (str, float, int, bool))}
    settings["candidate_radius"] = cache.cfg.get("candidate_radius", "full")
    settings["candidate_radius_km"] = cache.radius_km
    settings["popular_radius_km"] = cache.popular_radius_km
    settings["random_negatives"] = cache.random_negatives
    settings["max_train_candidates"] = args.max_train_candidates
    settings["causal_counts"] = cfg.causal_counts
    settings["transition_attn"] = cfg.transition_attn
    settings["transition_lambda"] = cfg.transition_lambda
    settings["transition_score_bias"] = cfg.transition_score_bias
    settings["category_score_bias"] = cfg.category_score_bias
    settings["pop_score_bias"] = cfg.pop_score_bias
    settings["geo_score_bias"] = cfg.geo_score_bias
    settings["geo_score_power"] = cfg.geo_score_power
    settings["gauss_geo_bias"] = cfg.gauss_geo_bias
    settings["gauss_sigma_km"] = cfg.gauss_sigma_km
    settings["geo_adapt"] = cfg.geo_adapt
    settings["geo_adapt_temp"] = cfg.geo_adapt_temp
    settings["revisit_score_bias"] = cfg.revisit_score_bias
    settings["history_hit_bias"] = cfg.history_hit_bias
    settings["history_hit_k"] = cfg.history_hit_k
    settings["revisit_recency_bias"] = cfg.revisit_recency_bias
    settings["revisit_recency_tau"] = cfg.revisit_recency_tau
    settings["radiation_score_bias"] = cfg.radiation_score_bias
    settings["radiation_kernel"] = cfg.radiation_kernel
    settings["radiation_soft_km"] = cfg.radiation_soft_km
    settings["jump_length_bias"] = cfg.jump_length_bias
    settings["jump_dist"] = cfg.jump_dist
    settings["jump_mu"] = cfg.jump_mu
    settings["jump_sigma"] = cfg.jump_sigma
    settings["jump_beta"] = cfg.jump_beta
    settings["jump_d0"] = cfg.jump_d0
    settings["return_gate"] = cfg.return_gate
    settings["explore_spatial_bias"] = cfg.explore_spatial_bias
    settings["explore_score_bias"] = cfg.explore_score_bias
    settings["tsb_history_k"] = cfg.tsb_history_k
    settings["tsb_history_decay"] = cfg.tsb_history_decay
    settings["tsb_dt_edges"] = list(cfg.tsb_dt_edges)
    settings["cat_stage_bias"] = cfg.cat_stage_bias
    settings["transition_2hop_bias"] = cfg.transition_2hop_bias
    settings["hour_pop_score_bias"] = cfg.hour_pop_score_bias
    settings["tsb_decay_hours"] = cfg.tsb_decay_hours
    settings["tsb_session_gap_hours"] = cfg.tsb_session_gap_hours
    settings["surd_bias_gate"] = cfg.surd_bias_gate
    settings["surd_score_bias"] = cfg.surd_score_bias
    settings["tsb_gate_mode"] = cfg.tsb_gate_mode
    settings["surd_attn_gate"] = cfg.surd_attn_gate
    settings["visit_bins"] = cfg.visit_bins
    settings["candidate_context"] = cfg.candidate_context
    settings["candidate_context_time"] = cfg.candidate_context_time
    settings["session_summary"] = cfg.session_summary
    settings["user_pref"] = cfg.user_pref
    settings["user_id_emb"] = cfg.user_id_emb
    settings["transition_pmi"] = cfg.transition_pmi
    settings["expert_cross_attn"] = cfg.expert_cross_attn
    settings["expert_cross_attn_sparse"] = cfg.expert_cross_attn_sparse
    settings["neg_sample"] = neg_sample
    settings["neg_hard_frac"] = neg_hard_frac
    settings["neg_probe_mult"] = neg_probe_mult
    settings["soft_target_beta"] = soft_target_beta
    settings["soft_target_prior"] = ",".join(soft_priors) if soft_priors else "off"
    settings["ema_decay"] = ema_decay if use_ema_soft else None
    # Default on: test after every epoch (in addition to final best-ckpt test).
    eval_test_every_epoch = bool(getattr(args, "eval_test_every_epoch", True))
    settings["eval_test_every_epoch"] = eval_test_every_epoch
    max_train_candidates = None if args.max_train_candidates < 0 else args.max_train_candidates
    if neg_sample == "adaptive_mix":
        print(f"neg_sample=adaptive_mix hard_frac={neg_hard_frac} "
              f"probe_mult={neg_probe_mult} negatives={args.negatives}", flush=True)
    category_frequency = Counter(data["C"].tolist())
    test = cache.split("test")
    pool_stats, pools_by_split = report_candidate_pool_stats(
        cache, {"train": data, "validation": val, "test": test})
    train_pools = pools_by_split["train"]
    # Mode category per POI (runtime; no prepare) for candidate-context / typed queries.
    if needs_side_features(cfg):
        ensure_poi_cat_table(cache, train_data=data)
    # Causal online tables: fit visit-bin edges on the train stream once; freeze for val/test.
    online_tables = None
    visit_bin_edges = None
    if cfg.causal_counts:
        online_tables = new_count_tables(cache, cfg)
        visit_bin_edges = online_tables.fit_visit_bins(data, cfg.visit_bins)
        print(f"causal_counts=on transition_attn={cfg.transition_attn} "
              f"visit_bins={cfg.visit_bins} bin_edges={visit_bin_edges.tolist()} "
              f"(train chronological, no shuffle; val/test start from train-end tables)",
              flush=True)
    checkpoint_metric = str(getattr(args, "checkpoint_metric", "ndcg10") or "ndcg10")
    if checkpoint_metric not in ("ndcg10", "acc1"):
        raise ValueError(f"unsupported --checkpoint-metric {checkpoint_metric!r}")
    settings["checkpoint_metric"] = checkpoint_metric
    selection_tag = (
        "best_checkpoint_by_Acc@1" if checkpoint_metric == "acc1"
        else "best_checkpoint_by_NDCG@10")
    if eval_test_every_epoch:
        print("eval_test_every_epoch=on (per-epoch test → results/.../eval_test/epoch_*.json; "
              f"checkpoint selected by val {checkpoint_metric})", flush=True)
    else:
        print("eval_test_every_epoch=off (final best-ckpt test only)", flush=True)
    print(f"checkpoint_metric={checkpoint_metric} → {selection_tag}", flush=True)
    # Persist pool-size audit under results/ (not beside checkpoints).
    stats_dir = results_run_dir(out)
    stats_dir.mkdir(parents=True, exist_ok=True)
    dump_json(dict(
        candidate_construction=cache.candidate_construction,
        catalog_size=len(cache.pois),
        cache=str(Path(args.cache).resolve()),
        cache_id=cache.id,
        train_sampling=dict(
            negatives=args.negatives,
            max_train_candidates=args.max_train_candidates,
            neg_sample=neg_sample,
            neg_hard_frac=neg_hard_frac if neg_sample == "adaptive_mix" else None,
            neg_probe_mult=neg_probe_mult if neg_sample == "adaptive_mix" else None,
            eval_uses_full_pool=True,
        ),
        causal_counts=dict(
            enabled=cfg.causal_counts,
            transition_attn=cfg.transition_attn,
            transition_lambda=cfg.transition_lambda,
            transition_score_bias=cfg.transition_score_bias,
            category_score_bias=cfg.category_score_bias,
            pop_score_bias=cfg.pop_score_bias,
            geo_score_bias=cfg.geo_score_bias,
            geo_score_power=cfg.geo_score_power,
            visit_bins=cfg.visit_bins,
            visit_bin_edges=None if visit_bin_edges is None else visit_bin_edges.tolist(),
        ),
        overlays=dict(
            candidate_context=cfg.candidate_context,
            candidate_context_time=cfg.candidate_context_time,
            session_summary=cfg.session_summary,
            user_pref=cfg.user_pref,
            user_id_emb=cfg.user_id_emb,
            transition_pmi=cfg.transition_pmi,
            expert_cross_attn=cfg.expert_cross_attn,
            expert_cross_attn_sparse=cfg.expert_cross_attn_sparse,
        ),
        definition=("per-sample pool size after multi-source union "
                    "(neighborhood + popular + user-frequent + random negatives) "
                    "+ force-include in-catalog target (always, including val/test) "
                    "+ empty-union fallback to full catalog; stats match the full "
                    "constructed pool used at evaluation (train may subsample)"),
        splits=pool_stats,
    ), stats_dir / "candidate_pool_stats.json")
    print(f"Wrote {stats_dir / 'candidate_pool_stats.json'}", flush=True)
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="epochs", dynamic_ncols=True)
    train_end_state = None
    results_history = []
    test_eval_dir = resolve_train_eval_dir(out, "test")
    for epoch in epoch_bar:
        check_interrupt()
        model.train()
        loss_sum, components = 0., torch.zeros(4, dtype=torch.float64)
        counts = torch.zeros(15, dtype=torch.long, device=device)
        if cfg.causal_counts:
            online_tables.reset()
            # Chronological batches (stable by prediction timestamp); no shuffle.
            steps = list(chronological_indices(data).split(args.batch_size))
        else:
            steps = list(torch.randperm(len(data["end"]), generator=batches).split(args.batch_size))
        for idx in tqdm(steps, desc=f"train ep{epoch}", leave=False, dynamic_ncols=True):
            check_interrupt()
            h, valid, target = device_batch(data, idx, device)
            batch_pools = [train_pools[i] for i in idx.tolist()]
            if neg_sample == "adaptive_mix":
                # Probe a larger random pool subset, score no-grad, then mix hard+random.
                probe_n = max(int(args.negatives), int(math.ceil(args.negatives * neg_probe_mult)))
                probe_cap = None
                if max_train_candidates is not None:
                    probe_cap = max(max_train_candidates, probe_n + 1)
                probe_rows, _, probe_mask = sample_candidates(
                    target.detach().cpu(), len(cache.pois), probe_n, rng,
                    pools=batch_pools, max_candidates=probe_cap)
                probe_ctx = None
                if cfg.causal_counts or needs_side_features(cfg):
                    probe_lists = masked_candidate_lists(probe_rows, probe_mask)
                    # Read-only probe (no table update yet).
                    probe_ctx = gather_features_causal(
                        cfg, cache, data, idx, probe_lists, int(valid.shape[1]),
                        online_tables, update=False, train_data=data)
                    probe_ctx = count_ctx_to_device(probe_ctx, device)
                pr = probe_rows.to(device)
                pm = probe_mask.to(device)
                safe_probe = pr.masked_fill(~pm, -1)
                with torch.no_grad():
                    probe_prior, probe_conf = final(safe_probe)
                    probe_out = model(
                        h, valid, pr, probe_prior, probe_conf, candidate_mask=pm,
                        count_ctx=probe_ctx,
                        surd_specific=lookup_surd_specific(final, safe_probe, cfg))
                    probe_scores = probe_out["score"].detach().cpu()
                rows, positive, cand_mask = mix_adaptive_negatives(
                    probe_rows, probe_mask, probe_scores, target.detach().cpu(),
                    int(args.negatives), neg_hard_frac, rng)
            else:
                rows, positive, cand_mask = sample_candidates(
                    target.detach().cpu(), len(cache.pois), args.negatives, rng,
                    pools=batch_pools, max_candidates=max_train_candidates)
            count_ctx = None
            if cfg.causal_counts or needs_side_features(cfg):
                # Per-sample read from current tables, then update with GT next hop.
                cand_lists = masked_candidate_lists(rows, cand_mask)
                count_ctx = gather_features_causal(
                    cfg, cache, data, idx, cand_lists, int(valid.shape[1]),
                    online_tables, update=cfg.causal_counts, train_data=data)
                count_ctx = count_ctx_to_device(count_ctx, device)
            rows, cand_mask = rows.to(device), cand_mask.to(device)
            safe_rows = rows.masked_fill(~cand_mask, -1)
            prior, confidence = final(safe_rows)
            result = model(h, valid, rows, prior, confidence, candidate_mask=cand_mask,
                           count_ctx=count_ctx,
                           surd_specific=lookup_surd_specific(final, safe_rows, cfg))
            soft_tgts = None
            if soft_target_beta > 0 and soft_priors:
                mixes = []
                if count_priors:
                    mixes.append(soft_prior_targets(
                        positive.to(device), cand_mask, count_ctx, rows, valid,
                        count_priors, soft_target_beta))
                if use_ema_soft and ema_teacher is not None:
                    with torch.no_grad():
                        teacher_out = ema_teacher(
                            h, valid, rows, prior, confidence,
                            candidate_mask=cand_mask, count_ctx=count_ctx,
                            surd_specific=lookup_surd_specific(
                                final, safe_rows, cfg))
                        mixes.append(soft_ema_targets(
                            positive.to(device), cand_mask,
                            teacher_out["score"], soft_target_beta))
                if mixes:
                    soft_tgts = blend_soft_target_mixes(mixes, cand_mask)
            loss, parts = objective(
                result, positive.to(device), args.lambda_surd, args.lambda_balance,
                float(getattr(args, "lambda_bpr", 0.0) or 0.0),
                float(getattr(args, "bpr_margin", 0.0) or 0.0),
                float(getattr(args, "ce_focal_gamma", 0.0) or 0.0),
                str(getattr(args, "bpr_mode", "max") or "max"),
                soft_targets=soft_tgts)
            if kd_teacher is not None and lambda_kd > 0:
                with torch.no_grad():
                    teach_out = kd_teacher(
                        h, valid, rows, prior, confidence,
                        candidate_mask=cand_mask, count_ctx=count_ctx,
                        surd_specific=lookup_surd_specific(final, safe_rows, cfg))
                kd = top_k_kd_loss(
                    result["score"], teach_out["score"], cand_mask,
                    top_k=kd_top_k, temp=kd_temp)
                loss = loss + lambda_kd * kd
            if poi_geo_laplace > 0 and geo_edge_src is not None:
                # projections["P"] rows 0/1 are pad/unk; catalog rows start at index 2.
                poi_emb = model.projections["P"].weight[2:2 + len(cache.pois)]
                loss = loss + poi_geo_laplace_loss(
                    poi_emb, geo_edge_src, geo_edge_dst, geo_edge_w, poi_geo_laplace)
            lambda_cat_aux = float(getattr(args, "lambda_cat_aux", 0.0) or 0.0)
            if lambda_cat_aux > 0 and "cat_aux_logits" in result:
                # Next-event category catalog row (token−2); skip pad/unk.
                cat_labels = (data["cat_token"][data["end"][idx]] - 2).to(device)
                valid_cat = cat_labels >= 0
                if bool(valid_cat.any()):
                    n_cat = int(model.n_categories)
                    cat_y = cat_labels[valid_cat].clamp(0, n_cat - 1)
                    cat_loss = torch.nn.functional.cross_entropy(
                        result["cat_aux_logits"][valid_cat], cat_y)
                    loss = loss + lambda_cat_aux * cat_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            opt.step()
            if ema_teacher is not None:
                with torch.no_grad():
                    for tp, sp in zip(ema_teacher.parameters(), model.parameters()):
                        tp.mul_(ema_decay).add_(sp.data, alpha=1.0 - ema_decay)
                    for tb, sb in zip(ema_teacher.buffers(), model.buffers()):
                        tb.copy_(sb)
            counts += torch.bincount(result["top"].detach().reshape(-1), minlength=15)
            loss_sum += float(loss.detach()) * len(idx)
            components += torch.stack(parts).double().cpu() * len(idx)
        # Val starts from a clone of tables at end of this training epoch.
        val_tables = online_tables.clone() if cfg.causal_counts else None
        if cfg.causal_counts:
            train_end_state = online_tables.state_dict()
        # Pass category_frequency so val (and saved best) always include groups + Acc@1.
        metrics = evaluate(model, final, val, cache, args.eval_batch_size, args.chunk,
                           category_frequency, desc=f"val ep{epoch}", count_tables=val_tables)
        row = metrics["all_targets"]
        if checkpoint_metric == "acc1":
            score = row["Acc@1"]
        else:
            score = row["NDCG@10"]
        record = dict(epoch=epoch, loss=loss_sum / len(data["end"]),
                      loss_terms=dict(zip(
                          ("rec", "surd", "balance", "bpr"),
                          (components / len(data["end"])).tolist())),
                      dispatch_counts=dict(zip(EXPERT_ORDER, counts.cpu().tolist())), validation=metrics)
        results_epoch = dict(
            epoch=epoch,
            loss=record["loss"],
            validation=primary_metric_snapshot(metrics),
        )
        # Optional per-epoch test on *current* weights (not best ckpt); separate table clone.
        if eval_test_every_epoch:
            epoch_test_tables = online_tables.clone() if cfg.causal_counts else None
            epoch_test = evaluate(
                model, final, test, cache, args.eval_batch_size, args.chunk,
                category_frequency, desc=f"test ep{epoch}", count_tables=epoch_test_tables)
            record["test"] = epoch_test
            results_epoch["test"] = primary_metric_snapshot(epoch_test)
            test_eval_dir.mkdir(parents=True, exist_ok=True)
            epoch_path = test_eval_dir / f"epoch_{epoch:03d}.json"
            dump_json(dict(
                timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                epoch=epoch,
                split="test",
                selection="current_epoch_weights",
                cache=str(Path(args.cache).resolve()),
                cache_id=cache.id,
                primary_metrics=list(PRIMARY_METRICS),
                candidate_construction=cache.candidate_construction,
                candidate_pool_stats=pool_stats.get("test"),
                **epoch_test,
            ), epoch_path)
            print(f"Wrote {epoch_path}", flush=True)
            trow = epoch_test["all_targets"]
            print(f"epoch={epoch}  "
                  f"test Acc@1={trow['Acc@1']:.4f} NDCG@1={trow['NDCG@1']:.4f} "
                  f"MRR={trow['MRR']:.4f}  Acc@10={trow['Acc@10']:.4f} "
                  f"NDCG@10={trow['NDCG@10']:.4f}", flush=True)
        history.append(record)
        results_history.append(results_epoch)
        if score > best:
            best = score
            best_metrics = metrics
            payload = dict(cache_id=cache.id, config=asdict(cfg), weights=model.state_dict(),
                           epoch=epoch, validation_ndcg10=row["NDCG@10"],
                           validation_acc1=row["Acc@1"],
                           checkpoint_metric=checkpoint_metric,
                           checkpoint_score=score,
                           training_args=settings)
            if train_end_state is not None:
                payload["count_tables"] = train_end_state
                payload["visit_bin_edges"] = visit_bin_edges.tolist()
            save_checkpoint(payload, out / "best.pt")
        dump_json({"training_args": settings, "candidate_pool_stats": pool_stats,
                   "epochs": history}, out / "history.json")
        # Compact per-epoch val (+ optional test) history under results/.
        dump_json({"training_args": settings, "candidate_pool_stats": pool_stats,
                   "eval_test_every_epoch": eval_test_every_epoch,
                   "checkpoint_metric": checkpoint_metric,
                   "epochs": results_history}, stats_dir / "history.json")
        postfix = dict(loss=f"{record['loss']:.5f}", acc1=f"{row['Acc@1']:.4f}",
                       ndcg10=f"{row['NDCG@10']:.6f}", ckpt=f"{score:.6f}")
        if eval_test_every_epoch and "test" in results_epoch:
            postfix["test_acc1"] = f"{results_epoch['test']['Acc@1']:.4f}"
        epoch_bar.set_postfix(**postfix)
        print(f"epoch={epoch} loss={record['loss']:.5f}  "
              f"val Acc@1={row['Acc@1']:.4f} NDCG@1={row['NDCG@1']:.4f} MRR={row['MRR']:.4f}  "
              f"Acc@10={row['Acc@10']:.4f} NDCG@10={row['NDCG@10']:.4f}  "
              f"ckpt_score({checkpoint_metric})={score:.6f}", flush=True)
    if best_metrics is not None:
        best_path = out / "best.pt"
        val_payload = build_eval_payload(args, cache, best_metrics, device,
                                         checkpoint_path=best_path)
        val_payload["split"] = "validation"
        val_payload["selection"] = selection_tag
        val_payload["candidate_pool_stats"] = pool_stats.get("validation")
        print_eval_tables(val_payload)
        save_eval_results(val_payload, resolve_train_eval_dir(out, "validation"))
        # Final test once on the selected checkpoint (not the last-epoch weights).
        checkpoint = load_file(best_path)
        model.load_state_dict(checkpoint["weights"])
        model.eval()
        test_tables = None
        if cfg.causal_counts:
            # Test starts from train-end tables stored with the best checkpoint
            # (same snapshot as val would clone; independent of val updates).
            if checkpoint.get("count_tables") is not None:
                test_tables = new_count_tables(cache, cfg)
                test_tables.load_state_dict(checkpoint["count_tables"])
            else:
                edges = checkpoint.get("visit_bin_edges")
                test_tables = replay_train_tables(
                    cache, cfg, data,
                    visit_bin_edges=None if edges is None else torch.tensor(edges))
        test_metrics = evaluate(model, final, test, cache, args.eval_batch_size,
                                args.chunk, category_frequency, desc="eval test",
                                count_tables=test_tables)
        test_payload = build_eval_payload(args, cache, test_metrics, device,
                                          checkpoint_path=best_path, checkpoint=checkpoint)
        test_payload["split"] = "test"
        test_payload["selection"] = selection_tag
        test_payload["candidate_pool_stats"] = pool_stats.get("test")
        print_eval_tables(test_payload)
        save_eval_results(test_payload, resolve_train_eval_dir(out, "test"))
        # Auto-append one ablation table row (test metrics; best-val in notes).
        best_val = primary_metric_snapshot(best_metrics)
        notes = (
            f"best_val Acc@1={_metric_cell(best_val, 'Acc@1')} "
            f"NDCG@1={_metric_cell(best_val, 'NDCG@1')} "
            f"MRR={_metric_cell(best_val, 'MRR')}; "
            f"ckpt_epoch={checkpoint.get('epoch')}; "
            f"selection={selection_tag}"
        )
        append_ablation_log(
            ablation_row_from_eval(
                cache, test_metrics["all_targets"],
                run_name=run_name_for_results(out),
                cache_path=Path(args.cache).resolve(),
                key_flags=format_ablation_key_flags(cache, settings, asdict(cfg)),
                notes=notes,
            ),
            out,
        )


def load_model(args):
    cache, device = Cache(args.cache), select_device(args.device)
    checkpoint = load_file(args.checkpoint)
    if cache.id != checkpoint["cache_id"]:
        raise ValueError("checkpoint trained with a different cache; check --cache and --checkpoint")
    cfg = model_config_from_checkpoint(checkpoint["config"])
    model = cache.model(cfg).to(device)
    load_weights(model, checkpoint["weights"])
    print(f"model_param_device={next(model.parameters()).device} "
          f"router_mode={cfg.router_mode} causal_counts={cfg.causal_counts} "
          f"transition_attn={cfg.transition_attn} "
          f"candidate_context={cfg.candidate_context} "
          f"expert_cross_attn={cfg.expert_cross_attn}", flush=True)
    if needs_side_features(cfg):
        ensure_poi_cat_table(cache)
    return cache, model.eval(), cache.table("final", device), checkpoint


def train_end_count_tables(cache, cfg, checkpoint, train_data):
    """Tables at end of training: prefer checkpoint snapshot, else replay train."""
    if not cfg.causal_counts:
        return None
    if checkpoint.get("count_tables") is not None:
        tables = new_count_tables(cache, cfg)
        tables.load_state_dict(checkpoint["count_tables"])
        return tables
    edges = checkpoint.get("visit_bin_edges")
    return replay_train_tables(
        cache, cfg, train_data,
        visit_bin_edges=None if edges is None else torch.tensor(edges))


def report(value, dest):
    if dest:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        dump_json(value, dest)
        print("Wrote", dest)
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def evaluate_cli(args):
    install_interrupt_handler()
    cache, model, table, checkpoint = load_model(args)
    device = table.routing_prior.device
    cfg = apply_eval_bias_overrides(model, args)
    training = cache.split("train")
    category_frequency = Counter(training["C"].tolist())
    data = cache.split(args.split)
    pool_stats, _ = report_candidate_pool_stats(cache, {args.split: data})
    # Val/test: clone tables at end of training, then chronological score→update.
    count_tables = train_end_count_tables(cache, cfg, checkpoint, training)
    if count_tables is not None:
        print(f"causal_counts: loaded train-end tables "
              f"(visit_bins={cfg.visit_bins}); continuing chronological updates on "
              f"{args.split}", flush=True)
    if float(getattr(cfg, "transition_2hop_bias", 0.0) or 0.0) > 0:
        print(f"building transition A² log1p (n_pois={len(cache.pois)}) …", flush=True)
        object.__setattr__(
            cfg, "_transition_a2_log1p",
            build_transition_a2_log1p(training, len(cache.pois)))
    score_temp = float(getattr(args, "score_temperature", 1.0) or 1.0)
    if score_temp <= 0:
        raise ValueError("--score-temperature must be positive")
    tta_modes = parse_tta_modes(getattr(args, "tta", "off"))
    tta_keep_last = not bool(getattr(args, "tta_no_keep_last", False))
    tta_views = int(getattr(args, "tta_views", 1) or 1)
    tta_fuse = str(getattr(args, "tta_fuse", "mean") or "mean")
    score_calib = str(getattr(args, "score_calib", "off") or "off")
    calib_lambda = float(getattr(args, "calib_lambda", 0.0) or 0.0)
    if tta_views < 1:
        raise ValueError("--tta-views must be >= 1")
    if tta_modes:
        print(f"tta modes={','.join(tta_modes)} views={tta_views} "
              f"fuse={tta_fuse} keep_last={tta_keep_last}", flush=True)
    if score_calib not in ("", "off", "none"):
        print(f"score_calib={score_calib} lambda={calib_lambda}", flush=True)
    cand_prior = str(getattr(args, "candidate_spatial_prior", "off") or "off")
    cand_top_m = int(getattr(args, "candidate_prior_top_m", 0) or 0)
    cand_gbeta = float(getattr(args, "candidate_prior_gravity_beta", 1.0) or 1.0)
    mc_dropout = int(getattr(args, "mc_dropout", 0) or 0)
    mc_keep = bool(getattr(args, "mc_keep_train_dropout", False))
    if mc_dropout > 1:
        print(f"mc_dropout={mc_dropout} keep_train_dropout={mc_keep}", flush=True)
    metrics = evaluate(
        model, table, data, cache, args.eval_batch_size,
        args.chunk, category_frequency, desc=f"eval {args.split}",
        count_tables=count_tables, score_temperature=score_temp,
        tta_modes=tta_modes, tta_keep_last=tta_keep_last, tta_views=tta_views,
        tta_fuse=tta_fuse, score_calib=score_calib, calib_lambda=calib_lambda,
        candidate_spatial_prior=cand_prior, candidate_prior_top_m=cand_top_m,
        candidate_prior_gravity_beta=cand_gbeta,
        mc_dropout=mc_dropout, mc_keep_train_dropout=mc_keep)
    payload = build_eval_payload(args, cache, metrics, device,
                                 checkpoint_path=args.checkpoint, checkpoint=checkpoint)
    payload["candidate_pool_stats"] = pool_stats.get(args.split)
    payload["score_temperature"] = score_temp
    payload["eval_geo_score_bias"] = float(cfg.geo_score_bias)
    payload["eval_revisit_score_bias"] = float(cfg.revisit_score_bias)
    payload["tta"] = ",".join(tta_modes) if tta_modes else "off"
    payload["tta_fuse"] = tta_fuse if tta_modes else "off"
    payload["tta_views"] = tta_views if tta_modes else 0
    payload["score_calib"] = score_calib
    payload["calib_lambda"] = calib_lambda
    payload["candidate_spatial_prior"] = cand_prior
    payload["candidate_prior_top_m"] = cand_top_m
    payload["mc_dropout"] = mc_dropout
    if metrics.get("candidate_spatial_prior"):
        payload["candidate_prior_stats"] = metrics["candidate_spatial_prior"]
    print_eval_tables(payload)
    out_dir = resolve_eval_out(args)
    save_eval_results(payload, out_dir, legacy_output=args.output)
    # Append ablation row for standalone evaluate (same columns as train).
    ckpt_parent = Path(args.checkpoint).resolve().parent
    training_args = checkpoint.get("training_args") or {}
    model_cfg = checkpoint.get("config") or {}
    notes = f"evaluate --split {args.split}"
    if checkpoint.get("epoch") is not None:
        notes += f"; ckpt_epoch={checkpoint['epoch']}"
    if checkpoint.get("validation_ndcg10") is not None:
        notes += f"; ckpt_val_NDCG@10={checkpoint['validation_ndcg10']:.6f}"
    append_ablation_log(
        ablation_row_from_eval(
            cache, metrics["all_targets"],
            run_name=run_name_for_results(ckpt_parent),
            cache_path=Path(args.cache).resolve(),
            key_flags=format_ablation_key_flags(cache, training_args, model_cfg),
            notes=notes,
        ),
        ckpt_parent,
    )


@torch.inference_mode()
def explain(args):
    install_interrupt_handler()
    cache, model, table, checkpoint = load_model(args)
    cfg = model.cfg
    data = cache.split(args.split)
    i = args.sample_index
    if not 0 <= i < len(data["end"]):
        raise ValueError("sample-index outside split")
    device = table.routing_prior.device
    # Bring causal tables to the state just before sample i (train-end + prior samples).
    count_tables = train_end_count_tables(cache, cfg, checkpoint, cache.split("train"))
    count_ctx = None
    if count_tables is not None:
        order = chronological_indices(data)
        for j in order.tolist():
            if j == i:
                break
            count_tables.update_sample(data, j)
    h, valid, target = device_batch(data, torch.tensor([i]), device)
    pools, resolutions = cache.candidate_pools(data)
    pool, resolution = pools[i], resolutions[i]
    if count_tables is not None or needs_side_features(cfg):
        if cache.uses_full_catalog_pools():
            cand_lists = [list(range(len(cache.pois)))]
        else:
            cand_lists = [pool.tolist()]
        count_ctx = gather_features_causal(
            cfg, cache, data, torch.tensor([i]), cand_lists, int(valid.shape[1]),
            count_tables, update=False, train_data=cache.split("train"))
        count_ctx = count_ctx_to_device(count_ctx, device)
    encoded = model.encode(h, valid, count_ctx)
    if args.poi_id is not None and args.poi_id not in table.row_of:
        raise ValueError("candidate POI outside the training catalog")
    if cache.uses_full_catalog_pools():
        scores = all_scores(model, table, encoded, valid, len(cache.pois), args.chunk,
                            count_ctx=count_ctx)
        if args.poi_id is None:
            row = int(target[0]) if target[0] >= 0 else int(scores.argmax())
        else:
            row = table.row_of[args.poi_id]
        candidate_rank = int(ranks_of(scores, torch.tensor([row]))[0])
    else:
        pool_scores, pool_rows, pool_mask = score_pools(
            model, table, encoded, valid, [pool], args.chunk, count_ctx=count_ctx)
        if args.poi_id is None:
            if target[0] >= 0:
                row = int(target[0])
            else:
                active = pool_mask[0]
                row = int(pool_rows[0][active][pool_scores[0][active].argmax()])
        else:
            row = table.row_of[args.poi_id]
        candidate_rank = ranks_in_pools(pool_scores, pool_rows, pool_mask, torch.tensor([row]))[0]
    rows = torch.tensor([[row]], device=valid.device)
    prior, confidence = table(rows)
    # Single-candidate slice of count context for the detailed score path.
    detail_ctx = None
    if count_ctx is not None:
        # Find column of `row` inside the gathered candidate list, or rebuild.
        if cache.uses_full_catalog_pools():
            detail_ctx = slice_count_ctx(count_ctx, row, row + 1)
        else:
            pos = (pool == row).nonzero(as_tuple=False)
            if pos.numel():
                j = int(pos[0])
                detail_ctx = slice_count_ctx(count_ctx, j, j + 1)
            else:
                detail_ctx = None
    result = model.score(encoded, valid, rows, prior, confidence, details=True,
                         count_ctx=detail_ctx,
                         surd_specific=lookup_surd_specific(table, rows, model.cfg))
    def get(key):
        return result[key][0, 0].tolist()
    top = [EXPERT_ORDER[j] for j in get("top")]
    start, end = int(data["start"][i]), int(data["end"][i])
    raw_poi = cache.pois[row]
    user, trail = data["trails"][data["trail"][end]]
    record = dict(sample_index=i, split=args.split, user_id=user, trail_id=trail,
                  prefix_end_utc_us=int(data["us"][end - 1]), target_utc_us=int(data["us"][end]),
                  candidate_poi_id=raw_poi, target_poi_id=int(data["P"][end]), is_true_next_poi=row == int(target[0]),
                  history_poi_ids=data["P"][start:end].tolist(), history_category_ids=data["C"][start:end].tolist(),
                  history_time_features=h["T"][0].tolist(), history_location_features=h["L"][0].tolist(),
                  prior_version=cache.id, source_order=SOURCE_ORDER, atom_order=ATOM_ORDER, expert_order=EXPERT_ORDER,
                  candidate_construction=cache.candidate_construction,
                  candidate_pool_size=int(pool.numel()), candidate_pool_resolution=resolution,
                  target_count=int(table.target_count[row]), total_information_bits=float(table.total_information[row]),
                  near_tie_in_surd=bool(table.near_tie[row]),
                  exact_tie_in_route=len(set(get("logits"))) < len(EXPERT_ORDER),
                  raw_surd=table.raw_surd[row].tolist(), normalized_surd=table.normalized_surd[row].tolist(),
                  routing_prior=prior[0, 0].tolist(), confidence=float(confidence),
                  effective_confidence=get("confidence"), us_mass=get("mass"),
                  alpha=get("alpha"), redundancy_mass=float(prior[..., 4:15].sum()), route_mlp_logits=get("learned"),
                  explicit_prior_bias=get("bias"), final_route_logits=get("logits"),
                  dense_route_probability=result["log_pi"][0, 0].exp().tolist(), selected_experts=top,
                  selected_weights=get("weights"), gate=get("gate"), base_score=get("base"),
                  expert_projected_scores=dict(zip(top, get("projected"))),
                  expert_score_contributions=dict(zip(top, get("contributions"))),
                  moe_bias_contribution=get("moe_bias"), final_score=get("score"),
                  candidate_rank=candidate_rank)
    report(record, args.output)


@torch.inference_mode()
def recommend(args):
    install_interrupt_handler()
    cache, model, table, checkpoint = load_model(args)
    cfg = model.cfg
    groups, _ = read_split(args.history)
    cats = Categories.from_manifest(cache.meta["preprocessing"])
    data, _ = build_split(
        groups, cats, cache.pois, cache.meta["cat_vocab"], cache.cfg["max_length"], "predict")
    pools, resolutions = cache.candidate_pools(data)
    # Use frozen train-end tables (no GT updates on recommend histories).
    base_tables = train_end_count_tables(cache, cfg, checkpoint, cache.split("train"))
    records = []
    batches = list(torch.arange(len(data["end"])).split(args.eval_batch_size))
    for idx in tqdm(batches, desc="recommend", dynamic_ncols=True):
        check_interrupt()
        h, valid, _ = device_batch(data, idx, table.routing_prior.device, labels=False)
        count_ctx = None
        if base_tables is not None or needs_side_features(cfg):
            if cache.uses_full_catalog_pools():
                cand_lists = [list(range(len(cache.pois))) for _ in idx.tolist()]
            else:
                cand_lists = [pools[i].tolist() for i in idx.tolist()]
            # Read-only: do not mutate the shared train-end snapshot across batches.
            count_ctx = gather_features_causal(
                model.cfg, cache, data, idx, cand_lists, int(valid.shape[1]),
                base_tables, update=False, train_data=cache.split("train"))
            count_ctx = count_ctx_to_device(count_ctx, table.routing_prior.device)
        encoded = model.encode(h, valid, count_ctx)
        if cache.uses_full_catalog_pools():
            scores = all_scores(model, table, encoded, valid, len(cache.pois), args.chunk,
                                count_ctx=count_ctx)
            top = torch.argsort(scores, descending=True, stable=True)[:, :args.top_n]
            for i, js, row_scores in zip(idx.tolist(), top.tolist(), scores):
                user, trail = data["trails"][data["trail"][data["end"][i] - 1]]
                records.append(dict(user_id=user, trail_id=trail,
                                    candidate_pool_resolution=resolutions[i],
                                    recommendations=[
                                        dict(venue_id=cache.pois[j], score=float(row_scores[j]))
                                        for j in js]))
        else:
            batch_pools = [pools[i] for i in idx.tolist()]
            pool_scores, pool_rows, pool_mask = score_pools(
                model, table, encoded, valid, batch_pools, args.chunk, count_ctx=count_ctx)
            for bi, i in enumerate(idx.tolist()):
                active = pool_mask[bi]
                scores = pool_scores[bi][active]
                rows = pool_rows[bi][active]
                order = torch.argsort(scores, descending=True, stable=True)[:args.top_n]
                user, trail = data["trails"][data["trail"][data["end"][i] - 1]]
                records.append(dict(
                    user_id=user, trail_id=trail,
                    candidate_pool_resolution=resolutions[i],
                    candidate_pool_size=int(active.sum()),
                    recommendations=[
                        dict(venue_id=cache.pois[int(rows[j])], score=float(scores[j]))
                        for j in order.tolist()]))
    report(records, args.output)
