"""City scale helpers: large-dataset set + optional fast model structure.

``large_city_fast`` (CLI / YAML) shrinks expensive WIN structure on Bandung-or-
larger Massive-STEPS cities only. Small cities ignore the flag.
"""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping

# Massive-STEPS cities **≥ Bandung** by raw data/ CSV footprint (GPU box 2026-11-26).
# f63 test protocol: these use Bandung-style mix top-512/epoch + final full-pool;
# smaller cities use full constructed test every epoch.
LARGE_CITIES = frozenset({
    "bandung",
    "kuwait_city",
    "sao_paulo",
    "new_york",
    "jakarta",
    "petaling_jaya",
    "istanbul",
})

# Structure overrides applied when large_city_fast=on *and* city ∈ LARGE_CITIES.
# Keeps candidate_context=on so geo/jump priors still work; drops heavy overlays.
FAST_STRUCTURE: dict[str, Any] = {
    "dim": 128,
    "heads": 4,
    "layers": 2,
    "transition_attn": "bias",
    "candidate_context_time": "off",
    "session_summary": "off",
}

# Val + test candidate-pool shrink under large_city_fast.
# ``mix`` = geo + global pop + user-frequent + history revisit + tip-hour POI.
# GT is always force-included (#10). Acc@1 becomes shortlist Acc.
FAST_EVAL_POOL: dict[str, Any] = {
    "candidate_spatial_prior": "mix",
    "candidate_prior_top_m": 512,
}


def is_large_city(city: str | None) -> bool:
    return bool(city) and str(city).strip().lower() in LARGE_CITIES


def resolve_city_name(args: Any, cache: Any | None = None) -> str | None:
    city = getattr(args, "city", None)
    if city:
        return str(city).strip().lower()
    if cache is not None:
        cfg = getattr(cache, "cfg", None) or {}
        if isinstance(cfg, Mapping) and cfg.get("city"):
            return str(cfg["city"]).strip().lower()
        meta = getattr(cache, "meta", None) or {}
        if isinstance(meta, Mapping) and meta.get("city"):
            return str(meta["city"]).strip().lower()
    # Infer from --out / --cache path: runs/<city>/...
    for attr in ("out", "cache"):
        raw = getattr(args, attr, None)
        if not raw:
            continue
        parts = str(raw).replace("\\", "/").split("/")
        if "runs" in parts:
            i = parts.index("runs")
            if i + 1 < len(parts) and parts[i + 1]:
                return parts[i + 1].strip().lower()
        if "configs" in parts:
            i = parts.index("configs")
            if i + 1 < len(parts) and parts[i + 1]:
                return parts[i + 1].strip().lower()
    return None


def large_city_fast_enabled(args: Any) -> bool:
    raw = getattr(args, "large_city_fast", "off")
    if isinstance(raw, bool):
        return raw
    return str(raw or "off").lower() in ("on", "true", "1", "yes")


def resolve_final_negatives(args: Any) -> int:
    """Non-GT shortlist count: ``final_negatives``, else ``final_candidates-1``."""
    n = int(getattr(args, "final_negatives", 0) or 0)
    if n > 0:
        return n
    total = int(getattr(args, "final_candidates", 0) or 0)
    if total > 1:
        return total - 1
    if total == 1:
        return 0
    return 0


def apply_large_city_fast(
    args: Any,
    *,
    city: str | None,
    locals_map: MutableMapping[str, Any],
) -> dict[str, Any]:
    """Mutate train locals (and args mirrors) for fast structure when gated on.

    ``locals_map`` should contain keys used to build ``ModelConfig``:
    dim/heads/layers come from ``args``; transition_attn / candidate_context_time /
    session_summary are local variables in ``train()``.

    Also fills default eval pool shrink (``FAST_EVAL_POOL``) when spatial prior
    is still off / top-M unset — applied to both validation and test.
    """
    report: dict[str, Any] = {
        "requested": large_city_fast_enabled(args),
        "city": city,
        "applied": False,
        "changes": {},
        "eval_pool": {},
    }
    if not report["requested"]:
        return report
    if not is_large_city(city):
        report["skipped"] = "city_not_large"
        return report

    before: dict[str, Any] = {}
    after: dict[str, Any] = {}

    # Backbone size lives on args.
    for key in ("dim", "heads", "layers"):
        if key not in FAST_STRUCTURE:
            continue
        old = getattr(args, key)
        new = FAST_STRUCTURE[key]
        before[key] = old
        if old != new:
            setattr(args, key, new)
            after[key] = new

    # Overlay modes are locals in train().
    for key in ("transition_attn", "candidate_context_time", "session_summary"):
        if key not in FAST_STRUCTURE or key not in locals_map:
            continue
        old = locals_map[key]
        new = FAST_STRUCTURE[key]
        before[key] = old
        if old != new:
            locals_map[key] = new
            after[key] = new

    # Eval pool shrink defaults (do not override explicit user settings).
    # final_negatives → train/val shortlist; test may still use test_prior_top_m.
    final_neg = resolve_final_negatives(args)
    prior = str(getattr(args, "candidate_spatial_prior", "off") or "off").lower()
    top_m = int(getattr(args, "candidate_prior_top_m", 0) or 0)
    eval_pool: dict[str, Any] = {}
    if final_neg > 0 or int(getattr(args, "final_candidates", 0) or 0) > 0:
        test_m = resolve_test_prior_top_m(args, city)
        eval_pool = {
            "final_negatives": final_neg,
            "final_extra_random": int(getattr(args, "final_extra_random", -1)),
            "candidate_spatial_prior": "off",
            "candidate_prior_top_m": 0,
            "test_prior_top_m": test_m,
            "test_spatial_prior": resolve_test_spatial_prior(args, top_m=test_m),
        }
    elif prior in ("", "off", "none") and top_m <= 0:
        for key, new in FAST_EVAL_POOL.items():
            old = getattr(args, key, None)
            before[key] = old
            setattr(args, key, new)
            after[key] = new
            eval_pool[key] = new
    else:
        eval_pool = {
            "candidate_spatial_prior": prior,
            "candidate_prior_top_m": top_m,
        }

    report["applied"] = True
    report["before"] = before
    report["after"] = {**before, **after}
    report["changes"] = after
    report["eval_pool"] = eval_pool
    # Back-compat alias used by older log lines.
    report["val_pool"] = eval_pool
    return report


def resolve_test_prior_top_m(args: Any, city: str | None = None) -> int:
    """Per-epoch test top-M (0 = full constructed pool).

    Explicit ``--test-prior-top-m`` wins. Otherwise, when train/val use the
    final_negatives shortlist and the city is **≥ Bandung** (``LARGE_CITIES``),
    default to ``FAST_EVAL_POOL`` top-M (512) — same as Bandung. Smaller
    cities default to 0 (full constructed test every epoch).
    """
    raw = getattr(args, "test_prior_top_m", None)
    if raw is not None and str(raw) != "":
        return max(0, int(raw))
    final_neg = resolve_final_negatives(args)
    if final_neg <= 0 and int(getattr(args, "final_candidates", 0) or 0) <= 0:
        return 0
    city_name = city if city is not None else resolve_city_name(args)
    if is_large_city(city_name):
        return int(FAST_EVAL_POOL["candidate_prior_top_m"])
    return 0


def resolve_test_spatial_prior(args: Any, *, top_m: int | None = None) -> str:
    """Spatial prior used with ``test_prior_top_m`` (default ``mix``)."""
    m = int(top_m) if top_m is not None else resolve_test_prior_top_m(args)
    if m < 1:
        return "off"
    raw = str(getattr(args, "test_spatial_prior", "") or "").lower()
    if raw in ("", "auto"):
        return str(FAST_EVAL_POOL["candidate_spatial_prior"])
    return raw


def final_test_full_pool_enabled(args: Any, city: str | None = None) -> bool:
    """Whether the post-train best-ckpt eval should use the full constructed pool.

    ``auto`` (default): full once when per-epoch test used top-M (ledger Acc).
    """
    raw = getattr(args, "final_test_full_pool", "auto")
    if isinstance(raw, bool):
        return raw
    s = str(raw or "auto").lower()
    if s in ("on", "true", "1", "yes", "full"):
        return True
    if s in ("off", "false", "0", "no", "top_m"):
        return False
    # auto
    return resolve_test_prior_top_m(args, city) > 0


def eval_pool_shrink_kwargs(
    args: Any,
    *,
    split: str = "validation",
    city: str | None = None,
    force_full: bool = False,
) -> dict[str, Any]:
    """Kwargs for ``evaluate()`` pool shrink (empty ⇒ full constructed pool).

    With ``final_negatives`` / ``final_candidates``:
    - **validation** (and train): shared 63+GT shortlist protocol.
    - **test**: no shortlist; optional ``test_prior_top_m`` mix shrink for
      large-city per-epoch checkpointing. ``force_full=True`` skips top-M
      (used for the rare post-train full-pool ledger eval).
    """
    final_neg = resolve_final_negatives(args)
    out: dict[str, Any] = {}
    hist_cap = int(getattr(args, "history_max_length", 0) or 0)
    if hist_cap > 0:
        out["history_max_length"] = hist_cap
    split_l = str(split or "validation").lower()
    if split_l in ("test", "eval_test"):
        if force_full:
            return out
        top_m = resolve_test_prior_top_m(args, city)
        if top_m > 0:
            prior = resolve_test_spatial_prior(args, top_m=top_m)
            if prior not in ("", "off", "none"):
                out.update({
                    "candidate_spatial_prior": prior,
                    "candidate_prior_top_m": top_m,
                    "candidate_prior_gravity_beta": float(
                        getattr(args, "candidate_prior_gravity_beta", 1.0) or 1.0),
                })
                return out
        # Explicit shared val/test prior (legacy, when not in f63 shortlist mode).
        prior = str(getattr(args, "candidate_spatial_prior", "off") or "off").lower()
        shared_m = int(getattr(args, "candidate_prior_top_m", 0) or 0)
        if final_neg > 0 or int(getattr(args, "final_candidates", 0) or 0) > 0:
            return out
        if prior in ("", "off", "none") or shared_m < 1:
            return out
        out.update({
            "candidate_spatial_prior": prior,
            "candidate_prior_top_m": shared_m,
            "candidate_prior_gravity_beta": float(
                getattr(args, "candidate_prior_gravity_beta", 1.0) or 1.0),
        })
        return out
    if final_neg > 0 or int(getattr(args, "final_candidates", 0) or 0) > 0:
        out["final_negatives"] = final_neg
        out["final_extra_random"] = int(getattr(args, "final_extra_random", -1))
        out["final_candidates_seed"] = int(
            getattr(args, "final_candidates_seed", getattr(args, "seed", 0)) or 0)
        return out
    prior = str(getattr(args, "candidate_spatial_prior", "off") or "off").lower()
    top_m = int(getattr(args, "candidate_prior_top_m", 0) or 0)
    if prior in ("", "off", "none") or top_m < 1:
        return out
    out.update({
        "candidate_spatial_prior": prior,
        "candidate_prior_top_m": top_m,
        "candidate_prior_gravity_beta": float(
            getattr(args, "candidate_prior_gravity_beta", 1.0) or 1.0),
    })
    return out


def val_pool_shrink_kwargs(args: Any, city: str | None = None) -> dict[str, Any]:
    """Val/eval shortlist kwargs (train+val share final_negatives when set)."""
    return eval_pool_shrink_kwargs(args, split="validation", city=city)


def test_pool_shrink_kwargs(
    args: Any,
    city: str | None = None,
    *,
    force_full: bool = False,
) -> dict[str, Any]:
    """Test-pool kwargs: optional mix top-M; ``force_full`` → full constructed."""
    return eval_pool_shrink_kwargs(
        args, split="test", city=city, force_full=force_full)


def checkpoint_on_climb_pb_enabled(args: Any) -> bool:
    raw = getattr(args, "checkpoint_on_climb_pb", "off")
    if isinstance(raw, bool):
        return raw
    return str(raw or "off").lower() in ("on", "true", "1", "yes")


def _read_json(path) -> dict[str, Any] | None:
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return None
    try:
        import json

        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def climb_lock_acc(city: str | None, *, root: str | None = None) -> float | None:
    """Climb Acc bar a run must beat to keep ``best.pt`` (state lock, else policy)."""
    if not city:
        return None
    from pathlib import Path

    base = Path(root) if root else Path.cwd()
    city = str(city).strip().lower()
    state = _read_json(base / "runs" / "climb_state" / f"{city}.json") or {}
    for key in ("lock_acc", "best_acc"):
        if state.get(key) is not None:
            try:
                return float(state[key])
            except (TypeError, ValueError):
                pass
    # Policy fallbacks: scripts/data/<city>_climb_policy.json or bdg/mel aliases.
    candidates = [
        base / "scripts" / "data" / f"{city}_climb_policy.json",
        base / "scripts" / "data" / "bdg_climb_policy.json" if city == "bandung" else None,
        base / "scripts" / "data" / "mel_climb_policy.json" if city == "melbourne" else None,
    ]
    for path in candidates:
        if path is None:
            continue
        pol = _read_json(path) or {}
        if pol.get("lock_acc") is not None:
            try:
                return float(pol["lock_acc"])
            except (TypeError, ValueError):
                pass
    return None


def is_climb_breakthrough(acc: float | None, lock: float | None) -> bool:
    """True when Acc@1 is a real lift over the climb lock (6 decimal places)."""
    if acc is None or lock is None:
        return False
    return round(float(acc), 6) > round(float(lock), 6)


def extract_known_acc1(metrics: Mapping[str, Any] | None) -> float | None:
    """Prefer ``known_only.Acc@1``, else ``all_targets.Acc@1``."""
    if not isinstance(metrics, Mapping):
        return None
    for key in ("known_only", "all_targets"):
        block = metrics.get(key)
        if isinstance(block, Mapping) and block.get("Acc@1") is not None:
            try:
                return float(block["Acc@1"])
            except (TypeError, ValueError):
                continue
    return None
