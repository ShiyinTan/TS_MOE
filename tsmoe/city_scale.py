"""City scale helpers: large-dataset set + optional fast model structure.

``large_city_fast`` (CLI / YAML) shrinks expensive WIN structure on Bandung-or-
larger Massive-STEPS cities only. Small cities ignore the flag.
"""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping

# Massive-STEPS raw data/ CSV ≥ Bandung (~57MB) on the GPU box (2026-11-26).
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
    """
    report: dict[str, Any] = {
        "requested": large_city_fast_enabled(args),
        "city": city,
        "applied": False,
        "changes": {},
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

    report["applied"] = True
    report["before"] = before
    report["after"] = {**before, **after}
    report["changes"] = after
    return report
