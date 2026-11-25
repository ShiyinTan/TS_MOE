#!/usr/bin/env python3
"""Load / update ``configs/climb_retention.yaml`` (shared by prune tools)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyYAML required") from e

ROOT = Path(__file__).resolve().parents[1]
RETENTION_PATH = ROOT / "configs" / "climb_retention.yaml"


def load_retention(path: Path = RETENTION_PATH) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if "cities" not in data:
        raise ValueError(f"missing cities in {path}")
    return data


def save_retention(data: dict[str, Any], path: Path = RETENTION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
    )


def sthgcn_map(data: dict[str, Any] | None = None) -> dict[str, float]:
    data = data or load_retention()
    return {k: float(v) for k, v in (data.get("sthgcn") or {}).items()}


def city_keep_configs(city: str, data: dict[str, Any] | None = None) -> set[str]:
    data = data or load_retention()
    c = (data.get("cities") or {}).get(city) or {}
    return set(c.get("keep_configs") or [])


def city_keep_results(city: str, data: dict[str, Any] | None = None) -> set[str]:
    data = data or load_retention()
    c = (data.get("cities") or {}).get(city) or {}
    return set(c.get("keep_results") or [])


def city_keep_prefixes(city: str, data: dict[str, Any] | None = None) -> list[str]:
    data = data or load_retention()
    c = (data.get("cities") or {}).get(city) or {}
    return list(c.get("keep_results_prefixes") or [])


def city_keep_wave(city: str, data: dict[str, Any] | None = None) -> str | None:
    data = data or load_retention()
    c = (data.get("cities") or {}).get(city) or {}
    w = c.get("keep_wave")
    return str(w) if w else None


def city_aggressive(city: str, data: dict[str, Any] | None = None) -> bool:
    data = data or load_retention()
    c = (data.get("cities") or {}).get(city) or {}
    return bool(c.get("aggressive_configs", c.get("status") == "STOPPED"))


def register_lock(
    city: str,
    *,
    lock_config: str,
    results_tag: str | None = None,
    acc: float | None = None,
    keep_wave: str | None = None,
    status: str | None = None,
    foundation_configs: list[str] | None = None,
    path: Path = RETENTION_PATH,
) -> dict[str, Any]:
    """Update retention after a climb Acc lock / new active wave."""
    data = load_retention(path)
    cities = data.setdefault("cities", {})
    c = cities.setdefault(city, {})
    cfg_name = Path(lock_config).name
    keeps = set(c.get("keep_configs") or [])
    keeps.update({"base.yaml", "peak.yaml", "win.yaml", cfg_name})
    if foundation_configs:
        keeps.update(foundation_configs)
    c["keep_configs"] = sorted(keeps)
    c["lock_config"] = (
        lock_config if lock_config.startswith("configs/") else f"configs/{city}/{cfg_name}"
    )
    if acc is not None:
        c["lock_acc"] = float(acc)
    if status:
        c["status"] = status
        c["aggressive_configs"] = status == "STOPPED"
    if keep_wave is not None:
        c["keep_wave"] = keep_wave or None
        # refresh results prefix for active wave
        if keep_wave:
            # derive common prefix from melbourne-style climb_*_w17_
            prefixes = set(c.get("keep_results_prefixes") or [])
            # generic: keep any existing prefixes that match wave id
            wid = keep_wave[1:] if keep_wave.startswith("w") else keep_wave
            prefixes = {p for p in prefixes if f"_w{wid}_" in p or p.endswith(f"_w{wid}_")}
            # add a broad prefix if city uses climb_*_wN_
            prefixes.add(f"_w{wid}_")  # matcher treats as substring via startswith on full tag — see sync
            c["keep_results_prefixes"] = sorted(prefixes)
            c["status"] = c.get("status") or "ACTIVE"
            c["aggressive_configs"] = False
    if results_tag:
        tags = set(c.get("keep_results") or [])
        tags.add(results_tag)
        # also keep win recipe_id stem without path
        tags.add(Path(lock_config).stem)
        c["keep_results"] = sorted(tags)
    cities[city] = c
    data["cities"] = cities
    save_retention(data, path)
    return data
