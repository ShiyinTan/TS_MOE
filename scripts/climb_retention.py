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
        # refresh results prefix for active wave (results prune uses startswith)
        if keep_wave:
            wave = str(keep_wave)
            if not wave.startswith("w"):
                wave = f"w{wave}"
            wid = wave[1:]
            needle = f"_w{wid}_"
            prefixes = {
                p
                for p in (c.get("keep_results_prefixes") or [])
                if isinstance(p, str) and needle in p and not p.startswith("_")
            }
            # Prefer full dir prefix through …_wN_ (e.g. climb_peak_soft010_f63_w1_).
            # Bare "_w1_" is NOT valid: startswith("_w1_") never matches climb_* tags.
            if results_tag and needle in str(results_tag):
                tag = str(results_tag)
                prefixes.add(tag[: tag.find(needle) + len(needle)])
            # Policy stem_prefix → <stem>_wN_
            try:
                alias = {"melbourne": "mel", "bandung": "bdg", "beijing": "bj", "palembang": "plg"}.get(
                    city, city[:3]
                )
                pol_path = Path(__file__).resolve().parents[1] / "scripts" / "data" / f"{alias}_climb_policy.json"
                if pol_path.is_file():
                    import json

                    pol = json.loads(pol_path.read_text())
                    stem = str(pol.get("stem_prefix") or "").strip()
                    if stem:
                        prefixes.add(f"{stem}_{wave}_")
            except Exception:
                pass
            c["keep_results_prefixes"] = sorted(prefixes)
            c["status"] = c.get("status") or "ACTIVE"
            c["aggressive_configs"] = False
    if results_tag:
        tags = set(c.get("keep_results") or [])
        tags.add(results_tag)
        # also keep win recipe_id stem without path
        tags.add(Path(lock_config).stem)
        c["keep_results"] = sorted(tags)
        # If ACTIVE wave is set, ensure prefix covers this tag's wave stem.
        wave = c.get("keep_wave")
        if wave and f"_{wave}_" in str(results_tag):
            prefs = set(c.get("keep_results_prefixes") or [])
            tag = str(results_tag)
            needle = f"_{wave}_"
            prefs.add(tag[: tag.find(needle) + len(needle)])
            prefs = {p for p in prefs if not str(p).startswith("_")}
            c["keep_results_prefixes"] = sorted(prefs)
    cities[city] = c
    data["cities"] = cities
    save_retention(data, path)
    return data
