#!/usr/bin/env python3
"""Reset climb state + build f63 densify w1 for Bandung/MEL/BJ/PLG (GPU-local).

Backs up existing ``runs/climb_state/<city>.json`` to ``*_pre_f63.json``,
reloads base from ``*_climb_policy.json`` (already f63/L128), synthesizes
``w1`` overrides from densify_families/combos, and materializes YAMLs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from city_climb_autonomy import (  # noqa: E402
    build_next_wave_recipes,
    load_policy,
    save_overrides,
    save_state,
    state_path,
)
from gen_city_wave_local import ensure_wave_configs, rewrite_wave  # noqa: E402

DEFAULT_CITIES = ("bandung", "melbourne", "beijing", "palembang")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def bootstrap_city(city: str, *, force_yaml: bool = True) -> dict:
    policy = load_policy(city)
    sp = state_path(city)
    sp.parent.mkdir(parents=True, exist_ok=True)
    if sp.is_file():
        bak = sp.with_name(f"{city}_pre_f63.json")
        if not bak.exists():
            shutil.copy2(sp, bak)
        else:
            # keep first backup; also write timestamped snapshot
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            shutil.copy2(sp, sp.with_name(f"{city}_pre_f63_{ts}.json"))

    wave = str(policy.get("active_wave") or "w1")
    state = {
        "city": city,
        "lock_acc": float(policy.get("lock_acc") or 0.0),
        "target_acc": float(policy["target_acc"]),
        "active_wave": wave,
        "base": dict(policy.get("base") or {}),
        "best_acc": float(policy.get("lock_acc") or 0.0),
        "best_tag": None,
        "waves_done": [],
        "tried_fingerprints": [],
        "target_met": False,
        "auto_advance": bool(policy.get("auto_advance", True)),
        "protocol": "f63_L128",
        "updated_at": _now(),
    }
    save_state(city, state)

    # Fresh densify from policy families (no lever_rank pruning on bootstrap).
    recipes = build_next_wave_recipes(
        city, policy, state, wave, use_lever_rank=False
    )
    if not recipes:
        raise RuntimeError(f"{city}: empty f63 wave {wave}")
    save_overrides(city, wave, recipes)
    # Refresh tried fingerprints from the new wave so advance stays consistent.
    state["tried_fingerprints"] = [
        json.dumps(ov, sort_keys=True, separators=(",", ":")) for ov in recipes.values()
    ]
    save_state(city, state)

    if force_yaml:
        n = rewrite_wave(wave, city)
    else:
        n = ensure_wave_configs(wave, city)
    return {
        "city": city,
        "wave": wave,
        "n_recipes": len(recipes),
        "n_yamls": n,
        "cache": state["base"].get("cache"),
        "lock_acc": state["lock_acc"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cities",
        nargs="*",
        default=list(DEFAULT_CITIES),
        help="Cities to bootstrap (default: all four)",
    )
    args = ap.parse_args()
    out = []
    for city in args.cities:
        info = bootstrap_city(city)
        out.append(info)
        print(json.dumps(info), flush=True)
    print(json.dumps({"event": "bootstrap_f63_done", "cities": out}, indent=2))


if __name__ == "__main__":
    main()
