#!/usr/bin/env python3
"""Materialize densify-wave YAMLs for any city with a climb policy (GPU-local).

Uses ``scripts/data/<alias>_climb_policy.json`` base + ``<alias>_<wave>_overrides.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CITY_ALIAS = {
    "melbourne": "mel",
    "bandung": "bdg",
    "beijing": "bj",
    "palembang": "plg",
}

# Stable key order for readable YAMLs (extras appended sorted).
ORDERED_KEYS = (
    "cache",
    "session_summary",
    "eval_test_every_epoch",
    "batch_size",
    "eval_batch_size",
    "threads",
    "soft_target_beta",
    "soft_target_prior",
    "transition_attn",
    "geo_score_bias",
    "revisit_score_bias",
    "seed",
    "lambda_bpr",
    "jump_length_bias",
    "jump_dist",
    "pmi_score_bias",
    "pop_score_bias",
    "weight_decay",
    "epochs",
    "large_city_fast",
    "checkpoint_on_climb_pb",
    "history_max_length",
    "min_history_length",
    "final_negatives",
    "final_candidates",
    "final_extra_random",
    "final_candidates_seed",
    "early_stop_patience",
    "checkpoint_metric",
    "candidate_spatial_prior",
    "candidate_prior_top_m",
)


def alias_for(city: str) -> str:
    return CITY_ALIAS.get(city, city[:3])


def policy_path(city: str) -> Path:
    return ROOT / "scripts" / "data" / f"{alias_for(city)}_climb_policy.json"


def overrides_path(city: str, wave: str) -> Path:
    return ROOT / "scripts" / "data" / f"{alias_for(city)}_{wave}_overrides.json"


def load_base(city: str) -> dict:
    state_path = ROOT / "runs" / "climb_state" / f"{city}.json"
    if state_path.is_file():
        try:
            st = json.loads(state_path.read_text())
            if st.get("base"):
                return dict(st["base"])
        except Exception:
            pass
    path = policy_path(city)
    if path.is_file():
        base = json.loads(path.read_text()).get("base") or {}
        if base:
            return dict(base)
    raise FileNotFoundError(f"no climb base for city={city}")


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".")
    if isinstance(v, int):
        return str(v)
    return json.dumps(str(v))


def write_one(city: str, stem: str, overrides: dict, city_dir: Path, base: dict | None = None) -> Path:
    cfg = dict(base or load_base(city))
    cfg.update(overrides or {})
    cfg["out"] = f"runs/{city}/{stem}"
    cfg["recipe_id"] = stem
    cfg["description"] = f"{city} Acc climb · {stem} · local densify (gitignored until lock)"
    lines = [f"extends: {cfg.pop('extends')}"]
    for k in ORDERED_KEYS:
        if k in cfg:
            lines.append(f"{k}: {_fmt(cfg.pop(k))}")
    for k in sorted(cfg.keys()):
        if k in ("out", "recipe_id", "description"):
            continue
        lines.append(f"{k}: {_fmt(cfg[k])}")
    lines.append(f"out: {cfg['out']}")
    lines.append(f"recipe_id: {cfg['recipe_id']}")
    lines.append(f'description: "{cfg["description"]}"')
    path = city_dir / f"{stem}.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def ensure_wave_configs(wave: str, city: str) -> int:
    src = overrides_path(city, wave)
    if not src.is_file():
        raise FileNotFoundError(f"no overrides for city={city} wave={wave}: {src}")
    recipes = json.loads(src.read_text())
    city_dir = ROOT / "configs" / city
    city_dir.mkdir(parents=True, exist_ok=True)
    base = load_base(city)
    n = 0
    for stem, ov in recipes.items():
        path = city_dir / f"{stem}.yaml"
        if path.is_file():
            continue
        write_one(city, stem, ov, city_dir, base=base)
        n += 1
    return n


def rewrite_wave(wave: str, city: str) -> int:
    src = overrides_path(city, wave)
    recipes = json.loads(src.read_text())
    city_dir = ROOT / "configs" / city
    city_dir.mkdir(parents=True, exist_ok=True)
    base = load_base(city)
    for stem, ov in recipes.items():
        write_one(city, stem, ov, city_dir, base=base)
    return len(recipes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    ap.add_argument("--wave", default="w1")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.force:
        n = rewrite_wave(args.wave, args.city)
        print(f"{args.city} {args.wave}: rewrote {n}")
    else:
        n = ensure_wave_configs(args.wave, args.city)
        print(f"{args.city} {args.wave}: ensured {n}")


if __name__ == "__main__":
    main()
