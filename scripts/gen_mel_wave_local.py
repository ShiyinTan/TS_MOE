#!/usr/bin/env python3
"""Materialize Melbourne densify-wave YAMLs **locally** (GPU box) — not for git.

Wave configs are gitignored. After ``git pull``, regenerate::

    python scripts/gen_mel_wave_local.py --wave w17

Packer calls ``ensure_wave_configs()`` so missing files are written before launch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Shared WIN baseline fields for mel w17 densify.
W17_BASE = {
    "extends": "./climb_bpr020_soft010.yaml",
    "cache": "runs/melbourne/cache_tgt_all_eval_last",
    "session_summary": "on",
    "eval_test_every_epoch": False,
    "batch_size": 48,
    "eval_batch_size": 32,
    "threads": 4,
    "soft_target_beta": 0.078,
    "seed": 46,
    "lambda_bpr": 0.205,
    "jump_length_bias": 0.10,
    "jump_dist": "lognormal",
    "pmi_score_bias": 0.05,
}

WAVE_SOURCES = {
    "w17": ROOT / "scripts" / "data" / "mel_w17_overrides.json",
}


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        # keep readable floats
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s
    if isinstance(v, int):
        return str(v)
    return json.dumps(str(v))


def write_one(stem: str, overrides: dict, city_dir: Path) -> Path:
    cfg = dict(W17_BASE)
    cfg.update(overrides or {})
    cfg["out"] = f"runs/melbourne/{stem}"
    cfg["recipe_id"] = stem
    cfg["description"] = (
        f"melbourne clear-margin climb · {stem} · local densify wave "
        "(not committed to git until Acc lock)"
    )
    lines = [f"extends: {cfg.pop('extends')}"]
    for k in (
        "cache",
        "session_summary",
        "eval_test_every_epoch",
        "batch_size",
        "eval_batch_size",
        "threads",
        "soft_target_beta",
        "seed",
        "lambda_bpr",
        "jump_length_bias",
        "jump_dist",
        "pmi_score_bias",
    ):
        if k in cfg:
            lines.append(f"{k}: {_fmt(cfg.pop(k))}")
    # remaining overrides (gauss, geo, epochs, …)
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


def ensure_wave_configs(wave: str = "w17", city: str = "melbourne") -> int:
    src = WAVE_SOURCES.get(wave)
    if src is None or not src.is_file():
        raise FileNotFoundError(f"no overrides for wave={wave}: {src}")
    recipes = json.loads(src.read_text())
    city_dir = ROOT / "configs" / city
    city_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for stem, ov in recipes.items():
        path = city_dir / f"{stem}.yaml"
        if path.is_file():
            continue
        write_one(stem, ov, city_dir)
        n += 1
    return n


def rewrite_all(wave: str = "w17", city: str = "melbourne") -> int:
    src = WAVE_SOURCES[wave]
    recipes = json.loads(src.read_text())
    city_dir = ROOT / "configs" / city
    city_dir.mkdir(parents=True, exist_ok=True)
    for stem, ov in recipes.items():
        write_one(stem, ov, city_dir)
    return len(recipes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wave", default="w17")
    ap.add_argument("--city", default="melbourne")
    ap.add_argument("--force", action="store_true", help="Rewrite even if files exist")
    args = ap.parse_args()
    n = rewrite_all(args.wave, args.city) if args.force else ensure_wave_configs(args.wave, args.city)
    print(f"{args.city} {args.wave}: wrote/ensured {n} local yamls under configs/{args.city}/")


if __name__ == "__main__":
    main()
