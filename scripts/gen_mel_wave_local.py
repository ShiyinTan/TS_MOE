#!/usr/bin/env python3
"""Materialize Melbourne densify-wave YAMLs **locally** (GPU box) — not for git.

Wave configs are gitignored. After ``git pull``, regenerate::

    python scripts/gen_mel_wave_local.py --wave w17

Packer / ``mel_climb_autonomy`` call ``ensure_wave_configs()`` so missing files
are written before launch. Base knobs come from ``scripts/data/mel_climb_policy.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "scripts" / "data" / "mel_climb_policy.json"

# Fallback if policy missing (f63 shortlist on L128).
DEFAULT_BASE = {
    "extends": "./climb_bpr020_soft010_f63.yaml",
    "cache": "runs/melbourne/cache_tgt_all_eval_last_L128",
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
    "final_negatives": 63,
    "final_extra_random": -1,
    "history_max_length": 128,
    "min_history_length": 1,
    "epochs": 50,
    "early_stop_patience": 5,
    "checkpoint_metric": "acc1",
    "checkpoint_on_climb_pb": "on",
}


def load_base() -> dict:
    # Prefer GPU-local state base (updated on Acc lift); else committed policy.
    state_path = ROOT / "runs" / "climb_state" / "melbourne.json"
    if state_path.is_file():
        try:
            st = json.loads(state_path.read_text())
            if st.get("base"):
                return dict(st["base"])
        except Exception:
            pass
    if POLICY_PATH.is_file():
        try:
            base = json.loads(POLICY_PATH.read_text()).get("base") or {}
            if base:
                return dict(base)
        except Exception:
            pass
    return dict(DEFAULT_BASE)


def overrides_path(wave: str) -> Path:
    return ROOT / "scripts" / "data" / f"mel_{wave}_overrides.json"


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s
    if isinstance(v, int):
        return str(v)
    return json.dumps(str(v))


def write_one(stem: str, overrides: dict, city_dir: Path, base: dict | None = None) -> Path:
    cfg = dict(base or load_base())
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
        "history_max_length",
        "min_history_length",
        "final_negatives",
        "final_extra_random",
        "early_stop_patience",
        "checkpoint_metric",
        "epochs",
    ):
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


def ensure_wave_configs(wave: str = "w17", city: str = "melbourne") -> int:
    src = overrides_path(wave)
    if not src.is_file():
        raise FileNotFoundError(f"no overrides for wave={wave}: {src}")
    recipes = json.loads(src.read_text())
    city_dir = ROOT / "configs" / city
    city_dir.mkdir(parents=True, exist_ok=True)
    base = load_base()
    n = 0
    for stem, ov in recipes.items():
        path = city_dir / f"{stem}.yaml"
        if path.is_file():
            continue
        write_one(stem, ov, city_dir, base=base)
        n += 1
    return n


def rewrite_all(wave: str = "w17", city: str = "melbourne") -> int:
    src = overrides_path(wave)
    recipes = json.loads(src.read_text())
    city_dir = ROOT / "configs" / city
    city_dir.mkdir(parents=True, exist_ok=True)
    base = load_base()
    for stem, ov in recipes.items():
        write_one(stem, ov, city_dir, base=base)
    return len(recipes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wave", default=None, help="default: policy.active_wave")
    ap.add_argument("--city", default="melbourne")
    ap.add_argument("--force", action="store_true", help="Rewrite even if files exist")
    args = ap.parse_args()
    wave = args.wave
    if wave is None:
        if POLICY_PATH.is_file():
            wave = json.loads(POLICY_PATH.read_text()).get("active_wave", "w17")
        else:
            wave = "w17"
    n = rewrite_all(wave, args.city) if args.force else ensure_wave_configs(wave, args.city)
    print(f"{args.city} {wave}: wrote/ensured {n} local yamls under configs/{args.city}/")


if __name__ == "__main__":
    main()
