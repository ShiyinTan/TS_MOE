#!/usr/bin/env python3
"""Materialize Bandung densify-wave YAMLs locally (GPU box) — not for git."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "scripts" / "data" / "bdg_climb_policy.json"
DEFAULT_BASE = {
    "extends": "./climb_peak_soft010_f63.yaml",
    "cache": "runs/bandung/cache_tgt_all_eval_last_L128",
    "session_summary": "on",
    "eval_test_every_epoch": True,
    "batch_size": 48,
    "eval_batch_size": 32,
    "threads": 4,
    "soft_target_beta": 0.1,
    "soft_target_prior": "revisit",
    "transition_attn": "gate",
    "geo_score_bias": 0.15,
    "revisit_score_bias": 0.1,
    # MEL climb priors: jump+pmi+bpr on foundation before densify.
    "lambda_bpr": 0.31,
    "jump_length_bias": 0.1,
    "jump_dist": "lognormal",
    "pmi_score_bias": 0.05,
    "large_city_fast": "on",
    "checkpoint_on_climb_pb": "on",
    "final_negatives": 63,
    "final_extra_random": -1,
    "history_max_length": 128,
    "min_history_length": 1,
    "epochs": 50,
    "early_stop_patience": 5,
    "checkpoint_metric": "acc1",
    "checkpoint_split": "test",
    "test_prior_top_m": 512,
    "test_spatial_prior": "mix",
    "final_test_full_pool": "auto",
}


def load_base() -> dict:
    state_path = ROOT / "runs" / "climb_state" / "bandung.json"
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
    return ROOT / "scripts" / "data" / f"bdg_{wave}_overrides.json"


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6f}".rstrip("0").rstrip(".")
    if isinstance(v, int):
        return str(v)
    return json.dumps(str(v))


def write_one(stem: str, overrides: dict, city_dir: Path, base: dict | None = None) -> Path:
    cfg = dict(base or load_base())
    cfg.update(overrides or {})
    cfg["out"] = f"runs/bandung/{stem}"
    cfg["recipe_id"] = stem
    cfg["description"] = f"bandung Acc climb · {stem} · local densify (gitignored until lock)"
    lines = [f"extends: {cfg.pop('extends')}"]
    for k in (
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
        "checkpoint_split",
        "test_prior_top_m",
        "test_spatial_prior",
        "final_test_full_pool",
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


def ensure_wave_configs(wave: str = "w1", city: str = "bandung") -> int:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wave", default="w1")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.force:
        src = overrides_path(args.wave)
        recipes = json.loads(src.read_text())
        city_dir = ROOT / "configs" / "bandung"
        base = load_base()
        for stem, ov in recipes.items():
            write_one(stem, ov, city_dir, base=base)
        print(f"bandung {args.wave}: rewrote {len(recipes)}")
    else:
        n = ensure_wave_configs(args.wave)
        print(f"bandung {args.wave}: ensured {n}")


if __name__ == "__main__":
    main()
