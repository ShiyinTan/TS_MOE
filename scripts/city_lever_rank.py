#!/usr/bin/env python3
"""Rank densify levers for any climb city (MEL priors + local Acc history).

    python scripts/city_lever_rank.py --city bandung
    python scripts/city_lever_rank.py --city melbourne --print

Writes ``runs/climb_state/lever_rank_<city>.json`` (and ``lever_rank.json`` for melbourne).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "runs" / "climb_state"
PRIORS_PATH = ROOT / "scripts" / "data" / "climb_lever_priors.json"

CITY_ALIAS = {
    "melbourne": "mel",
    "bandung": "bdg",
}


def _priors() -> dict[str, Any]:
    if PRIORS_PATH.is_file():
        return json.loads(PRIORS_PATH.read_text())
    return {}


def _policy_path(city: str) -> Path:
    alias = CITY_ALIAS.get(city, city[:3])
    return ROOT / "scripts" / "data" / f"{alias}_climb_policy.json"


def _overrides_glob(city: str) -> str:
    alias = CITY_ALIAS.get(city, city[:3])
    return f"{alias}_w*_overrides.json"


def _load_policy(city: str) -> dict[str, Any]:
    p = _policy_path(city)
    if p.is_file():
        return json.loads(p.read_text())
    return {}


def _load_overrides(city: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted((ROOT / "scripts" / "data").glob(_overrides_glob(city))):
        try:
            out.update(json.loads(path.read_text()))
        except Exception:
            continue
    return out


def _scan_results(city: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    root = ROOT / "results" / city
    if not root.is_dir():
        return rows
    for metrics in root.glob("*/eval_test/metrics.json"):
        tag = metrics.parent.parent.name
        try:
            acc = (json.loads(metrics.read_text()).get("known_only") or {}).get("Acc@1")
        except Exception:
            continue
        if acc is None:
            continue
        rows.append((tag, float(acc)))
    return rows


def analyze(city: str, lock: float | None = None) -> dict[str, Any]:
    priors = _priors()
    policy = _load_policy(city)
    lock = float(lock if lock is not None else policy.get("lock_acc") or 0.0)
    overrides = _load_overrides(city)
    rows = _scan_results(city)
    ban_default = set(priors.get("ban_keys") or []) | set(policy.get("ban_keys") or [])
    # Early cities with thin history: also ban optional early losers (gauss)
    if len(rows) < 40:
        ban_default |= set(priors.get("ban_keys_optional_early") or [])
    keep_default = list(priors.get("keep_keys") or [])

    by_key: dict[str, list[tuple[Any, float, str]]] = defaultdict(list)
    for tag, acc in rows:
        ov = overrides.get(tag)
        if not ov:
            continue
        for k, v in ov.items():
            by_key[k].append((v, acc, tag))

    key_rank: list[dict[str, Any]] = []
    for key, items in by_key.items():
        accs = [a for _, a, _ in items]
        n = len(items)
        best = max(accs)
        mean = sum(accs) / n
        lifts = sum(1 for a in accs if round(a, 6) > round(lock, 6))
        ties = sum(1 for a in accs if round(a, 6) == round(lock, 6))
        under = sum(1 for a in accs if round(a, 6) < round(lock, 6))
        score = (best - lock) * 1000.0 + (mean - lock) * 200.0 + lifts * 5.0 + ties * 0.5 - under * 0.05
        good_vals = []
        for v, a, _ in items:
            if a >= lock - 0.002:
                good_vals.append(round(v, 4) if isinstance(v, float) else v)
        uniq: list[Any] = []
        seen: set[Any] = set()
        for v in good_vals:
            if v in seen:
                continue
            seen.add(v)
            uniq.append(v)
        key_rank.append(
            {
                "key": key,
                "n": n,
                "best": best,
                "mean": mean,
                "lifts": lifts,
                "ties": ties,
                "under": under,
                "score": round(score, 4),
                "good_values": uniq,
                "verdict": (
                    "keep_narrow"
                    if best >= lock - 0.001 or ties > 0
                    else ("drop" if best < lock - 0.004 and mean < lock - 0.006 else "shrink")
                ),
            }
        )
    key_rank.sort(key=lambda r: -r["score"])

    ban_keys = {r["key"] for r in key_rank if r["verdict"] == "drop"} | ban_default
    # Never ban MEL/policy keep levers — flat Acc history often marks them "drop"
    # and then advance_failed_empty (no wN+1 recipes). Narrow via suggested_values.
    for k in list(ban_keys):
        if k in keep_default:
            ban_keys.discard(k)

    keep_keys: list[str] = []
    for r in key_rank:
        if r["key"] in ban_keys:
            continue
        if r["verdict"] in ("keep_narrow", "shrink") or r["key"] in keep_default:
            keep_keys.append(r["key"])
    for k in keep_default:
        if k not in ban_keys and k not in keep_keys:
            keep_keys.append(k)

    base = dict(policy.get("base") or {})
    centers = dict(priors.get("default_centers") or {})
    centers.update({k: base[k] for k in base if isinstance(base.get(k), (int, float))})
    spans = {k: tuple(v) for k, v in (priors.get("narrow_spans") or {}).items()}

    def _frange(center: float, lo_d: float, hi_d: float, step: float) -> list[float]:
        if step <= 0:
            return [round(center, 4)]
        vals = []
        x = center + lo_d
        end = center + hi_d + 1e-12
        while x <= end:
            vals.append(round(x, 4))
            x += step
        return vals

    suggested: dict[str, list[Any]] = {}
    for key in keep_keys:
        row = next((r for r in key_rank if r["key"] == key), None)
        goods = list(row["good_values"]) if row else []
        if key == "epochs":
            suggested[key] = [2, 3]
            continue
        if key == "transition_attn":
            suggested[key] = ["gate"]
            continue
        if key == "weight_decay":
            suggested[key] = [0.005, 0.008, 0.01]
            continue
        center = float(centers.get(key, 0.1))
        if key in spans:
            lo, hi, step = spans[key]
            grid = _frange(center, lo, hi, step)
        else:
            grid = [center]
        for g in goods:
            if isinstance(g, (int, float)):
                grid.append(round(float(g), 4))
        suggested[key] = sorted(set(grid))

    # Combos: MEL soft×jump / soft×bpr / jump×pmi first; then policy combos
    softs = suggested.get("soft_target_beta") or [centers["soft_target_beta"]]
    jumps = suggested.get("jump_length_bias") or [centers["jump_length_bias"]]
    bprs = suggested.get("lambda_bpr") or [centers["lambda_bpr"]]
    pmis = suggested.get("pmi_score_bias") or [centers["pmi_score_bias"]]
    c_soft = softs[len(softs) // 2]
    c_jump = jumps[len(jumps) // 2]
    c_bpr = bprs[len(bprs) // 2]
    c_pmi = pmis[len(pmis) // 2]
    combos = [
        {"tag": "soft_jump", "overrides": {"soft_target_beta": c_soft, "jump_length_bias": c_jump}},
        {"tag": "soft_bpr", "overrides": {"soft_target_beta": c_soft, "lambda_bpr": c_bpr}},
        {"tag": "jump_pmi", "overrides": {"jump_length_bias": c_jump, "pmi_score_bias": c_pmi}},
        {
            "tag": "soft_jump_pmi",
            "overrides": {
                "soft_target_beta": c_soft,
                "jump_length_bias": c_jump,
                "pmi_score_bias": c_pmi,
            },
        },
        {"tag": "ep2_soft", "overrides": {"epochs": 2, "soft_target_beta": c_soft}},
        {
            "tag": "soft_hi_jump_lo",
            "overrides": {
                "soft_target_beta": softs[-1] if softs else c_soft,
                "jump_length_bias": jumps[0] if jumps else c_jump,
            },
        },
    ]
    for combo in policy.get("combo_recipes") or []:
        combos.append(combo)

    out = {
        "city": city,
        "lock_acc": lock,
        "n_results": len(rows),
        "n_override_tags": len(overrides),
        "ban_keys": sorted(ban_keys),
        "keep_keys": keep_keys,
        "key_rank": key_rank,
        "suggested_values": suggested,
        "suggested_combos": combos,
        "priors": "climb_lever_priors.json",
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"lever_rank_{city}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    if city == "melbourne":
        (STATE_DIR / "lever_rank.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    ap.add_argument("--lock", type=float, default=None)
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    report = analyze(args.city, lock=args.lock)
    if args.print:
        print(json.dumps({k: report[k] for k in ("city", "lock_acc", "ban_keys", "keep_keys", "suggested_values")}, indent=2))
    else:
        print(f"wrote runs/climb_state/lever_rank_{args.city}.json n={report['n_results']}")


if __name__ == "__main__":
    main()
