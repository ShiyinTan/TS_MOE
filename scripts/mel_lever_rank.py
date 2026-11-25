#!/usr/bin/env python3
"""Rank Melbourne densify levers from historical known_only Acc@1.

Writes ``runs/climb_state/lever_rank.json`` used by ``mel_climb_autonomy`` to
drop dead switches and shrink value grids before synthesizing the next wave.

    python scripts/mel_lever_rank.py
    python scripts/mel_lever_rank.py --print
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "runs" / "climb_state"
OUT_PATH = STATE_DIR / "lever_rank.json"
POLICY_PATH = ROOT / "scripts" / "data" / "mel_climb_policy.json"

# Heuristic defaults when history is thin.
DEFAULT_BAN_KEYS = {
    "seed",  # repeatedly far below lock
    "transition_2hop_bias",
    "pop_score_bias",
    "hour_pop_score_bias",
    "gauss_geo_bias",
}
DEFAULT_KEEP_KEYS = {
    "soft_target_beta",
    "lambda_bpr",
    "jump_length_bias",
    "revisit_score_bias",
    "geo_score_bias",
    "epochs",
    "transition_attn",
    "weight_decay",
    "pmi_score_bias",  # keep but narrow — WIN uses 0.05
}


def _lock() -> float:
    if POLICY_PATH.is_file():
        try:
            return float(json.loads(POLICY_PATH.read_text()).get("lock_acc", 0.170692))
        except Exception:
            pass
    st = STATE_DIR / "melbourne.json"
    if st.is_file():
        try:
            return float(json.loads(st.read_text()).get("lock_acc", 0.170692))
        except Exception:
            pass
    return 0.170692


def _load_all_overrides() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted((ROOT / "scripts" / "data").glob("mel_w*_overrides.json")):
        try:
            out.update(json.loads(path.read_text()))
        except Exception:
            continue
    return out


def _scan_results(city: str = "melbourne") -> list[tuple[str, float]]:
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


def _family(stem: str) -> str:
    m = re.search(r"_pmi05_(.+)$", stem)
    if not m:
        return "other"
    suf = m.group(1)
    letters = "".join(c for c in suf if c.isalpha() or c == "_")
    return letters.split("_")[0] if letters else suf


def analyze(lock: float | None = None) -> dict[str, Any]:
    lock = _lock() if lock is None else float(lock)
    overrides = _load_all_overrides()
    rows = _scan_results()

    by_key: dict[str, list[tuple[Any, float, str]]] = defaultdict(list)
    by_fam: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for tag, acc in rows:
        by_fam[_family(tag)].append((acc, tag))
        ov = overrides.get(tag)
        if not ov:
            continue
        for k, v in ov.items():
            by_key[k].append((v, acc, tag))

    key_rank: list[dict[str, Any]] = []
    for key, items in by_key.items():
        n = len(items)
        accs = [a for _, a, _ in items]
        best = max(accs)
        mean = sum(accs) / n
        lifts = sum(1 for a in accs if round(a, 6) > round(lock, 6))
        ties = sum(1 for a in accs if round(a, 6) == round(lock, 6))
        under = sum(1 for a in accs if round(a, 6) < round(lock, 6))
        # score: emphasize best proximity to/above lock, penalize mean gap
        score = (best - lock) * 1000.0 + (mean - lock) * 200.0 + lifts * 5.0 + ties * 0.5 - under * 0.05
        # promising value band: values that hit >= lock-0.002
        good_vals = []
        for v, a, _ in items:
            if a >= lock - 0.002:
                if isinstance(v, float):
                    good_vals.append(round(v, 4))
                else:
                    good_vals.append(v)
        # unique preserve order
        seen = set()
        uniq_good = []
        for v in sorted(good_vals, key=lambda x: (isinstance(x, str), x)):
            if v in seen:
                continue
            seen.add(v)
            uniq_good.append(v)
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
                "good_values": uniq_good,
                "verdict": (
                    "keep_narrow"
                    if best >= lock - 0.001 or ties > 0
                    else ("drop" if best < lock - 0.004 and mean < lock - 0.006 else "shrink")
                ),
            }
        )
    key_rank.sort(key=lambda r: -r["score"])

    fam_rank: list[dict[str, Any]] = []
    for fam, items in by_fam.items():
        accs = [a for a, _ in items]
        n = len(items)
        best = max(accs)
        mean = sum(accs) / n
        lifts = sum(1 for a in accs if round(a, 6) > round(lock, 6))
        ties = sum(1 for a in accs if round(a, 6) == round(lock, 6))
        score = (best - lock) * 1000.0 + (mean - lock) * 200.0 + lifts * 5.0 + ties * 0.5
        fam_rank.append(
            {
                "family": fam,
                "n": n,
                "best": best,
                "mean": mean,
                "lifts": lifts,
                "ties": ties,
                "score": round(score, 4),
                "verdict": (
                    "keep"
                    if best >= lock - 0.001 or ties > 0
                    else ("drop" if best < lock - 0.004 else "shrink")
                ),
            }
        )
    fam_rank.sort(key=lambda r: -r["score"])

    ban_keys = {r["key"] for r in key_rank if r["verdict"] == "drop"}
    # force-ban clear losers from this MEL climb (even with thin history)
    ban_keys |= set(DEFAULT_BAN_KEYS)
    # never ban soft/bpr/jump/… keepers unless history says drop hard
    for k in list(ban_keys):
        if k in DEFAULT_KEEP_KEYS:
            row = next((r for r in key_rank if r["key"] == k), None)
            if row is None or row["verdict"] != "drop":
                ban_keys.discard(k)

    keep_keys = []
    for r in key_rank:
        if r["key"] in ban_keys:
            continue
        if r["verdict"] in ("keep_narrow", "shrink") or r["key"] in DEFAULT_KEEP_KEYS:
            keep_keys.append(r["key"])
    for k in DEFAULT_KEEP_KEYS:
        if k not in ban_keys and k not in keep_keys:
            keep_keys.append(k)

    # Suggested narrowed grids around WIN + good_values
    base = {}
    if POLICY_PATH.is_file():
        base = dict(json.loads(POLICY_PATH.read_text()).get("base") or {})

    suggested: dict[str, list[Any]] = {}
    win_defaults = {
        "soft_target_beta": 0.078,
        "lambda_bpr": 0.205,
        "jump_length_bias": 0.10,
        "pmi_score_bias": 0.05,
        "revisit_score_bias": 0.10,
        "geo_score_bias": 0.15,
        "weight_decay": 0.01,
        "epochs": 2,
    }
    narrow_spans = {
        "soft_target_beta": (-0.006, 0.012, 0.002),
        "lambda_bpr": (-0.008, 0.008, 0.002),
        "jump_length_bias": (-0.02, 0.03, 0.005),
        "pmi_score_bias": (-0.015, 0.02, 0.005),
        "revisit_score_bias": (-0.02, 0.05, 0.01),
        "geo_score_bias": (-0.05, 0.03, 0.01),
        "weight_decay": (0.0, 0.0, 0.0),  # discrete handled below
    }

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
        center = float(base.get(key, win_defaults.get(key, 0.1)))
        if key in narrow_spans:
            lo, hi, step = narrow_spans[key]
            grid = _frange(center, lo, hi, step)
        else:
            grid = [center]
        # merge good historical values near center
        for g in goods:
            if isinstance(g, (int, float)) and abs(float(g) - center) <= abs(narrow_spans.get(key, (0, 0.05, 0))[1]) + 0.02:
                grid.append(round(float(g), 4))
        suggested[key] = sorted(set(grid))

    # Priority combos from near-lock levers only
    combos = [
        {"tag": "soft080_bpr205", "overrides": {"soft_target_beta": 0.08, "lambda_bpr": 0.205}},
        {"tag": "soft082_jump10", "overrides": {"soft_target_beta": 0.082, "jump_length_bias": 0.10}},
        {"tag": "soft076_jump11", "overrides": {"soft_target_beta": 0.076, "jump_length_bias": 0.11}},
        {"tag": "bpr202_jump10", "overrides": {"lambda_bpr": 0.202, "jump_length_bias": 0.10}},
        {"tag": "bpr208_soft080", "overrides": {"lambda_bpr": 0.208, "soft_target_beta": 0.08}},
        {"tag": "rev10_soft080", "overrides": {"revisit_score_bias": 0.10, "soft_target_beta": 0.08}},
        {"tag": "geo15_soft080", "overrides": {"geo_score_bias": 0.15, "soft_target_beta": 0.08}},
        {"tag": "ep2_soft080", "overrides": {"epochs": 2, "soft_target_beta": 0.08}},
        {"tag": "ep3_bpr202", "overrides": {"epochs": 3, "lambda_bpr": 0.202}},
        {"tag": "trgate_soft080", "overrides": {"transition_attn": "gate", "soft_target_beta": 0.08}},
        {"tag": "trgate_rev10", "overrides": {"transition_attn": "gate", "revisit_score_bias": 0.10}},
        {"tag": "wd010_soft080", "overrides": {"weight_decay": 0.01, "soft_target_beta": 0.08}},
        {"tag": "pmi04_soft080", "overrides": {"pmi_score_bias": 0.04, "soft_target_beta": 0.08}},
        {"tag": "pmi06_jump10", "overrides": {"pmi_score_bias": 0.06, "jump_length_bias": 0.10}},
        {"tag": "soft085_bpr202_jump11", "overrides": {"soft_target_beta": 0.085, "lambda_bpr": 0.202, "jump_length_bias": 0.11}},
    ]

    report = {
        "lock_acc": lock,
        "n_results": len(rows),
        "ban_keys": sorted(ban_keys),
        "keep_keys": keep_keys,
        "suggested_values": suggested,
        "suggested_combos": combos,
        "key_rank": key_rank,
        "family_rank": fam_rank,
        "notes": (
            "drop seed/hop/pop/hpop/gauss — historically under lock; "
            "focus soft/bpr/jump/rev/geo/ep/trgate/wd/pmi-narrow + combos"
        ),
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--lock", type=float, default=None)
    args = ap.parse_args()
    report = analyze(args.lock)
    if args.print:
        print(
            json.dumps(
                {
                    "n_results": report["n_results"],
                    "ban_keys": report["ban_keys"],
                    "keep_keys": report["keep_keys"],
                    "suggested_values": report["suggested_values"],
                    "top_keys": report["key_rank"][:8],
                    "drop_families": [f for f in report["family_rank"] if f["verdict"] == "drop"][:10],
                },
                indent=2,
            )
        )
    else:
        print(f"wrote {OUT_PATH} n={report['n_results']} ban={report['ban_keys']}")


if __name__ == "__main__":
    main()
