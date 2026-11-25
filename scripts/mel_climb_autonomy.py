#!/usr/bin/env python3
"""GPU-local Melbourne climb autonomy — no cloud needed to keep training.

Responsibilities
----------------
* Track active wave / lock / best Acc under ``runs/climb_state/``.
* When the active wave finishes, write a harvest snapshot locally, then
  (if ``auto_advance``) synthesize the next densify wave overrides + YAMLs
  and keep the packer queue full.
* On Acc lift (> lock), update local lock/base and draft a WIN yaml copy
  under ``runs/climb_state/`` (git promote still happens at harvest).

Cloud agents should only ``mel_acc_oneline`` / read ``harvest_latest.json``
on a sparse timer — never required to launch the next wave.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "scripts" / "data" / "mel_climb_policy.json"
STATE_DIR = ROOT / "runs" / "climb_state"
STATE_PATH = STATE_DIR / "melbourne.json"
HARVEST_PATH = STATE_DIR / "harvest_latest.json"
HARVEST_LOG = STATE_DIR / "harvest_log.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_policy() -> dict[str, Any]:
    """Committed densify template. Mutable runtime fields live in state."""
    return json.loads(POLICY_PATH.read_text())


def effective_policy() -> dict[str, Any]:
    """Policy overlayed with GPU-local state (active_wave / lock / base)."""
    policy = load_policy()
    state = load_state()
    out = dict(policy)
    if state.get("active_wave"):
        out["active_wave"] = state["active_wave"]
    if state.get("lock_acc") is not None:
        out["lock_acc"] = float(state["lock_acc"])
    if state.get("base"):
        out["base"] = state["base"]
    if "auto_advance" in state:
        out["auto_advance"] = bool(state["auto_advance"])
    return out


def load_state() -> dict[str, Any]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    policy = json.loads(POLICY_PATH.read_text())
    st = {
        "city": "melbourne",
        "lock_acc": float(policy["lock_acc"]),
        "target_acc": float(policy["target_acc"]),
        "active_wave": policy["active_wave"],
        "base": dict(policy.get("base") or {}),
        "best_acc": float(policy["lock_acc"]),
        "best_tag": None,
        "waves_done": [],
        "tried_fingerprints": [],
        "target_met": False,
        "auto_advance": bool(policy.get("auto_advance", True)),
        "updated_at": _now(),
    }
    save_state(st)
    return st


def save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def overrides_path(wave: str) -> Path:
    return ROOT / "scripts" / "data" / f"mel_{wave}_overrides.json"


def wave_num(wave: str) -> int:
    m = re.fullmatch(r"w(\d+)", wave)
    if not m:
        raise ValueError(f"bad wave id: {wave}")
    return int(m.group(1))


def next_wave_id(wave: str) -> str:
    return f"w{wave_num(wave) + 1}"


def fingerprint(overrides: dict[str, Any]) -> str:
    return json.dumps(overrides, sort_keys=True, separators=(",", ":"))


def stem_for(policy: dict[str, Any], wave: str, tag: str) -> str:
    return f"{policy['stem_prefix']}_{wave}_{policy['stem_mid']}_{tag}"


def load_overrides(wave: str) -> dict[str, dict]:
    path = overrides_path(wave)
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def save_overrides(wave: str, recipes: dict[str, dict]) -> Path:
    path = overrides_path(wave)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipes, indent=2) + "\n")
    return path


def queue_for_wave(wave: str | None = None) -> list[str]:
    policy = effective_policy()
    wave = wave or policy["active_wave"]
    recipes = load_overrides(wave)
    if not recipes:
        return []
    return sorted(recipes.keys())


def scan_wave_metrics(wave: str, city: str = "melbourne") -> dict[str, Any]:
    recipes = load_overrides(wave)
    results_root = ROOT / "results" / city
    done = 0
    lifts = 0
    top = 0.0
    best = None
    rows: list[dict[str, Any]] = []
    policy = effective_policy()
    lock = float(policy["lock_acc"])
    for stem in sorted(recipes.keys()):
        metrics = results_root / stem / "eval_test" / "metrics.json"
        if not metrics.is_file():
            continue
        try:
            acc = (json.loads(metrics.read_text()).get("known_only") or {}).get("Acc@1")
        except Exception:
            continue
        if acc is None:
            continue
        acc_f = float(acc)
        done += 1
        row = {"tag": stem, "acc": acc_f, "lift": acc_f > lock}
        rows.append(row)
        if acc_f > top:
            top = acc_f
            best = stem
        if acc_f > lock:
            lifts += 1
    rows.sort(key=lambda r: -r["acc"])
    return {
        "wave": wave,
        "total": len(recipes),
        "done": done,
        "pending": max(0, len(recipes) - done),
        "top": top,
        "best": best,
        "lifts": lifts,
        "complete": bool(recipes) and done >= len(recipes),
        "top_rows": rows[:10],
    }


def write_harvest(snapshot: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = dict(snapshot)
    snapshot["written_at"] = _now()
    HARVEST_PATH.write_text(json.dumps(snapshot, indent=2) + "\n")
    with HARVEST_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snapshot, separators=(",", ":")) + "\n")


def _fmt_tag_value(v: Any) -> str:
    if isinstance(v, float):
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        if "." in s:
            # 0.078 -> 0078, 0.1 -> 010 style compact
            frac = s.split(".", 1)[1]
            return frac.ljust(2, "0")[:4]
        return s
    return str(v).replace(".", "").replace("-", "m")


def build_next_wave_recipes(
    policy: dict[str, Any],
    state: dict[str, Any],
    new_wave: str,
) -> dict[str, dict]:
    """Synthesize densify overrides for ``new_wave`` without repeating tried fps."""
    tried = set(state.get("tried_fingerprints") or [])
    # mark current wave fps as tried
    for ov in load_overrides(policy["active_wave"]).values():
        tried.add(fingerprint(ov))

    phase = wave_num(new_wave)
    families = list(policy.get("densify_families") or [])
    # rotate family order by wave number so grids shift each wave
    if families:
        rot = phase % len(families)
        families = families[rot:] + families[:rot]

    recipes: dict[str, dict] = {}
    target_n = int(policy.get("wave_size") or 72)

    def try_add(tag: str, overrides: dict[str, Any]) -> None:
        if len(recipes) >= target_n:
            return
        # nudge numeric values slightly by wave phase so grids aren't identical
        ov = deepcopy(overrides)
        for k, v in list(ov.items()):
            if isinstance(v, float) and k != "seed":
                # tiny phase nudge ± up to ~1% every 3 waves, clipped later by uniqueness
                nudge = 1.0 + 0.002 * ((phase % 5) - 2)
                ov[k] = round(v * nudge, 4)
            elif isinstance(v, int) and k == "seed":
                ov[k] = int(v) + ((phase - 17) % 7)  # mild seed drift
        fp = fingerprint(ov)
        if fp in tried:
            return
        stem = stem_for(policy, new_wave, tag)
        if stem in recipes:
            return
        recipes[stem] = ov
        tried.add(fp)

    # single-knob densify
    for fam in families:
        key = fam["key"]
        tag_pfx = fam["tag"]
        values = list(fam.get("values") or [])
        # shift value window
        if values:
            shift = phase % max(1, len(values))
            values = values[shift:] + values[:shift]
        for v in values:
            if isinstance(v, float):
                tag = f"{tag_pfx}{_fmt_tag_value(v)}"
            else:
                tag = f"{tag_pfx}{v}"
            try_add(tag, {key: v})
            if len(recipes) >= target_n:
                break
        if len(recipes) >= target_n:
            break

    # combo recipes (rotated)
    combos = list(policy.get("combo_recipes") or [])
    if combos:
        shift = phase % len(combos)
        combos = combos[shift:] + combos[:shift]
    for combo in combos:
        try_add(combo["tag"], dict(combo.get("overrides") or {}))
        if len(recipes) >= target_n:
            break

    # if still short, add pairwise soft×pmi / bpr×jump fillers
    if len(recipes) < target_n:
        softs = [0.07, 0.074, 0.08, 0.085, 0.09]
        pmis = [0.03, 0.04, 0.06, 0.07, 0.08]
        for i, (s, p) in enumerate(zip(softs, pmis)):
            try_add(f"soft{_fmt_tag_value(s)}_pmi{_fmt_tag_value(p)}", {
                "soft_target_beta": s,
                "pmi_score_bias": p,
            })
            if len(recipes) >= target_n:
                break

    state["tried_fingerprints"] = sorted(tried)[-5000:]  # bound growth
    return recipes


def draft_local_win(best_tag: str, acc: float, policy: dict[str, Any]) -> Path | None:
    """Copy best wave yaml → runs/climb_state draft WIN (not git)."""
    src = ROOT / "configs" / "melbourne" / f"{best_tag}.yaml"
    if not src.is_file():
        return None
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    dst = STATE_DIR / f"draft_win_acc{acc:.6f}_{best_tag}.yaml"
    text = src.read_text()
    header = (
        f"# LOCAL DRAFT WIN Acc@1={acc:.6f} · {_now()}\n"
        f"# Promote later via scripts/promote_climb_config.py (do not git-add wave yaml).\n"
    )
    dst.write_text(header + text)
    (STATE_DIR / "draft_win_latest.yaml").write_text(header + text)
    return dst


def apply_lift_to_state(state: dict[str, Any], wave: str, best_tag: str, acc: float) -> None:
    """Absorb winning overrides into local state base + raise lock (not git policy)."""
    policy = load_policy()
    recipes = load_overrides(wave)
    ov = recipes.get(best_tag) or {}
    base = dict(state.get("base") or policy.get("base") or {})
    base.update(ov)
    state["base"] = base
    state["lock_acc"] = float(acc)


def advance_wave_if_needed(force: bool = False) -> dict[str, Any]:
    """If active wave is complete, snapshot harvest and optionally open next wave."""
    from gen_mel_wave_local import ensure_wave_configs  # local import

    state = load_state()
    policy = effective_policy()
    wave = policy["active_wave"]
    summary = scan_wave_metrics(wave)
    snapshot = {
        "event": "wave_progress",
        "wave": wave,
        "lock_acc": float(policy["lock_acc"]),
        "target_acc": float(policy["target_acc"]),
        **summary,
        "auto_advance": bool(policy.get("auto_advance", True)),
    }

    write_harvest(snapshot)

    if state.get("target_met") and not force:
        snapshot["event"] = "target_met_idle"
        write_harvest(snapshot)
        return snapshot

    if not summary["complete"] and not force:
        snapshot["event"] = "in_progress"
        write_harvest(snapshot)
        return snapshot

    top = float(summary["top"] or 0.0)
    best = summary["best"]
    lock = float(policy["lock_acc"])
    lifted = top > lock
    snapshot["event"] = "wave_complete"
    snapshot["lifted"] = lifted
    write_harvest(snapshot)

    if wave not in state.get("waves_done", []):
        state.setdefault("waves_done", []).append(wave)

    if lifted and best:
        draft_local_win(best, top, policy)
        apply_lift_to_state(state, wave, best, top)
        state["best_acc"] = top
        state["best_tag"] = best
        if top >= float(policy["target_acc"]):
            state["target_met"] = True
            save_state(state)
            snapshot["event"] = "target_met"
            write_harvest({**snapshot, "event": "target_met", "lock_acc": top})
            return snapshot
    else:
        state["best_acc"] = max(float(state.get("best_acc") or 0.0), top)
        if best and not state.get("best_tag"):
            state["best_tag"] = best

    auto = state.get("auto_advance", policy.get("auto_advance", True))
    if not auto:
        save_state(state)
        snapshot["event"] = "wave_complete_await_manual"
        write_harvest(snapshot)
        return snapshot

    new_wave = next_wave_id(wave)
    # rebuild effective policy after possible lift base update
    save_state(state)
    policy = effective_policy()
    recipes = build_next_wave_recipes(policy, state, new_wave)
    if not recipes:
        snapshot["event"] = "advance_failed_empty"
        write_harvest(snapshot)
        save_state(state)
        return snapshot

    save_overrides(new_wave, recipes)
    state["active_wave"] = new_wave
    save_state(state)

    n = ensure_wave_configs(new_wave, "melbourne")
    out = {
        **snapshot,
        "event": "advanced",
        "new_wave": new_wave,
        "new_total": len(recipes),
        "ensured_yamls": n,
        "lifted": lifted,
        "lock_acc": float(state["lock_acc"]),
    }
    write_harvest(out)
    return out


def ensure_active_wave_ready() -> tuple[str, list[str]]:
    """Ensure overrides+yamls exist for active wave; return (wave, queue)."""
    from gen_mel_wave_local import ensure_wave_configs

    state = load_state()
    policy = effective_policy()
    wave = policy["active_wave"]

    recipes = load_overrides(wave)
    if not recipes:
        # synthesize (should not happen for committed w17)
        recipes = build_next_wave_recipes(policy, state, wave)
        save_overrides(wave, recipes)
        save_state(state)

    ensure_wave_configs(wave, "melbourne")
    return wave, sorted(recipes.keys())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true", help="Print harvest snapshot JSON")
    ap.add_argument("--advance", action="store_true", help="Advance wave if complete")
    ap.add_argument("--force-advance", action="store_true", help="Advance even if incomplete")
    ap.add_argument("--ensure", action="store_true", help="Ensure active wave configs exist")
    args = ap.parse_args()

    if args.ensure:
        wave, q = ensure_active_wave_ready()
        print(json.dumps({"wave": wave, "queue": len(q)}))
        return
    if args.advance or args.force_advance:
        out = advance_wave_if_needed(force=args.force_advance)
        print(json.dumps(out, indent=2))
        return
    policy = effective_policy()
    summary = scan_wave_metrics(policy["active_wave"])
    snap = {
        "event": "status",
        "wave": policy["active_wave"],
        "lock_acc": float(policy["lock_acc"]),
        "target_acc": float(policy["target_acc"]),
        **summary,
    }
    write_harvest(snap)
    print(json.dumps(snap, indent=2))


if __name__ == "__main__":
    main()
