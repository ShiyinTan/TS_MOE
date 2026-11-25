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
LEVER_RANK_PATH = STATE_DIR / "lever_rank.json"


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


def is_real_lift(acc: float, lock: float) -> bool:
    """TIE ≠ lift. Compare known_only Acc@1 at 6 decimal places."""
    return round(float(acc), 6) > round(float(lock), 6)


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
        row = {"tag": stem, "acc": acc_f, "lift": is_real_lift(acc_f, lock)}
        rows.append(row)
        if acc_f > top:
            top = acc_f
            best = stem
        if is_real_lift(acc_f, lock):
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


def _refresh_lever_rank() -> dict[str, Any]:
    """Run historical lever analysis (local, free) and return the report."""
    from mel_lever_rank import analyze

    return analyze()


def _lever_guided_families(policy: dict[str, Any], rank: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Build densify families from lever_rank (drop bans, shrink grids)."""
    ban = set((rank or {}).get("ban_keys") or policy.get("ban_keys") or [])
    suggested = dict((rank or {}).get("suggested_values") or {})
    tag_for = {
        "lambda_bpr": "bpr",
        "jump_length_bias": "jump",
        "pmi_score_bias": "pmi",
        "soft_target_beta": "soft",
        "revisit_score_bias": "rev",
        "geo_score_bias": "geo",
        "weight_decay": "wd",
        "epochs": "ep",
        "transition_attn": "trgate",
    }
    families: list[dict[str, Any]] = []
    # Prefer suggested keep_keys order; else policy densify_families filtered.
    keep_keys = list((rank or {}).get("keep_keys") or [])
    if not keep_keys:
        for fam in policy.get("densify_families") or []:
            if fam.get("key") not in ban:
                families.append(fam)
        return families

    for key in keep_keys:
        if key in ban:
            continue
        if key == "transition_attn":
            # handled via combos / single gate recipe
            families.append({"tag": "trgate", "key": key, "values": ["gate"]})
            continue
        values = suggested.get(key)
        if not values:
            # fall back to policy family values, truncated
            src = next((f for f in (policy.get("densify_families") or []) if f.get("key") == key), None)
            values = list((src or {}).get("values") or [])[:5]
        if not values:
            continue
        families.append({"tag": tag_for.get(key, key[:4]), "key": key, "values": values})
    return families


def build_next_wave_recipes(
    policy: dict[str, Any],
    state: dict[str, Any],
    new_wave: str,
    *,
    use_lever_rank: bool = True,
) -> dict[str, dict]:
    """Synthesize densify overrides for ``new_wave`` without repeating tried fps.

    When lever history exists, drop dead switches (seed/hop/pop/…) and shrink
    grids around near-lock levers (soft/bpr/jump/…).
    """
    tried = set(state.get("tried_fingerprints") or [])
    for ov in load_overrides(policy["active_wave"]).values():
        tried.add(fingerprint(ov))

    rank = None
    if use_lever_rank:
        try:
            rank = _refresh_lever_rank()
            state["lever_rank_at"] = _now()
            state["lever_ban_keys"] = list(rank.get("ban_keys") or [])
        except Exception as e:
            print(f"lever_rank skip: {e!r}", flush=True)

    phase = wave_num(new_wave)
    families = _lever_guided_families(policy, rank)
    # mild rotate among kept families only
    if families:
        rot = phase % len(families)
        families = families[rot:] + families[:rot]

    recipes: dict[str, dict] = {}
    target_n = int(policy.get("wave_size") or 32)
    ban = set((rank or {}).get("ban_keys") or policy.get("ban_keys") or [])

    def try_add(tag: str, overrides: dict[str, Any]) -> None:
        if len(recipes) >= target_n:
            return
        if any(k in ban for k in overrides):
            return
        ov = deepcopy(overrides)
        # No random nudge: keep exact grids so history comparisons stay clean.
        for k, v in list(ov.items()):
            if isinstance(v, float):
                ov[k] = round(v, 4)
        fp = fingerprint(ov)
        if fp in tried:
            return
        stem = stem_for(policy, new_wave, tag)
        if stem in recipes:
            return
        recipes[stem] = ov
        tried.add(fp)

    # 1) priority combos first (highest ROI)
    combos = list((rank or {}).get("suggested_combos") or policy.get("combo_recipes") or [])
    for combo in combos:
        try_add(combo["tag"], dict(combo.get("overrides") or {}))
        if len(recipes) >= target_n:
            break

    # 2) single-knob densify on kept/narrowed families
    for fam in families:
        key = fam["key"]
        tag_pfx = fam["tag"]
        values = list(fam.get("values") or [])
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

    # 3) filler: soft × jump / soft × bpr near WIN only
    if len(recipes) < target_n:
        softs = (rank or {}).get("suggested_values", {}).get("soft_target_beta") or [
            0.074, 0.076, 0.08, 0.082, 0.085
        ]
        jumps = (rank or {}).get("suggested_values", {}).get("jump_length_bias") or [
            0.09, 0.1, 0.11, 0.12
        ]
        bprs = (rank or {}).get("suggested_values", {}).get("lambda_bpr") or [
            0.2, 0.202, 0.205, 0.208
        ]
        for s in softs:
            for j in jumps[:3]:
                try_add(
                    f"soft{_fmt_tag_value(s)}_jump{_fmt_tag_value(j)}",
                    {"soft_target_beta": s, "jump_length_bias": j},
                )
                if len(recipes) >= target_n:
                    break
            if len(recipes) >= target_n:
                break
        for s in softs[:4]:
            for b in bprs[:3]:
                try_add(
                    f"soft{_fmt_tag_value(s)}_bpr{_fmt_tag_value(b)}",
                    {"soft_target_beta": s, "lambda_bpr": b},
                )
                if len(recipes) >= target_n:
                    break
            if len(recipes) >= target_n:
                break

    state["tried_fingerprints"] = sorted(tried)[-5000:]
    return recipes


def prefetch_next_wave() -> dict[str, Any]:
    """Open the next densify wave early while the current wave is still draining.

    Keeps GPU slots full: when launchable pending hits 0 but jobs are still
    running, synthesize/switch to ``wN+1`` without waiting for full completion.
    """
    from gen_mel_wave_local import ensure_wave_configs

    state = load_state()
    policy = effective_policy()
    wave = policy["active_wave"]
    nxt = next_wave_id(wave)

    if load_overrides(nxt):
        state["active_wave"] = nxt
        save_state(state)
        n = ensure_wave_configs(nxt, "melbourne")
        out = {
            "event": "prefetch_switch",
            "from_wave": wave,
            "new_wave": nxt,
            "new_total": len(load_overrides(nxt)),
            "ensured_yamls": n,
        }
        write_harvest(out)
        return out

    # Build next wave from lever rank now (current wave fps already in tried).
    recipes = build_next_wave_recipes(policy, state, nxt, use_lever_rank=True)
    if not recipes:
        return {"event": "prefetch_failed_empty", "wave": wave}
    save_overrides(nxt, recipes)
    state["active_wave"] = nxt
    save_state(state)
    n = ensure_wave_configs(nxt, "melbourne")
    out = {
        "event": "prefetch_advanced",
        "from_wave": wave,
        "new_wave": nxt,
        "new_total": len(recipes),
        "ensured_yamls": n,
        "ban_keys": state.get("lever_ban_keys"),
    }
    write_harvest(out)
    return out


def rebuild_active_wave(*, keep_done: bool = True) -> dict[str, Any]:
    """Replace the active wave queue with a lever-pruned set (GPU-local).

    Already-finished recipes are kept so metrics are not discarded; pending
    dead-lever jobs are dropped from the overrides queue so packer stops
    launching them.
    """
    from gen_mel_wave_local import ensure_wave_configs

    state = load_state()
    policy = effective_policy()
    wave = policy["active_wave"]
    old = load_overrides(wave)
    done_stems = {
        stem
        for stem in old
        if (ROOT / "results" / "melbourne" / stem / "eval_test" / "metrics.json").is_file()
    }

    # Synthesize as if advancing into this same wave id (fingerprints exclude done).
    # Temporarily pretend active_wave is previous so tried set still loads old fps.
    recipes_new = build_next_wave_recipes(policy, state, wave, use_lever_rank=True)

    if keep_done:
        for stem in done_stems:
            recipes_new[stem] = old.get(stem) or recipes_new.get(stem) or {}

    save_overrides(wave, recipes_new)
    save_state(state)
    n = ensure_wave_configs(wave, "melbourne")
    pending = [s for s in recipes_new if s not in done_stems]
    out = {
        "event": "rebuild_active",
        "wave": wave,
        "kept_done": len(done_stems),
        "new_pending": len(pending),
        "total": len(recipes_new),
        "ban_keys": state.get("lever_ban_keys"),
        "ensured_yamls": n,
    }
    write_harvest({**out, **scan_wave_metrics(wave)})
    return out


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
    lifted = is_real_lift(top, lock)
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
        if round(top, 6) >= round(float(policy["target_acc"]), 6):
            state["target_met"] = True
            save_state(state)
            snapshot["event"] = "target_met"
            write_harvest({**snapshot, "event": "target_met", "lock_acc": top})
            return snapshot
    else:
        # Keep canonical lock (do not bump on TIE float noise).
        state["lock_acc"] = round(lock, 6)
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
    ap.add_argument(
        "--rebuild-active",
        action="store_true",
        help="Rebuild active wave from lever_rank (drop dead pending levers)",
    )
    ap.add_argument("--rank-levers", action="store_true", help="Refresh lever_rank.json only")
    ap.add_argument(
        "--prefetch",
        action="store_true",
        help="Open next wave early to fill free GPUs while current drains",
    )
    args = ap.parse_args()

    if args.prefetch:
        out = prefetch_next_wave()
        print(json.dumps(out, indent=2))
        return
    if args.rank_levers:
        report = _refresh_lever_rank()
        print(json.dumps({"ban_keys": report.get("ban_keys"), "keep_keys": report.get("keep_keys"), "n": report.get("n_results")}))
        return
    if args.rebuild_active:
        out = rebuild_active_wave(keep_done=True)
        print(json.dumps(out, indent=2))
        return
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
