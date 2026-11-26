#!/usr/bin/env python3
"""GPU-local Acc climb autonomy for any city (MEL lessons → shared loop).

Responsibilities
----------------
* Track active wave / lock / best Acc under ``runs/climb_state/<city>.json``.
* On wave finish: harvest snapshot, absorb real lifts (6dp), open next densify
  wave via lever-ranked soft×jump×bpr×pmi grids (ban seed/hop/pop/…).
* On target_met: finalize Acc ledger + handoff next city (via packer).

    python scripts/city_climb_autonomy.py --city bandung --status
    python scripts/city_climb_autonomy.py --city bandung --advance
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "runs" / "climb_state"
HARVEST_PATH = STATE_DIR / "harvest_latest.json"
HARVEST_LOG = STATE_DIR / "harvest_log.jsonl"

CITY_ALIAS = {
    "melbourne": "mel",
    "bandung": "bdg",
    "beijing": "bj",
    "palembang": "plg",
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def alias_for(city: str) -> str:
    return CITY_ALIAS.get(city, city[:3])


def policy_path(city: str) -> Path:
    return ROOT / "scripts" / "data" / f"{alias_for(city)}_climb_policy.json"


def state_path(city: str) -> Path:
    return STATE_DIR / f"{city}.json"


def overrides_path(city: str, wave: str) -> Path:
    return ROOT / "scripts" / "data" / f"{alias_for(city)}_{wave}_overrides.json"


def ensure_wave_mod(city: str):
    if city == "melbourne":
        from gen_mel_wave_local import ensure_wave_configs

        return ensure_wave_configs
    if city == "bandung":
        from gen_bdg_wave_local import ensure_wave_configs

        return ensure_wave_configs
    raise ValueError(f"no wave generator for city={city}")


def load_policy(city: str) -> dict[str, Any]:
    return json.loads(policy_path(city).read_text())


def load_state(city: str) -> dict[str, Any]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    sp = state_path(city)
    if sp.is_file():
        return json.loads(sp.read_text())
    policy = load_policy(city)
    st = {
        "city": city,
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
    save_state(city, st)
    return st


def save_state(city: str, state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    state_path(city).write_text(json.dumps(state, indent=2) + "\n")


def effective_policy(city: str) -> dict[str, Any]:
    policy = load_policy(city)
    state = load_state(city)
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


def write_harvest(payload: dict[str, Any], city: str | None = None) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    blob = dict(payload)
    blob["written_at"] = _now()
    if city:
        blob["city"] = city
    HARVEST_PATH.write_text(json.dumps(blob, indent=2) + "\n")
    if city:
        (STATE_DIR / f"harvest_{city}.json").write_text(json.dumps(blob, indent=2) + "\n")
    with HARVEST_LOG.open("a") as f:
        f.write(json.dumps(blob, separators=(",", ":")) + "\n")


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
    return round(float(acc), 6) > round(float(lock), 6)


def stem_for(policy: dict[str, Any], wave: str, tag: str) -> str:
    return f"{policy['stem_prefix']}_{wave}_{policy['stem_mid']}_{tag}"


def load_overrides(city: str, wave: str) -> dict[str, dict]:
    path = overrides_path(city, wave)
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def save_overrides(city: str, wave: str, recipes: dict[str, dict]) -> Path:
    path = overrides_path(city, wave)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipes, indent=2) + "\n")
    return path


def _fmt_tag_value(v: Any) -> str:
    if isinstance(v, float):
        s = f"{v:.4f}".rstrip("0").rstrip(".")
        if "." in s:
            frac = s.split(".", 1)[1]
            return frac.ljust(2, "0")[:4]
        return s
    return str(v).replace(".", "").replace("-", "m")


def scan_wave_metrics(city: str, wave: str) -> dict[str, Any]:
    recipes = load_overrides(city, wave)
    top = 0.0
    best = None
    done_n = 0
    for stem in recipes:
        p = ROOT / "results" / city / stem / "eval_test" / "metrics.json"
        if not p.is_file():
            continue
        try:
            a = (json.loads(p.read_text()).get("known_only") or {}).get("Acc@1")
        except Exception:
            continue
        if a is None:
            continue
        done_n += 1
        if float(a) > top:
            top = float(a)
            best = stem
    total = len(recipes)
    return {
        "top": top if done_n else None,
        "best": best,
        "n_done": done_n,
        "n_total": total,
        "n_pending": max(0, total - done_n),
        "complete": total > 0 and done_n >= total,
    }


def _refresh_lever_rank(city: str, lock: float) -> dict[str, Any]:
    from city_lever_rank import analyze

    return analyze(city, lock=lock)


def _lever_guided_families(policy: dict[str, Any], rank: dict[str, Any] | None) -> list[dict[str, Any]]:
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
            families.append({"tag": "trgate", "key": key, "values": ["gate"]})
            continue
        values = suggested.get(key)
        if not values:
            src = next((f for f in (policy.get("densify_families") or []) if f.get("key") == key), None)
            values = list((src or {}).get("values") or [])[:6]
        if not values:
            continue
        families.append({"tag": tag_for.get(key, key[:4]), "key": key, "values": values})
    return families


def build_next_wave_recipes(
    city: str,
    policy: dict[str, Any],
    state: dict[str, Any],
    new_wave: str,
    *,
    use_lever_rank: bool = True,
) -> dict[str, dict]:
    tried = set(state.get("tried_fingerprints") or [])
    for ov in load_overrides(city, policy["active_wave"]).values():
        tried.add(fingerprint(ov))

    rank = None
    if use_lever_rank:
        try:
            rank = _refresh_lever_rank(city, float(policy.get("lock_acc") or 0.0))
            state["lever_rank_at"] = _now()
            state["lever_ban_keys"] = list(rank.get("ban_keys") or [])
        except Exception as e:
            print(f"lever_rank skip: {e!r}", flush=True)

    phase = wave_num(new_wave)
    families = _lever_guided_families(policy, rank)
    if families:
        rot = phase % len(families)
        families = families[rot:] + families[:rot]

    recipes: dict[str, dict] = {}
    target_n = int(policy.get("wave_size") or 28)
    ban = set((rank or {}).get("ban_keys") or policy.get("ban_keys") or [])

    def try_add(tag: str, overrides: dict[str, Any]) -> None:
        if len(recipes) >= target_n:
            return
        if any(k in ban for k in overrides):
            return
        ov = deepcopy(overrides)
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

    combos = list((rank or {}).get("suggested_combos") or policy.get("combo_recipes") or [])
    for combo in combos:
        try_add(combo["tag"], dict(combo.get("overrides") or {}))
        if len(recipes) >= target_n:
            break

    for fam in families:
        key = fam["key"]
        tag_pfx = fam["tag"]
        for v in list(fam.get("values") or []):
            tag = f"{tag_pfx}{_fmt_tag_value(v)}" if isinstance(v, float) else f"{tag_pfx}{v}"
            try_add(tag, {key: v})
            if len(recipes) >= target_n:
                break
        if len(recipes) >= target_n:
            break

    # Filler: MEL highest-ROI soft×jump / soft×bpr
    if len(recipes) < target_n:
        softs = (rank or {}).get("suggested_values", {}).get("soft_target_beta") or [
            0.08, 0.09, 0.1, 0.11, 0.12
        ]
        jumps = (rank or {}).get("suggested_values", {}).get("jump_length_bias") or [
            0.08, 0.09, 0.1, 0.11, 0.12
        ]
        bprs = (rank or {}).get("suggested_values", {}).get("lambda_bpr") or [
            0.22, 0.25, 0.27, 0.29
        ]
        pmis = (rank or {}).get("suggested_values", {}).get("pmi_score_bias") or [
            0.04, 0.05, 0.06, 0.08
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
        for j in jumps[:3]:
            for p in pmis[:3]:
                try_add(
                    f"jump{_fmt_tag_value(j)}_pmi{_fmt_tag_value(p)}",
                    {"jump_length_bias": j, "pmi_score_bias": p},
                )
                if len(recipes) >= target_n:
                    break
            if len(recipes) >= target_n:
                break

    # Last resort: lever bans / tried-set emptied the wave — fall back to policy
    # densify families + combos with *policy* bans only (keep Acc-search alive).
    if not recipes:
        ban = set(policy.get("ban_keys") or [])
        for combo in policy.get("combo_recipes") or []:
            try_add(combo["tag"], dict(combo.get("overrides") or {}))
            if len(recipes) >= target_n:
                break
        for fam in policy.get("densify_families") or []:
            key = fam.get("key")
            if not key or key in ban:
                continue
            tag_pfx = fam.get("tag") or str(key)[:4]
            for v in list(fam.get("values") or []):
                tag = (
                    f"{tag_pfx}{_fmt_tag_value(v)}"
                    if isinstance(v, float)
                    else f"{tag_pfx}{v}"
                )
                try_add(tag, {key: v})
                if len(recipes) >= target_n:
                    break
            if len(recipes) >= target_n:
                break
        # Fine grid around current base for leftover slots.
        if len(recipes) < target_n:
            base = dict(state.get("base") or policy.get("base") or {})
            c_soft = float(base.get("soft_target_beta") or 0.1)
            c_jump = float(base.get("jump_length_bias") or 0.1)
            c_bpr = float(base.get("lambda_bpr") or 0.25)
            c_pmi = float(base.get("pmi_score_bias") or 0.05)
            softs = [round(c_soft + d, 4) for d in (-0.02, -0.01, -0.005, 0.005, 0.01, 0.02)]
            jumps = [round(c_jump + d, 4) for d in (-0.02, -0.01, 0.01, 0.02)]
            bprs = [round(c_bpr + d, 4) for d in (-0.03, -0.02, -0.01, 0.01, 0.02, 0.03)]
            pmis = [round(c_pmi + d, 4) for d in (-0.02, -0.01, 0.01, 0.02)]
            for s in softs:
                for j in jumps:
                    try_add(
                        f"soft{_fmt_tag_value(s)}_jump{_fmt_tag_value(j)}",
                        {"soft_target_beta": s, "jump_length_bias": j},
                    )
                    if len(recipes) >= target_n:
                        break
                if len(recipes) >= target_n:
                    break
            for s in softs:
                for b in bprs:
                    try_add(
                        f"soft{_fmt_tag_value(s)}_bpr{_fmt_tag_value(b)}",
                        {"soft_target_beta": s, "lambda_bpr": b},
                    )
                    if len(recipes) >= target_n:
                        break
                if len(recipes) >= target_n:
                    break
            for j in jumps:
                for p in pmis:
                    try_add(
                        f"jump{_fmt_tag_value(j)}_pmi{_fmt_tag_value(p)}",
                        {"jump_length_bias": j, "pmi_score_bias": p},
                    )
                    if len(recipes) >= target_n:
                        break
                if len(recipes) >= target_n:
                    break

    state["tried_fingerprints"] = sorted(tried)[-5000:]
    return recipes


def draft_local_win(city: str, best_tag: str, acc: float) -> Path | None:
    src = ROOT / "configs" / city / f"{best_tag}.yaml"
    if not src.is_file():
        return None
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    dst = STATE_DIR / f"draft_win_{city}_acc{acc:.6f}_{best_tag}.yaml"
    header = (
        f"# LOCAL DRAFT WIN city={city} Acc@1={acc:.6f} · {_now()}\n"
        f"# Promote later via scripts/promote_climb_config.py\n"
    )
    text = header + src.read_text()
    dst.write_text(text)
    (STATE_DIR / f"draft_win_{city}_latest.yaml").write_text(text)
    return dst


def apply_lift_to_state(
    city: str, state: dict[str, Any], wave: str, best_tag: str, acc: float
) -> None:
    policy = load_policy(city)
    recipes = load_overrides(city, wave)
    ov = recipes.get(best_tag) or {}
    base = dict(state.get("base") or policy.get("base") or {})
    base.update(ov)
    state["base"] = base
    state["lock_acc"] = float(acc)


def run_finalize(city: str, policy: dict[str, Any], lock_acc: float) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "city_climb_finalize.py"),
        "--city",
        city,
        "--sthgcn",
        str(float(policy.get("sthgcn") or 0.0)),
        "--lock-acc",
        str(round(lock_acc, 6)),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.call(cmd, cwd=ROOT)


def prefetch_next_wave(city: str) -> dict[str, Any]:
    ensure = ensure_wave_mod(city)
    state = load_state(city)
    policy = effective_policy(city)
    wave = policy["active_wave"]
    nxt = next_wave_id(wave)
    if load_overrides(city, nxt):
        state["active_wave"] = nxt
        save_state(city, state)
        n = ensure(nxt, city)
        out = {
            "event": "prefetch_switch",
            "city": city,
            "from_wave": wave,
            "new_wave": nxt,
            "new_total": len(load_overrides(city, nxt)),
            "ensured_yamls": n,
        }
        write_harvest(out, city)
        return out
    recipes = build_next_wave_recipes(city, policy, state, nxt, use_lever_rank=True)
    if not recipes:
        return {"event": "prefetch_failed_empty", "city": city, "wave": wave}
    save_overrides(city, nxt, recipes)
    state["active_wave"] = nxt
    save_state(city, state)
    n = ensure(nxt, city)
    out = {
        "event": "prefetch_advanced",
        "city": city,
        "from_wave": wave,
        "new_wave": nxt,
        "new_total": len(recipes),
        "ensured_yamls": n,
        "ban_keys": state.get("lever_ban_keys"),
    }
    write_harvest(out, city)
    return out


def advance_wave_if_needed(city: str, force: bool = False) -> dict[str, Any]:
    ensure = ensure_wave_mod(city)
    state = load_state(city)
    policy = effective_policy(city)
    wave = policy["active_wave"]
    summary = scan_wave_metrics(city, wave)
    snapshot: dict[str, Any] = {
        "event": "wave_progress",
        "city": city,
        "wave": wave,
        "lock_acc": float(policy["lock_acc"]),
        "target_acc": float(policy["target_acc"]),
        **summary,
        "auto_advance": bool(policy.get("auto_advance", True)),
    }
    write_harvest(snapshot, city)

    if state.get("target_met") and not force:
        snapshot["event"] = "target_met_idle"
        write_harvest(snapshot, city)
        return snapshot
    if not summary["complete"] and not force:
        snapshot["event"] = "in_progress"
        write_harvest(snapshot, city)
        return snapshot

    top = float(summary["top"] or 0.0)
    best = summary["best"]
    lock = float(policy["lock_acc"])
    lifted = is_real_lift(top, lock)
    snapshot["event"] = "wave_complete"
    snapshot["lifted"] = lifted
    write_harvest(snapshot, city)

    if wave not in state.get("waves_done", []):
        state.setdefault("waves_done", []).append(wave)

    if lifted and best:
        draft_local_win(city, best, top)
        apply_lift_to_state(city, state, wave, best, top)
        state["best_acc"] = top
        state["best_tag"] = best
        if round(top, 6) >= round(float(policy["target_acc"]), 6):
            state["target_met"] = True
            save_state(city, state)
            snapshot["event"] = "target_met"
            write_harvest({**snapshot, "event": "target_met", "lock_acc": top}, city)
            try:
                run_finalize(city, policy, top)
            except Exception as e:
                print(f"finalize on target_met failed: {e!r}", flush=True)
            return snapshot
    else:
        state["lock_acc"] = round(lock, 6)
        state["best_acc"] = max(float(state.get("best_acc") or 0.0), top)
        if best and not state.get("best_tag"):
            state["best_tag"] = best

    auto = state.get("auto_advance", policy.get("auto_advance", True))
    if not auto:
        save_state(city, state)
        snapshot["event"] = "wave_complete_await_manual"
        write_harvest(snapshot, city)
        return snapshot

    new_wave = next_wave_id(wave)
    save_state(city, state)
    policy = effective_policy(city)
    recipes = build_next_wave_recipes(city, policy, state, new_wave)
    if not recipes:
        snapshot["event"] = "advance_failed_empty"
        write_harvest(snapshot, city)
        save_state(city, state)
        return snapshot

    save_overrides(city, new_wave, recipes)
    state["active_wave"] = new_wave
    save_state(city, state)
    n = ensure(new_wave, city)
    out = {
        **snapshot,
        "event": "advanced",
        "new_wave": new_wave,
        "new_total": len(recipes),
        "ensured_yamls": n,
        "ban_keys": state.get("lever_ban_keys"),
        "lifted": lifted,
        "lock_acc": float(state["lock_acc"]),
    }
    write_harvest(out, city)
    return out


def status(city: str) -> dict[str, Any]:
    policy = effective_policy(city)
    wave = policy["active_wave"]
    summary = scan_wave_metrics(city, wave)
    out = {
        "city": city,
        "wave": wave,
        "lock_acc": float(policy["lock_acc"]),
        "target_acc": float(policy["target_acc"]),
        "sthgcn": float(policy.get("sthgcn") or 0.0),
        **summary,
        "target_met": bool(load_state(city).get("target_met")),
    }
    write_harvest({"event": "status", **out}, city)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--advance", action="store_true")
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    city = args.city
    if args.status:
        print(json.dumps(status(city), indent=2))
        return
    if args.prefetch:
        print(json.dumps(prefetch_next_wave(city), indent=2))
        return
    if args.advance:
        print(json.dumps(advance_wave_if_needed(city, force=args.force), indent=2))
        return
    print(json.dumps(status(city), indent=2))


if __name__ == "__main__":
    main()
