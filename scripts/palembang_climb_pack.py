#!/usr/bin/env python3
"""Palembang Acc climb packer — f63 shortlist densify on L128 (GPU-local)."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gpu_pack_limits import (  # noqa: E402
    MIN_FREE_VRAM_MIB,
    city_cap,
    disk_ok,
    gpu_free_mib,
    host_ram_ok,
    max_our_per_gpu,
    peer_climb_packers_alive,
    resource_snapshot,
    scan_gpus,
)
from city_climb_autonomy import (  # noqa: E402
    advance_wave_if_needed,
    effective_policy,
    load_overrides,
    load_state,
    prefetch_next_wave,
    save_state,
    scan_wave_metrics,
    write_harvest,
)
from gen_city_wave_local import ensure_wave_configs  # noqa: E402

CITY = "palembang"
PY = os.environ.get("TSMOE_PYTHON", "/home/shiyintan/miniconda3/envs/tsmoe/bin/python")
MIN_FREE_MIB = MIN_FREE_VRAM_MIB
LOG = ROOT / "runs" / "packer_logs"
PIDDIR = ROOT / "logs" / "multicity"
STOP_SENTINEL = Path(
    os.environ.get("TSMOE_STOP_SENTINEL", str(ROOT / "logs" / "multicity" / "STOP"))
)
PACK_LOCK = PIDDIR / "packer_launch.lock"
CITY_GPU_CAP = max_our_per_gpu(CITY, shared=False)


def ensure_active_wave_ready() -> tuple[str, list[str]]:
    pol = effective_policy(CITY)
    st = load_state(CITY)
    wave = st.get("active_wave") or pol["active_wave"]
    ensure_wave_configs(wave, CITY)
    recipes = load_overrides(CITY, wave)
    return wave, sorted(recipes.keys())


def stop_requested() -> bool:
    if STOP_SENTINEL.exists():
        return True
    for alt in (ROOT / "logs" / "STOP", ROOT / "runs" / "packer_logs" / "STOP"):
        if alt.exists():
            return True
    return False


def done(tag: str) -> bool:
    return (ROOT / "results" / CITY / tag / "eval_test" / "metrics.json").exists()


def failed(tag: str) -> bool:
    log = LOG / f"{CITY}_{tag}.log"
    if not log.exists():
        return False
    try:
        text = log.read_text(errors="ignore")
    except Exception:
        return False
    marker = f"# {CITY}_climb_pack "
    idx = text.rfind(marker)
    tail = text[idx:] if idx >= 0 else text[-4000:]
    return (
        "__main__.py: error:" in tail
        or "ValueError:" in tail
        or "Traceback (most recent call last):" in tail
    )


def running(tag: str, live_cmds: set[str]) -> bool:
    needle = f"configs/{CITY}/{tag}.yaml"
    for cmd in live_cmds:
        if needle in cmd or f"--out runs/{CITY}/{tag}" in cmd:
            return True
    pf = PIDDIR / f"{CITY}_{tag}.pid"
    if pf.exists():
        pid = pf.read_text().strip()
        if pid and Path(f"/proc/{pid}").exists():
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(
                    errors="ignore"
                )
            except Exception:
                cmd = ""
            if tag in cmd:
                return True
    return False


def cache_ready(tag: str) -> bool:
    cfg = ROOT / "configs" / CITY / f"{tag}.yaml"
    if not cfg.exists():
        return False
    cache = f"runs/{CITY}/cache_abl_tgt_last"
    for line in cfg.read_text().splitlines():
        if line.strip().startswith("cache:"):
            cache = line.split(":", 1)[1].strip().strip("\"'")
            break
    return (ROOT / cache / "manifest.json").exists()


def launch(tag: str, gpu: int, gpu_limit: int, wave: str) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    PIDDIR.mkdir(parents=True, exist_ok=True)
    cfg_path = ROOT / "configs" / CITY / f"{tag}.yaml"
    if not cfg_path.is_file():
        ensure_wave_configs(wave, CITY)
    if not cfg_path.is_file():
        print(f"SKIP missing {cfg_path}", flush=True)
        return -1
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    lf = open(LOG / f"{CITY}_{tag}.log", "ab")
    lf.write(f"\n# {CITY}_climb_pack gpu={gpu} limit={gpu_limit} wave={wave}\n".encode())
    proc = subprocess.Popen(
        [PY, "-m", "tsmoe", "train", "--config", str(cfg_path.relative_to(ROOT))],
        cwd=str(ROOT),
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (PIDDIR / f"{CITY}_{tag}.pid").write_text(str(proc.pid))
    print(f"LAUNCH {CITY}/{tag} gpu={gpu} pid={proc.pid}", flush=True)
    return proc.pid


def main() -> None:
    wave, queue = ensure_active_wave_ready()
    pol = effective_policy(CITY)
    print(
        f"{CITY} climb packer wave={wave} queue={len(queue)} "
        f"max_per_gpu={CITY_GPU_CAP} target={pol['target_acc']} sthgcn={pol['sthgcn']} | "
        f"{resource_snapshot(ROOT / 'runs')}",
        flush=True,
    )
    while True:
        if stop_requested():
            print("STOP — exit", flush=True)
            break
        pol = effective_policy(CITY)
        st = load_state(CITY)
        if st.get("target_met"):
            print(
                f"target_met Acc≥{pol.get('target_acc')} — handoff next city (no idle)",
                flush=True,
            )
            try:
                subprocess.check_call(
                    [PY, str(ROOT / "scripts" / "city_climb_handoff.py"), "--from", CITY],
                    cwd=str(ROOT),
                )
            except Exception as e:
                print(f"handoff error: {e!r}", flush=True)
            break

        wave, queue = ensure_active_wave_ready()
        if not host_ram_ok() or not disk_ok(ROOT / "runs"):
            time.sleep(45)
            continue
        launched = 0
        pending: list[str] = []
        by_all = {g: 0 for g in range(4)}
        by_city = {g: 0 for g in range(4)}
        limits = {g: CITY_GPU_CAP for g in range(4)}
        free = {g: 0 for g in range(4)}
        live: set[str] = set()
        with open(PACK_LOCK, "a+", encoding="utf-8") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            try:
                by_all, by_city, live, limits, _foreign = scan_gpus(CITY)
                free = gpu_free_mib()
                pending = [
                    t
                    for t in queue
                    if not done(t) and not failed(t) and not running(t, live) and cache_ready(t)
                ]
                order = sorted(range(4), key=lambda g: (by_all[g], by_city[g], -free.get(g, 0)))
                for g in order:
                    lim = limits.get(g, CITY_GPU_CAP)
                    other = by_all[g] - by_city[g]
                    peers = peer_climb_packers_alive(CITY)
                    ccap = city_cap(lim, other_city_on_gpu=other, peer_packers=peers)
                    while (
                        by_all[g] < lim
                        and by_city[g] < ccap
                        and free.get(g, 0) >= MIN_FREE_MIB
                        and pending
                    ):
                        if stop_requested() or not host_ram_ok():
                            break
                        tag = pending.pop(0)
                        pid = launch(tag, g, lim, wave)
                        if pid < 0:
                            continue
                        by_all[g] += 1
                        by_city[g] += 1
                        free[g] = free.get(g, 0) - 2000
                        launched += 1
            finally:
                if launched:
                    time.sleep(2)
                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
        print(
            f"tick wave={wave} launched={launched} all={[by_all[g] for g in range(4)]} "
            f"pending~={len(pending)}",
            flush=True,
        )
        summary: dict = {}
        try:
            summary = scan_wave_metrics(CITY, wave)
            write_harvest(
                {
                    "event": "tick",
                    "wave": wave,
                    "lock_acc": float(pol["lock_acc"]),
                    "target_acc": float(pol["target_acc"]),
                    **summary,
                },
                CITY,
            )
        except Exception as e:
            print(f"harvest snapshot skip: {e!r}", flush=True)

        top = float(summary.get("top") or 0.0)
        best = summary.get("best")
        if round(top, 6) >= round(float(pol["target_acc"]), 6):
            st["target_met"] = True
            st["lock_acc"] = top
            st["best_tag"] = best
            save_state(CITY, st)
            print(f"TARGET HIT {top:.6f} best={best}", flush=True)
            subprocess.call(
                [PY, str(ROOT / "scripts" / "city_climb_handoff.py"), "--from", CITY],
                cwd=ROOT,
            )
            break

        if not pending and launched == 0:
            still = [t for t in queue if not done(t)]
            if not still:
                print(f"{CITY} wave {wave} DONE — autonomy advance", flush=True)
                try:
                    out = advance_wave_if_needed(CITY, force=False)
                    print(f"autonomy: {out.get('event')} {out}", flush=True)
                except Exception as e:
                    print(f"autonomy error: {e!r}", flush=True)
                time.sleep(10)
                continue
            under = any(by_all[g] < limits.get(g, CITY_GPU_CAP) for g in range(4))
            if under:
                print(
                    f"{CITY} wave {wave} draining (still={len(still)}) with free GPU — prefetch",
                    flush=True,
                )
                try:
                    out = prefetch_next_wave(CITY)
                    print(f"prefetch: {out.get('event')} {out}", flush=True)
                except Exception as e:
                    print(f"prefetch error: {e!r}", flush=True)
        time.sleep(45)


if __name__ == "__main__":
    main()
