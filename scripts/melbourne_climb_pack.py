#!/usr/bin/env python3
"""Melbourne Acc climb packer — GPU-local continuous densify.

* Queue comes from ``scripts/data/mel_<wave>_overrides.json`` (policy.active_wave).
* When the wave finishes, ``mel_climb_autonomy`` writes a local harvest snapshot
  and auto-opens the next wave — **no cloud wake required to keep training**.
* Cloud harvest only reads progress / promotes Acc locks.

Caps via ``gpu_pack_limits`` (≤4 free / ≤3 shared). Never kills foreign or
already-running our jobs. Exits only on STOP sentinel or target_met idle.
"""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gpu_pack_limits import (  # noqa: E402
    MAX_OUR_PER_GPU_FREE,
    MAX_OUR_PER_GPU_SHARED,
    MIN_FREE_VRAM_MIB,
    city_cap,
    disk_ok,
    gpu_free_mib,
    host_ram_ok,
    peer_climb_packers_alive,
    resource_snapshot,
    scan_gpus,
)
from gen_mel_wave_local import ensure_wave_configs  # noqa: E402
from mel_climb_autonomy import (  # noqa: E402
    advance_wave_if_needed,
    effective_policy,
    ensure_active_wave_ready,
    load_state,
    scan_wave_metrics,
    write_harvest,
)

CITY = "melbourne"
PY = os.environ.get("TSMOE_PYTHON", "/home/shiyintan/miniconda3/envs/tsmoe/bin/python")
MIN_FREE_MIB = MIN_FREE_VRAM_MIB
LOG = ROOT / "runs" / "packer_logs"
PIDDIR = ROOT / "logs" / "multicity"
STOP_SENTINEL = Path(
    os.environ.get("TSMOE_STOP_SENTINEL", str(ROOT / "logs" / "multicity" / "STOP"))
)
PACK_LOCK = PIDDIR / "packer_launch.lock"


def stop_requested() -> bool:
    if STOP_SENTINEL.exists():
        return True
    for alt in (ROOT / "logs" / "STOP", ROOT / "runs" / "packer_logs" / "STOP"):
        if alt.exists():
            return True
    return False


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
        or "error: --" in tail
        or "ValueError:" in tail
        or "Traceback (most recent call last):" in tail
    )


def done(tag: str) -> bool:
    return (ROOT / "results" / CITY / tag / "eval_test" / "metrics.json").exists()


def running(tag: str, live_cmds: set[str]) -> bool:
    needle = f"configs/{CITY}/{tag}.yaml"
    out_needle = f"--out runs/{CITY}/{tag}"
    for cmd in live_cmds:
        if needle in cmd or out_needle in cmd:
            return True
    pf = PIDDIR / f"{CITY}_{tag}.pid"
    if pf.exists():
        pid = pf.read_text().strip()
        if pid and Path(f"/proc/{pid}").exists():
            try:
                stat = open(f"/proc/{pid}/stat").read().split()
                state = stat[2] if len(stat) > 2 else ""
            except Exception:
                state = ""
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
            except Exception:
                cmd = ""
            if state != "Z" and tag in cmd:
                return True
            try:
                pf.unlink()
            except Exception:
                pass
    return False


def cache_ready(tag: str) -> bool:
    cfg = ROOT / "configs" / CITY / f"{tag}.yaml"
    if not cfg.exists():
        return False
    text = cfg.read_text()
    cache = f"runs/{CITY}/cache_abl_tgt_last"
    for line in text.splitlines():
        if line.strip().startswith("cache:"):
            cache = line.split(":", 1)[1].strip().strip("\"'")
            break
    return (ROOT / cache / "manifest.json").exists()


def launch(tag: str, gpu: int, gpu_limit: int, wave: str) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    PIDDIR.mkdir(parents=True, exist_ok=True)
    cfg_path = ROOT / "configs" / CITY / f"{tag}.yaml"
    if not cfg_path.is_file():
        try:
            ensure_wave_configs(wave, CITY)
        except FileNotFoundError as e:
            print(f"SKIP ensure failed {e}", flush=True)
            return -1
    if not cfg_path.is_file():
        print(f"SKIP missing local config {cfg_path}", flush=True)
        return -1
    cfg = str(cfg_path.relative_to(ROOT))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    lf = open(LOG / f"{CITY}_{tag}.log", "ab")
    lf.write(
        f"\n# {CITY}_climb_pack gpu={gpu} limit={gpu_limit} "
        f"city_cap={city_cap(gpu_limit)} wave={wave}\n".encode()
    )
    proc = subprocess.Popen(
        [PY, "-m", "tsmoe", "train", "--config", cfg],
        cwd=str(ROOT),
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (PIDDIR / f"{CITY}_{tag}.pid").write_text(str(proc.pid))
    print(f"LAUNCH {CITY}/{tag} gpu={gpu} pid={proc.pid} limit={gpu_limit}", flush=True)
    return proc.pid


def main():
    wave, queue = ensure_active_wave_ready()
    print(
        f"{CITY} climb packer wave={wave} queue={len(queue)} "
        f"max_free={MAX_OUR_PER_GPU_FREE} max_shared={MAX_OUR_PER_GPU_SHARED} "
        f"(foreign except dl-box → shared) stop={STOP_SENTINEL} | "
        f"{resource_snapshot(ROOT / 'runs')}",
        flush=True,
    )
    while True:
        if stop_requested():
            print(f"STOP sentinel present ({STOP_SENTINEL}) — exit", flush=True)
            break

        # Reload state/queue each tick so auto-advance + harvest edits apply live.
        policy = effective_policy()
        state = load_state()
        if state.get("target_met"):
            print(
                f"target_met Acc≥{policy.get('target_acc')} — idle (STOP to exit permanently)",
                flush=True,
            )
            time.sleep(120)
            continue

        wave, queue = ensure_active_wave_ready()

        if not host_ram_ok() or not disk_ok(ROOT / "runs"):
            print(
                f"resource gate — skip launches this tick | {resource_snapshot(ROOT / 'runs')}",
                flush=True,
            )
            time.sleep(45)
            continue

        PIDDIR.mkdir(parents=True, exist_ok=True)
        launched = 0
        over = []
        pending: list[str] = []
        by_all = {g: 0 for g in range(4)}
        by_city = {g: 0 for g in range(4)}
        limits = {g: MAX_OUR_PER_GPU_FREE for g in range(4)}
        foreign = {g: False for g in range(4)}
        free = {g: 0 for g in range(4)}
        live: set[str] = set()

        with open(PACK_LOCK, "a+", encoding="utf-8") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            try:
                by_all, by_city, live, limits, foreign = scan_gpus(CITY)
                free = gpu_free_mib()
                if not host_ram_ok() or not disk_ok(ROOT / "runs"):
                    pending = []
                else:
                    pending = [
                        t
                        for t in queue
                        if not done(t) and not failed(t) and not running(t, live) and cache_ready(t)
                    ]
                order = sorted(range(4), key=lambda g: (by_all[g], by_city[g], -free.get(g, 0)))
                for g in order:
                    lim = limits.get(g, MAX_OUR_PER_GPU_FREE)
                    other = by_all[g] - by_city[g]
                    peers = peer_climb_packers_alive(CITY)
                    ccap = city_cap(lim, other_city_on_gpu=other, peer_packers=peers)
                    if by_all[g] > lim:
                        over.append(f"gpu{g}:{by_all[g]}>lim{lim}")
                    while (
                        by_all[g] < lim
                        and by_city[g] < ccap
                        and free.get(g, 0) >= MIN_FREE_MIB
                        and pending
                    ):
                        if stop_requested() or not host_ram_ok() or not disk_ok(ROOT / "runs"):
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

        if over:
            print(
                f"over-limit (leave running, no kill): {', '.join(over)}",
                flush=True,
            )
        print(
            f"tick wave={wave} launched={launched} all={[by_all[g] for g in range(4)]} "
            f"city={[by_city[g] for g in range(4)]} "
            f"limits={[limits.get(g, MAX_OUR_PER_GPU_FREE) for g in range(4)]} "
            f"foreign={[bool(foreign.get(g)) for g in range(4)]} "
            f"free={[free.get(g, 0) for g in range(4)]} pending~={len(pending)}",
            flush=True,
        )

        # Keep local harvest snapshot fresh for sparse cloud wakes (no launch needed).
        try:
            summary = scan_wave_metrics(wave)
            write_harvest(
                {
                    "event": "tick",
                    "wave": wave,
                    "lock_acc": float(policy["lock_acc"]),
                    "target_acc": float(policy["target_acc"]),
                    **summary,
                }
            )
        except Exception as e:
            print(f"harvest snapshot skip: {e!r}", flush=True)

        if not pending and launched == 0:
            still = [t for t in queue if not done(t)]
            waiting_cache = [t for t in queue if not done(t) and not cache_ready(t)]
            if not still:
                # Wave finished — local autonomy advances; do NOT exit.
                print(f"{CITY} wave {wave} DONE — autonomy advance", flush=True)
                try:
                    out = advance_wave_if_needed(force=False)
                    print(f"autonomy: {out.get('event')} {out}", flush=True)
                except Exception as e:
                    print(f"autonomy error: {e!r}", flush=True)
                time.sleep(10)
                continue
            if waiting_cache and not any(running(t, live) for t in queue):
                print(f"waiting cache for {waiting_cache[:5]}…", flush=True)

        time.sleep(45)


if __name__ == "__main__":
    main()
