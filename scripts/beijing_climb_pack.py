#!/usr/bin/env python3
"""Beijing Acc climb packer (active beat city; dynamic GPU caps).

Sequential beat policy (prefs 2026-11-25): finish BJ clear margin before
starting next unfinished city. Per-GPU our-job caps via ``gpu_pack_limits``:
  - **2** when GPU has no foreign users (``dl-box`` ignored)
  - **2** when another user (e.g. camob) is on that GPU
Also gate on host RAM / free VRAM / disk. Caps only block new launches —
never kills already-running jobs (ours or foreign).

Never touches foreign processes. Prefer configs under configs/beijing/climb_*.yaml.
Exits immediately if logs/multicity/STOP (or TSMOE_STOP_SENTINEL) exists.
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

CITY = "beijing"
PY = os.environ.get("TSMOE_PYTHON", "/home/shiyintan/miniconda3/envs/tsmoe/bin/python")
MIN_FREE_MIB = MIN_FREE_VRAM_MIB
LOG = ROOT / "runs" / "packer_logs"
PIDDIR = ROOT / "logs" / "multicity"
STOP_SENTINEL = Path(
    os.environ.get("TSMOE_STOP_SENTINEL", str(ROOT / "logs" / "multicity" / "STOP"))
)
PACK_LOCK = PIDDIR / "packer_launch.lock"

QUEUE = [line.strip() for line in """
climb_s5g20_all_gauss020_wd005_w39_tgt_soft00285_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_soft0032_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_soft00365_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_soft0041_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_soft00465_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_soft0051_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_soft0056_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_soft0061_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_soft0068_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_soft0073_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_bpr0065_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_bpr0085_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_bpr0105_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_bpr013_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_bpr0155_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_bpr0185_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_bpr0215_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_bpr0245_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_bpr0275_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_bpr030_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_explore0035_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_explore0055_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_explore007_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_explore009_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_explore012_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_explore014_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_explore016_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_tsb145_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_tsb162_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_tsb178_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_tsb192_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_tsb205_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_tsb218_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_tsb235_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_seed33_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_seed36_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_seed38_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_seed41_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_seed47_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_seed49_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_seed53_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_seed60_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_seed66_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_seed77_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_soft0041_session_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_soft0051_session_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_soft0061_session_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_gauss015_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_gauss018_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_gauss022_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_gauss025_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_wd0002_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_wd00035_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_wd0007_ep3
climb_s5g20_all_gauss020_wd005_w39_tgt_wd001_ep3
climb_s5g20_all_gauss020_wd005_w39_filt_soft0041_ep3
climb_s5g20_all_gauss020_wd005_w39_p70_soft0041_ep3
climb_s5g20_all_gauss020_wd005_w39_p80_soft0041_ep3
climb_s5g20_all_gauss020_wd005_w39_abl_soft0041_ep3
climb_s5g20_all_gauss020_wd005_w39_filt_soft0051_ep3
climb_s5g20_all_gauss020_wd005_w39_p70_soft0051_ep3
climb_s5g20_all_gauss020_wd005_w39_p80_soft0051_ep3
climb_s5g20_all_gauss020_wd005_w39_abl_soft0051_ep3
""".strip().splitlines()]


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
    # Only judge the latest launch segment (log is append-only across retries).
    marker = f"# {CITY}_climb_pack "
    idx = text.rfind(marker)
    tail = text[idx:] if idx >= 0 else text[-4000:]
    # Catch argparse errors and runtime ValueError (e.g. missing candidate_context).
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
            # Zombies / empty cmdline are not live workers — clear pid and retry.
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


def launch(tag: str, gpu: int, gpu_limit: int) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    PIDDIR.mkdir(parents=True, exist_ok=True)
    cfg = f"configs/{CITY}/{tag}.yaml"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    lf = open(LOG / f"{CITY}_{tag}.log", "ab")
    lf.write(
        f"\\n# {CITY}_climb_pack gpu={gpu} limit={gpu_limit} "
        f"city_cap={city_cap(gpu_limit)}\\n".encode()
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
    print(
        f"{CITY} climb packer queue={len(QUEUE)} "
        f"max_free={MAX_OUR_PER_GPU_FREE} max_shared={MAX_OUR_PER_GPU_SHARED} "
        f"(foreign except dl-box → shared) stop={STOP_SENTINEL} | "
        f"{resource_snapshot(ROOT / 'runs')}",
        flush=True,
    )
    while True:
        if stop_requested():
            print(f"STOP sentinel present ({STOP_SENTINEL}) — exit", flush=True)
            break
        if not host_ram_ok() or not disk_ok(ROOT / "runs"):
            print(
                f"resource gate — skip launches this tick | {resource_snapshot(ROOT / 'runs')}",
                flush=True,
            )
            time.sleep(45)
            continue
        # Cross-packer lock so BJ/PL don't race-launch past the GPU cap.
        # Over-limit still never kills — lock only serializes new launches.
        PIDDIR.mkdir(parents=True, exist_ok=True)
        launched = 0
        over = []
        with open(PACK_LOCK, "a+", encoding="utf-8") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            try:
                by_all, by_city, live, limits, foreign = scan_gpus(CITY)
                free = gpu_free_mib()
                if not host_ram_ok() or not disk_ok(ROOT / "runs"):
                    pending = []
                else:
                    pending = [
                        t for t in QUEUE
                        if not done(t) and not failed(t) and not running(t, live) and cache_ready(t)
                    ]
                order = sorted(range(4), key=lambda g: (by_all[g], by_city[g], -free.get(g, 0)))
                for g in order:
                    lim = limits.get(g, MAX_OUR_PER_GPU_FREE)
                    other = by_all[g] - by_city[g]
                    peers = peer_climb_packers_alive(CITY)
                    ccap = city_cap(lim, other_city_on_gpu=other, peer_packers=peers)
                    # Over-limit: skip new launches only — never kill already-running jobs.
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
                        launch(tag, g, lim)
                        by_all[g] += 1
                        by_city[g] += 1
                        free[g] = free.get(g, 0) - 2000
                        launched += 1
            finally:
                # Let nvidia-smi see new PIDs before the other packer scans.
                if launched:
                    time.sleep(2)
                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
        if over:
            print(
                f"over-limit (leave running, no kill): {', '.join(over)}",
                flush=True,
            )
        print(
            f"tick launched={launched} all={[by_all[g] for g in range(4)]} "
            f"city={[by_city[g] for g in range(4)]} "
            f"limits={[limits.get(g, MAX_OUR_PER_GPU_FREE) for g in range(4)]} "
            f"foreign={[bool(foreign.get(g)) for g in range(4)]} "
            f"free={[free.get(g, 0) for g in range(4)]} pending~={len(pending)}",
            flush=True,
        )
        if not pending and launched == 0:
            waiting_cache = [t for t in QUEUE if not done(t) and not cache_ready(t)]
            still = [t for t in QUEUE if not done(t)]
            if not still:
                print(f"{CITY} climb queue DONE — exit", flush=True)
                break
            if waiting_cache and not any(running(t, live) for t in QUEUE):
                print(f"waiting cache for {waiting_cache}", flush=True)
        time.sleep(45)


if __name__ == "__main__":
    main()
