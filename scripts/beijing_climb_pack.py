#!/usr/bin/env python3
"""Beijing Acc climb packer (parallel cities; dynamic GPU caps).

Policy: after a city beats STHGCN, start the next city but keep optimizing the
winner. Per-GPU our-job caps via ``gpu_pack_limits``:
  - 4 when GPU has no foreign users (``dl-box`` ignored)
  - 2 when another user (e.g. camob) is on that GPU
Total our jobs per GPU are capped; cities share via ``all_counts``.
Over-limit only blocks new launches — never kills already-running jobs.

Never touches foreign processes. Prefer configs under configs/beijing/climb_*.yaml.
Exits immediately if logs/multicity/STOP (or TSMOE_STOP_SENTINEL) exists.
"""
from __future__ import annotations

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
    city_cap,
    scan_gpus,
)

CITY = "beijing"
PY = os.environ.get("TSMOE_PYTHON", "/home/shiyintan/miniconda3/envs/tsmoe/bin/python")
MIN_FREE_MIB = int(os.environ.get("TSMOE_MIN_FREE_MIB", "2500"))
LOG = ROOT / "runs" / "packer_logs"
PIDDIR = ROOT / "logs" / "multicity"
STOP_SENTINEL = Path(
    os.environ.get("TSMOE_STOP_SENTINEL", str(ROOT / "logs" / "multicity" / "STOP"))
)

QUEUE = [line.strip() for line in """
climb_s5g20_all_gauss020_wd005_w9_soft003_s050
climb_s5g20_all_gauss020_wd005_w9_soft005_s045
climb_s5g20_all_gauss020_wd005_w9_soft005_s055
climb_s5g20_all_gauss020_wd005_w9_soft007_s040
climb_s5g20_all_gauss020_wd005_w9_g016_s005
climb_s5g20_all_gauss020_wd005_w9_g024_s005
climb_s5g20_all_gauss020_wd005_w9_g020_s002
climb_s5g20_all_gauss020_wd005_w9_g020_s015
climb_s5g20_all_gauss020_wd005_w9_geo010_g015
climb_s5g20_all_gauss020_wd005_w9_geo015_g015
climb_s5g20_all_gauss020_wd005_w9_geo005_g018
climb_s5g20_all_gauss020_wd005_w9_wd004_g021
climb_s5g20_all_gauss020_wd005_w9_wd006_g019
climb_s5g20_all_gauss020_wd005_w9_wd009_g020
climb_s5g20_all_gauss020_wd005_w9_ep5
climb_s5g20_all_gauss020_wd005_w9_drop10
climb_s5g20_all_gauss020_wd005_w9_rev015
climb_s5g20_all_gauss020_wd005_w9_rev020
climb_s5g20_all_gauss020_wd005_w9_tsb16
climb_s5g20_all_gauss020_wd005_w9_tsb28
climb_s5g20_all_gauss020_wd005_w9_ssb025
climb_s5g20_all_gauss020_wd005_w9_ssb035
climb_s5g20_all_gauss020_wd005_w9_bpr012
climb_s5g20_all_gauss020_wd005_w9_bpr028
""".strip().splitlines()]


def stop_requested() -> bool:
    if STOP_SENTINEL.exists():
        return True
    for alt in (ROOT / "logs" / "STOP", ROOT / "runs" / "packer_logs" / "STOP"):
        if alt.exists():
            return True
    return False


def gpu_free():
    free = {}
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
    ).strip().splitlines():
        i, m = [x.strip() for x in line.split(",")]
        free[int(i)] = int(float(m))
    return free


def host_ram_ok() -> bool:
    try:
        meminfo = Path("/proc/meminfo").read_text()
    except Exception:
        return True
    avail_kb = None
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            avail_kb = int(line.split()[1])
            break
    if avail_kb is None:
        return True
    return avail_kb >= 16 * 1024 * 1024


def failed(tag: str) -> bool:
    log = LOG / f"{CITY}_{tag}.log"
    if not log.exists():
        return False
    try:
        tail = log.read_text(errors="ignore")[-4000:]
    except Exception:
        return False
    return "__main__.py: error:" in tail or "error: --" in tail


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
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\\0", b" ").decode(errors="ignore")
            except Exception:
                cmd = ""
            if tag in cmd:
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
        f"(foreign except dl-box → shared) stop={STOP_SENTINEL}",
        flush=True,
    )
    while True:
        if stop_requested():
            print(f"STOP sentinel present ({STOP_SENTINEL}) — exit", flush=True)
            break
        if not host_ram_ok():
            print("host MemAvailable < 16GiB — skip launches this tick", flush=True)
            time.sleep(45)
            continue
        by_all, by_city, live, limits, foreign = scan_gpus(CITY)
        free = gpu_free()
        pending = [
            t for t in QUEUE
            if not done(t) and not failed(t) and not running(t, live) and cache_ready(t)
        ]
        launched = 0
        order = sorted(range(4), key=lambda g: (by_all[g], by_city[g], -free.get(g, 0)))
        over = []
        for g in order:
            lim = limits.get(g, MAX_OUR_PER_GPU_FREE)
            ccap = city_cap(lim)
            # Over-limit: skip new launches only — never kill already-running jobs.
            if by_all[g] > lim:
                over.append(f"gpu{g}:{by_all[g]}>lim{lim}")
            while (
                by_all[g] < lim
                and by_city[g] < ccap
                and free.get(g, 0) >= MIN_FREE_MIB
                and pending
            ):
                if stop_requested() or not host_ram_ok():
                    break
                tag = pending.pop(0)
                launch(tag, g, lim)
                by_all[g] += 1
                by_city[g] += 1
                free[g] = free.get(g, 0) - 2000
                launched += 1
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
