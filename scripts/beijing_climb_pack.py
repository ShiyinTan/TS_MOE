#!/usr/bin/env python3
"""Beijing-only Acc climb packer: fill free GPU slots (max 2 of our jobs/GPU).

Never touches foreign processes. Prefer configs under configs/beijing/climb_*.yaml.
Exits immediately if logs/multicity/STOP (or TSMOE_STOP_SENTINEL) exists.
"""
from __future__ import annotations

import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("TSMOE_PYTHON", "/home/shiyintan/miniconda3/envs/tsmoe/bin/python")
# Hard cap: at most 2 of *our* train jobs per GPU (prefs 2026-11-24).
MAX_OUR_PER_GPU = int(os.environ.get("TSMOE_MAX_PER_GPU", "2"))
MIN_FREE_MIB = int(os.environ.get("TSMOE_MIN_FREE_MIB", "3500"))
LOG = ROOT / "runs" / "packer_logs"
PIDDIR = ROOT / "logs" / "multicity"
STOP_SENTINEL = Path(
    os.environ.get("TSMOE_STOP_SENTINEL", str(ROOT / "logs" / "multicity" / "STOP"))
)

# Wave: unused spatial/structure priors on city-best soft005_geo020 (0.420455).
QUEUE = [
    "climb_s5g20_gauss010",
    "climb_s5g20_gauss012",
    "climb_s5g20_geo020_gauss005",
    "climb_s5g20_dim512",
    "climb_s5g20_lr5e4",
    "climb_s5g20_lr2e3",
    "climb_s5g20_tsb_hk2",
    "climb_s5g20_tsb_hk3",
    "climb_s5g20_dtbins",
    "climb_s5g20_temp07",
    "climb_s5g20_filter_gauss015",
    "climb_s5g20_all_gauss015",
]


def stop_requested() -> bool:
    if STOP_SENTINEL.exists():
        return True
    for alt in (ROOT / "logs" / "STOP", ROOT / "runs" / "packer_logs" / "STOP"):
        if alt.exists():
            return True
    return False


def uuid_map():
    uu = {}
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    ).strip().splitlines():
        i, u = [x.strip() for x in line.split(",")]
        uu[u] = int(i)
    return uu


def our_by_gpu(uu):
    by = defaultdict(int)
    live = set()
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    ).strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        g = uu.get(parts[0])
        if g is None:
            continue
        try:
            cmd = open(f"/proc/{parts[1]}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
        except Exception:
            continue
        if "tsmoe" not in cmd:
            continue
        by[g] += 1
        live.add(cmd)
    return by, live


def gpu_free():
    free = {}
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
    ).strip().splitlines():
        i, m = [x.strip() for x in line.split(",")]
        free[int(i)] = int(float(m))
    return free


def host_ram_ok() -> bool:
    """Refuse new launches when available RAM is thin (do not oversubscribe)."""
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
    # Keep ≥16 GiB free for OS + camob + Jupyter.
    return avail_kb >= 16 * 1024 * 1024


def done(tag: str) -> bool:
    return (ROOT / "results" / "beijing" / tag / "eval_test" / "metrics.json").exists()


def running(tag: str, live_cmds: set[str]) -> bool:
    needle = f"configs/beijing/{tag}.yaml"
    out_needle = f"--out runs/beijing/{tag}"
    for cmd in live_cmds:
        if needle in cmd or out_needle in cmd:
            return True
    pf = PIDDIR / f"beijing_{tag}.pid"
    if pf.exists():
        pid = pf.read_text().strip()
        if pid and Path(f"/proc/{pid}").exists():
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
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
    cfg = ROOT / "configs" / "beijing" / f"{tag}.yaml"
    if not cfg.exists():
        return False
    text = cfg.read_text()
    cache = "runs/beijing/cache_abl_tgt_last"
    for line in text.splitlines():
        if line.strip().startswith("cache:"):
            cache = line.split(":", 1)[1].strip().strip("\"'")
            break
    return (ROOT / cache / "manifest.json").exists()


def launch(tag: str, gpu: int) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    PIDDIR.mkdir(parents=True, exist_ok=True)
    cfg = f"configs/beijing/{tag}.yaml"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    lf = open(LOG / f"beijing_{tag}.log", "ab")
    lf.write(f"\n# beijing_climb_pack gpu={gpu} max_per_gpu={MAX_OUR_PER_GPU}\n".encode())
    proc = subprocess.Popen(
        [PY, "-m", "tsmoe", "train", "--config", cfg],
        cwd=str(ROOT),
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (PIDDIR / f"beijing_{tag}.pid").write_text(str(proc.pid))
    print(f"LAUNCH beijing/{tag} gpu={gpu} pid={proc.pid}", flush=True)
    return proc.pid


def main():
    print(
        f"beijing climb packer queue={len(QUEUE)} max_per_gpu={MAX_OUR_PER_GPU} "
        f"stop={STOP_SENTINEL}",
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
        uu = uuid_map()
        by, live = our_by_gpu(uu)
        free = gpu_free()
        pending = [t for t in QUEUE if not done(t) and not running(t, live) and cache_ready(t)]
        launched = 0
        order = sorted(range(4), key=lambda g: (by[g], -free.get(g, 0)))
        for g in order:
            while by[g] < MAX_OUR_PER_GPU and free.get(g, 0) >= MIN_FREE_MIB and pending:
                if stop_requested() or not host_ram_ok():
                    break
                tag = pending.pop(0)
                launch(tag, g)
                by[g] += 1
                # optimistic free-mem decrement (~6 GiB/job for Beijing)
                free[g] = free.get(g, 0) - 6000
                launched += 1
        print(
            f"tick launched={launched} per_gpu={[by[g] for g in range(4)]} "
            f"free={[free.get(g, 0) for g in range(4)]} pending~={len(pending)}",
            flush=True,
        )
        if not pending and launched == 0:
            waiting_cache = [t for t in QUEUE if not done(t) and not cache_ready(t)]
            still = [t for t in QUEUE if not done(t)]
            if not still:
                print("beijing climb queue DONE — exit", flush=True)
                break
            if waiting_cache and not any(running(t, live) for t in QUEUE):
                print(f"waiting cache for {waiting_cache}", flush=True)
        time.sleep(45)


if __name__ == "__main__":
    main()
