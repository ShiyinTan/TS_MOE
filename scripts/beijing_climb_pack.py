#!/usr/bin/env python3
"""Beijing-only Acc climb packer: fill free GPU slots (cap 8) with climb configs.
Never touches foreign processes. Prefer configs under configs/beijing/climb_*.yaml.
"""
from __future__ import annotations

import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("TSMOE_PYTHON", "/home/shiyintan/miniconda3/envs/tsmoe/bin/python")
MAX_OUR = 8
MIN_FREE_MIB = 3500
LOG = ROOT / "runs" / "packer_logs"
PIDDIR = ROOT / "logs" / "multicity"

# Priority order: compose around city best ep2_trbias, then filter_last
QUEUE = [
    "climb_ep2_trbias_soft008",
    "climb_ep2_trbias_soft011",
    "climb_ep2_trbias_soft013",
    "climb_ep2_trbias_geo005",
    "climb_ep2_trbias_geo012",
    "climb_ep2_trbias_geo018",
    "climb_ep2_trbias_geo025",
    "climb_ep2_trbias_rev008",
    "climb_ep2_trbias_rev012",
    "climb_ep2_trbias_rev020",
    "climb_ep2_trbias_bpr027",
    "climb_ep2_trbias_bpr031",
    "climb_ep2_trbias_bpr033",
    "climb_ep2_trbias_tsb18",
    "climb_ep2_trbias_tsb22",
    "climb_ep2_trbias_hh005",
    "climb_ep2_trbias_hh010",
    "climb_ep2_trbias_soft005_geo020",
    "climb_ep2_trbias_soft005_rev015",
    "climb_ep4_trbias_soft005",
]


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


def done(tag: str) -> bool:
    return (ROOT / "results" / "beijing" / tag / "eval_test" / "metrics.json").exists()


def running(tag: str, live_cmds: set[str]) -> bool:
    needle = f"configs/beijing/{tag}.yaml"
    out_needle = f"--out runs/beijing/{tag}"
    for cmd in live_cmds:
        if needle in cmd or out_needle in cmd:
            return True
    # pid file
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
            cache = line.split(":", 1)[1].strip()
            break
    return (ROOT / cache / "manifest.json").exists()


def launch(tag: str, gpu: int) -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    PIDDIR.mkdir(parents=True, exist_ok=True)
    cfg = f"configs/beijing/{tag}.yaml"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    lf = open(LOG / f"beijing_{tag}.log", "ab")
    lf.write(f"\n# beijing_climb_pack gpu={gpu}\n".encode())
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
    print(f"beijing climb packer queue={len(QUEUE)}", flush=True)
    while True:
        uu = uuid_map()
        by, live = our_by_gpu(uu)
        free = gpu_free()
        pending = [t for t in QUEUE if not done(t) and not running(t, live) and cache_ready(t)]
        launched = 0
        order = sorted(range(4), key=lambda g: (by[g], -free.get(g, 0)))
        for g in order:
            while by[g] < MAX_OUR and free.get(g, 0) >= MIN_FREE_MIB and pending:
                tag = pending.pop(0)
                launch(tag, g)
                by[g] += 1
                launched += 1
        print(
            f"tick launched={launched} per_gpu={[by[g] for g in range(4)]} "
            f"free={[free.get(g, 0) for g in range(4)]} pending~={len(pending)}",
            flush=True,
        )
        if not pending and launched == 0:
            # exit when all done or waiting only on missing cache
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
