#!/usr/bin/env python3
"""Palembang Acc climb packer (parallel cities; dynamic GPU caps).

Policy: after a city beats STHGCN, start the next city but keep optimizing the
winner. Per-GPU our-job caps via ``gpu_pack_limits``:
  - 4 when GPU has no foreign users (``dl-box`` ignored)
  - 4 when another user (e.g. camob) is on that GPU
Total our jobs per GPU are capped; cities share via ``all_counts``.
Over-limit only blocks new launches — never kills already-running jobs.

Never touches foreign processes. Prefer configs under configs/palembang/climb_*.yaml.
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
    city_cap,
    peer_climb_packers_alive,
    scan_gpus,
)

CITY = "palembang"
PY = os.environ.get("TSMOE_PYTHON", "/home/shiyintan/miniconda3/envs/tsmoe/bin/python")
MIN_FREE_MIB = int(os.environ.get("TSMOE_MIN_FREE_MIB", "2500"))
LOG = ROOT / "runs" / "packer_logs"
PIDDIR = ROOT / "logs" / "multicity"
STOP_SENTINEL = Path(
    os.environ.get("TSMOE_STOP_SENTINEL", str(ROOT / "logs" / "multicity" / "STOP"))
)
PACK_LOCK = PIDDIR / "packer_launch.lock"

QUEUE = [line.strip() for line in """
climb_tsb20_bpr020_filter_w18_p80_seed55_soft00748_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0075_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed55_soft00752_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed55_soft00753_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed55_soft00755_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed55_soft00756_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed55_soft00758_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0076_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr025
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr0255
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr026
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr0265
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr0275
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr028
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr0285
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr029
climb_tsb20_bpr020_filter_w18_p80_seed52_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed53_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed54_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed56_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed57_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed58_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed64_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed65_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed66_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed67_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed68_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed70_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_ep2
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_ep3
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_ep4
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_hour002
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_hour0025
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_hour003
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_hour0035
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_hour004
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_tsb175
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_tsb18
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_tsb185
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_tsb19
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_explore003
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_explore005
climb_tsb20_bpr020_filter_w18_p80_seed55_soft0754_bpr027_explore007
climb_tsb20_bpr020_filter_w18_p80_seed55_soft00754_bpr026_ep2
climb_tsb20_bpr020_filter_w18_p80_seed55_soft00754_bpr028_ep2
climb_tsb20_bpr020_filter_w18_p80_seed55_soft00752_bpr027_ep2
climb_tsb20_bpr020_filter_w18_p80_seed55_soft00756_bpr027_ep2
climb_tsb20_bpr020_filter_w18_filt_seed55_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_filt_seed55_soft0754_bpr027_ep2
climb_tsb20_bpr020_filter_w18_tgt_seed55_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_tgt_seed55_soft0754_bpr027_ep2
climb_tsb20_bpr020_filter_w18_abl_seed55_soft0754_bpr027
climb_tsb20_bpr020_filter_w18_abl_seed55_soft0754_bpr027_ep2
climb_tsb20_bpr020_filter_w17_abl_seed55_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_abl_seed55_soft0752_bpr027_ep2
climb_tsb20_bpr020_filter_w17_filt_seed55_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_filt_seed55_soft0752_bpr027_ep2
climb_tsb20_bpr020_filter_w17_p80_seed52_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed53_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed54_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed55_soft00746_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed55_soft00748_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed55_soft00752_bpr026_ep2
climb_tsb20_bpr020_filter_w17_p80_seed55_soft00752_bpr028_ep2
climb_tsb20_bpr020_filter_w17_p80_seed55_soft00754_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed55_soft00754_bpr027_ep2
climb_tsb20_bpr020_filter_w17_p80_seed55_soft00756_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed55_soft00758_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0075_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0075_bpr027_ep2
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0076_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr025
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr0255
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr026
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr0265
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr0275
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_ep2
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_ep3
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_ep4
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_explore003
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_explore005
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_explore007
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_explore008
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_hour002
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_hour0025
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_hour0035
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_hour004
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_hour005
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_hour006
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_tsb175
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_tsb18
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_tsb185
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_tsb19
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr027_tsb20
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr028
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr0285
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr029
climb_tsb20_bpr020_filter_w17_p80_seed55_soft0752_bpr03
climb_tsb20_bpr020_filter_w17_p80_seed56_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed57_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed58_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed64_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed65_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed66_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed67_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed68_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed70_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_p80_seed72_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_tgt_seed55_soft0752_bpr027
climb_tsb20_bpr020_filter_w17_tgt_seed55_soft0752_bpr027_ep2
climb_tsb20_bpr020_filter_w16_abl_seed55_soft0752
climb_tsb20_bpr020_filter_w16_abl_seed55_soft0752_ep2
climb_tsb20_bpr020_filter_w16_abl_seed67_soft0752
climb_tsb20_bpr020_filter_w16_filt_seed55_soft0752
climb_tsb20_bpr020_filter_w16_filt_seed55_soft0752_ep2
climb_tsb20_bpr020_filter_w16_filt_seed67_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed52_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed53_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed54_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed55_soft00746
climb_tsb20_bpr020_filter_w16_p80_seed55_soft00748
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0075
climb_tsb20_bpr020_filter_w16_p80_seed55_soft00751
climb_tsb20_bpr020_filter_w16_p80_seed55_soft00753
climb_tsb20_bpr020_filter_w16_p80_seed55_soft00754
climb_tsb20_bpr020_filter_w16_p80_seed55_soft00754_ep2
climb_tsb20_bpr020_filter_w16_p80_seed55_soft00755
climb_tsb20_bpr020_filter_w16_p80_seed55_soft00756
climb_tsb20_bpr020_filter_w16_p80_seed55_soft00758
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0075_ep2
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0076
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_bpr017
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_bpr018
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_bpr019
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_bpr021
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_bpr022
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_bpr023
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_bpr024
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_bpr025
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_bpr026
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_bpr027
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_ep2
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_ep3
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_ep4
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_explore003
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_explore005
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_explore007
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_explore008
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_hour003
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_hour005
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_hour007
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_tsb175
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_tsb178
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_tsb182
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_tsb185
climb_tsb20_bpr020_filter_w16_p80_seed55_soft0752_tsb19
climb_tsb20_bpr020_filter_w16_p80_seed56_soft00752_ep2
climb_tsb20_bpr020_filter_w16_p80_seed56_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed57_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed58_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed64_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed65_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed66_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed67_soft00752_ep2
climb_tsb20_bpr020_filter_w16_p80_seed67_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed68_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed70_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed72_soft00752_ep2
climb_tsb20_bpr020_filter_w16_p80_seed72_soft0752
climb_tsb20_bpr020_filter_w16_p80_seed73_soft0752
climb_tsb20_bpr020_filter_w16_tgt_seed55_soft0752
climb_tsb20_bpr020_filter_w16_tgt_seed55_soft0752_ep2
climb_tsb20_bpr020_filter_w16_tgt_seed67_soft0752
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
        # Cross-packer lock so BJ/PL don't race-launch past the GPU cap.
        # Over-limit still never kills — lock only serializes new launches.
        PIDDIR.mkdir(parents=True, exist_ok=True)
        launched = 0
        over = []
        with open(PACK_LOCK, "a+", encoding="utf-8") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            try:
                by_all, by_city, live, limits, foreign = scan_gpus(CITY)
                free = gpu_free()
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
                        if stop_requested() or not host_ram_ok():
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
