#!/usr/bin/env python3
"""Melbourne Acc climb packer (parallel cities; dynamic GPU caps).

Policy: after a city beats STHGCN, start the next city but keep optimizing the
winner. Per-GPU our-job caps via ``gpu_pack_limits``:
  - **2** when GPU has no foreign users (``dl-box`` ignored)
  - **2** when another user (e.g. camob) is on that GPU
Total our jobs per GPU are capped; cities share via ``all_counts``.
Over-limit only blocks new launches — never kills already-running jobs.

Never touches foreign processes. Prefer configs under configs/melbourne/climb_*.yaml.
Wave densify YAMLs are **gitignored** — materialize with
``scripts/gen_mel_wave_local.py`` (also auto-called here). Commit only after Acc
lock via ``scripts/promote_climb_config.py``.
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
from gen_mel_wave_local import ensure_wave_configs  # noqa: E402

CITY = "melbourne"
PY = os.environ.get("TSMOE_PYTHON", "/home/shiyintan/miniconda3/envs/tsmoe/bin/python")
MIN_FREE_MIB = MIN_FREE_VRAM_MIB
LOG = ROOT / "runs" / "packer_logs"
PIDDIR = ROOT / "logs" / "multicity"
STOP_SENTINEL = Path(
    os.environ.get("TSMOE_STOP_SENTINEL", str(ROOT / "logs" / "multicity" / "STOP"))
)
PACK_LOCK = PIDDIR / "packer_launch.lock"

QUEUE = [line.strip() for line in """
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_bpr198
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_bpr200
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_bpr202
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_bpr208
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_bpr210
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_bpr212
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_bpr215
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ep2
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ep3
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gauss010
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gauss015
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gauss020
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo005
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo008
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo010
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo012
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo015
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hop010
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hop020
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hop030
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hpop005
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hpop008
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hpop010
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hpop012
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hpop015
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hpop10_pop08
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_jump008
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_jump009
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_jump011
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_jump012
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_jump013
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_jump015
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pmi003
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pmi004
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pmi006
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pmi007
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pmi008
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pmi010
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pmi012
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pop005
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pop008
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pop010
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pop012
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rev005
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rev008
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rev010
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rev012
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rev015
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rev10_geo10
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_seed42
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_seed43
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_seed44
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_seed45
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_seed47
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_seed47_soft080
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_seed48
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_seed50
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_seed50_pmi08
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_seed52
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_seed55
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_soft070
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_soft074
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_soft076
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_soft076_bpr208
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_soft080
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_soft080_pmi06
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_soft082
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_soft085
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_soft090
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_soft095
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_trgate
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_trgate_geo10
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_trgate_rev10
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_wd001
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_wd005
climb_bpr020_soft010_w17_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_wd010
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
    cfg_path = ROOT / "configs" / CITY / f"{tag}.yaml"
    if not cfg_path.is_file() and "_w17_" in tag:
        ensure_wave_configs("w17", CITY)
    if not cfg_path.is_file():
        print(f"SKIP missing local config {cfg_path}", flush=True)
        return -1
    cfg = str(cfg_path.relative_to(ROOT))
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
    n_ensured = ensure_wave_configs("w17", CITY)
    print(
        f"{CITY} climb packer queue={len(QUEUE)} ensured_local_w17={n_ensured} "
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
                        pid = launch(tag, g, lim)
                        if pid < 0:
                            continue
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
