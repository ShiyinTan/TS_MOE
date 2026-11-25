#!/usr/bin/env python3
"""One-line MEL Acc harvest for sparse agent wakes (token thrift).

Prefer reading the GPU-local autonomy snapshot when present::

    python scripts/mel_acc_oneline.py

Falls back to scanning results/ if state is missing. Does **not** launch trains.
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "scripts" / "data" / "mel_climb_policy.json"
STATE = ROOT / "runs" / "climb_state" / "melbourne.json"
HARVEST = ROOT / "runs" / "climb_state" / "harvest_latest.json"


def _pack_jobs() -> tuple[str, str]:
    pack = subprocess.getoutput(
        "pgrep -af melbourne_climb_pack.py | grep -v pgrep || echo DEAD"
    )
    trains = subprocess.getoutput(
        'pgrep -af "tsmoe train" | grep -v pgrep | wc -l'
    ).strip()
    pack_s = "alive" if "DEAD" not in pack else "DEAD"
    return pack_s, trains


def _ram_disk() -> tuple[float, str]:
    avail = 0.0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) / (1024 * 1024)
            break
    disk = subprocess.getoutput(
        f"df -BG {ROOT} | awk 'NR==2{{gsub(\"G\",\"\",$4); print $4}}'"
    ).strip()
    return avail, disk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default=None)
    ap.add_argument("--lock", type=float, default=None)
    ap.add_argument("--total", type=int, default=0)
    ap.add_argument("--city", default="melbourne")
    ap.add_argument("--refresh", action="store_true", help="Run autonomy status first")
    args = ap.parse_args()

    if args.refresh:
        subprocess.call(
            [str(ROOT / "scripts" / "mel_climb_autonomy.py"), "--status"],
            cwd=str(ROOT),
        )

    wave = args.wave
    lock = args.lock
    if STATE.is_file():
        try:
            st = json.loads(STATE.read_text())
            wave = wave or st.get("active_wave")
            if lock is None and st.get("lock_acc") is not None:
                lock = float(st["lock_acc"])
        except Exception:
            pass
    if POLICY.is_file():
        try:
            pol = json.loads(POLICY.read_text())
            wave = wave or pol.get("active_wave")
            if lock is None:
                lock = float(pol.get("lock_acc", 0.170692))
        except Exception:
            pass
    wave = wave or "w17"
    lock = 0.170692 if lock is None else float(lock)

    pack_s, trains = _pack_jobs()
    avail, disk = _ram_disk()

    # Prefer autonomy harvest snapshot (written continuously on the box).
    if HARVEST.is_file():
        try:
            h = json.loads(HARVEST.read_text())
            done = int(h.get("done") or 0)
            total = int(h.get("total") or args.total or 0)
            top = float(h.get("top") or 0.0)
            lifts = int(h.get("lifts") or 0)
            best = h.get("best") or "-"
            wave = h.get("wave") or wave
            lock = float(h.get("lock_acc") or lock)
            event = h.get("event") or "-"
            flag = "LIFT" if lifts else "flat"
            print(
                f"MEL {wave} {done}/{total} TOP={top:.6f} lifts={lifts} {flag} "
                f"jobs={trains} pack={pack_s} availRAM={avail:.0f}G diskFree={disk}G "
                f"best={best} event={event}"
            )
            return
        except Exception:
            pass

    city = args.city
    if args.total > 0:
        total = args.total
    else:
        total = len(list((ROOT / "configs" / city).glob(f"*{wave}*.yaml")))

    files = sorted(
        set(
            glob.glob(
                str(ROOT / f"results/{city}/**/*{wave}*/eval_test/metrics.json"),
                recursive=True,
            )
        )
    )
    done = top = lifts = 0
    best = "-"
    for p in files:
        try:
            a = (json.load(open(p)).get("known_only") or {}).get("Acc@1")
        except Exception:
            continue
        if a is None:
            continue
        done += 1
        if float(a) > top:
            top = float(a)
            best = Path(p).parts[-3]
        if float(a) > lock:
            lifts += 1

    flag = "LIFT" if lifts else "flat"
    print(
        f"MEL {wave} {done}/{total} TOP={top:.6f} lifts={lifts} {flag} "
        f"jobs={trains} pack={pack_s} availRAM={avail:.0f}G diskFree={disk}G "
        f"best={best}"
    )


if __name__ == "__main__":
    main()
