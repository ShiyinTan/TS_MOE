#!/usr/bin/env python3
"""One-line MEL Acc harvest for sparse agent wakes (token thrift).

Run on the GPU box; prints a single line. Example::

    python scripts/mel_acc_oneline.py --wave w17 --lock 0.170692 --total 76
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="w17")
    ap.add_argument("--lock", type=float, default=0.170692)
    ap.add_argument("--total", type=int, default=0, help="expected configs; 0=count yaml")
    ap.add_argument("--city", default="melbourne")
    args = ap.parse_args()

    city = args.city
    if args.total > 0:
        total = args.total
    else:
        total = len(list((ROOT / "configs" / city).glob(f"*{args.wave}*.yaml")))

    files = sorted(
        set(
            glob.glob(
                str(ROOT / f"results/{city}/**/*{args.wave}*/eval_test/metrics.json"),
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
        if float(a) > args.lock:
            lifts += 1

    pack = subprocess.getoutput(
        f"pgrep -af {city}_climb_pack.py | grep -v pgrep || echo DEAD"
    )
    trains = subprocess.getoutput(
        'pgrep -af "tsmoe train" | grep -v pgrep | wc -l'
    ).strip()
    # MemAvailable GiB
    avail = 0.0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            avail = int(line.split()[1]) / (1024 * 1024)
            break
    disk = subprocess.getoutput(
        f"df -BG {ROOT} | awk 'NR==2{{gsub(\"G\",\"\",$4); print $4}}'"
    ).strip()
    pack_s = "alive" if "DEAD" not in pack else "DEAD"
    flag = "LIFT" if lifts else "flat"
    print(
        f"MEL {args.wave} {done}/{total} TOP={top:.6f} lifts={lifts} {flag} "
        f"jobs={trains} pack={pack_s} availRAM={avail:.0f}G diskFree={disk}G "
        f"best={best}"
    )


if __name__ == "__main__":
    main()
