#!/usr/bin/env python3
"""GPU-local handoff: when a city hits target, start the next city's packer.

    python scripts/city_climb_handoff.py --from melbourne
    python scripts/city_climb_handoff.py --status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER_PATH = ROOT / "scripts" / "data" / "city_climb_order.json"
STATE_DIR = ROOT / "runs" / "climb_state"
PY = os.environ.get("TSMOE_PYTHON", "/home/shiyintan/miniconda3/envs/tsmoe/bin/python")


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_order() -> dict:
    return json.loads(ORDER_PATH.read_text())


def save_order(data: dict) -> None:
    ORDER_PATH.write_text(json.dumps(data, indent=2) + "\n")


def mark_city(data: dict, city: str, status: str, **extra) -> None:
    for c in data.get("cities") or []:
        if c.get("city") == city:
            c["status"] = status
            c.update(extra)
            break


def next_queued(data: dict) -> dict | None:
    for c in data.get("cities") or []:
        if c.get("status") == "QUEUED" and c.get("packer"):
            return c
    for c in data.get("cities") or []:
        if c.get("status") == "QUEUED":
            return c
    return None


def packer_alive(packer_rel: str) -> bool:
    name = Path(packer_rel).name
    out = subprocess.getoutput(f"pgrep -af {name} | grep -v pgrep || true")
    return bool(out.strip())


def start_packer(packer_rel: str) -> int:
    log = ROOT / "runs" / "packer_logs"
    log.mkdir(parents=True, exist_ok=True)
    out_path = log / f"{Path(packer_rel).stem}.out"
    env = os.environ.copy()
    proc = subprocess.Popen(
        [PY, "-u", str(ROOT / packer_rel)],
        cwd=str(ROOT),
        stdout=open(out_path, "ab"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    (STATE_DIR / "active_packer.pid").write_text(str(proc.pid))
    return proc.pid


def handoff(from_city: str | None = None) -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_order()
    if from_city:
        mark_city(data, from_city, "DONE", finished_at=_now())
    nxt = next_queued(data)
    if not nxt:
        data["active_city"] = None
        save_order(data)
        return {"event": "no_next_city", "from": from_city}
    packer = nxt.get("packer")
    if not packer:
        return {
            "event": "next_needs_packer",
            "from": from_city,
            "next": nxt["city"],
            "sthgcn": nxt.get("sthgcn"),
        }
    if packer_alive(packer):
        data["active_city"] = nxt["city"]
        mark_city(data, nxt["city"], "ACTIVE")
        save_order(data)
        return {"event": "already_running", "city": nxt["city"], "packer": packer}

    # Ensure bandung wave configs if needed
    if nxt["city"] == "bandung":
        subprocess.check_call([PY, str(ROOT / "scripts" / "gen_bdg_wave_local.py"), "--wave", "w1"], cwd=ROOT)

    pid = start_packer(packer)
    data["active_city"] = nxt["city"]
    mark_city(data, nxt["city"], "ACTIVE", started_at=_now(), packer_pid=pid)
    save_order(data)
    snap = {
        "event": "handoff_started",
        "from": from_city,
        "city": nxt["city"],
        "sthgcn": nxt.get("sthgcn"),
        "target_acc": nxt.get("target_acc"),
        "packer": packer,
        "pid": pid,
        "written_at": _now(),
    }
    (STATE_DIR / "handoff_latest.json").write_text(json.dumps(snap, indent=2) + "\n")
    return snap


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="from_city", default=None)
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        print(json.dumps(load_order(), indent=2))
        return
    print(json.dumps(handoff(args.from_city), indent=2))


if __name__ == "__main__":
    main()
