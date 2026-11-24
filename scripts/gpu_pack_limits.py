"""Per-GPU launch caps for multi-city climb packers.

Rules (prefs 2026-11-24; raised later same day):
- Default: up to ``TSMOE_MAX_PER_GPU`` (default **6**) of *our* ``tsmoe`` jobs per GPU.
- If another user has a compute process on that GPU — except users listed in
  ``TSMOE_IGNORE_FOREIGN_USERS`` (default **dl-box**) — cap drops to
  ``TSMOE_MAX_PER_GPU_SHARED`` (default **5**; raised from 3 to fill free VRAM).
- Caps only gate **new** launches. Never kill / stop / signal running jobs
  (ours or foreign) when a GPU is already over the limit — let them finish.
"""
from __future__ import annotations

import getpass
import os
import pwd
import subprocess
from collections import defaultdict


OUR_USER = os.environ.get("TSMOE_OUR_USER") or os.environ.get("USER") or getpass.getuser()
IGNORE_FOREIGN_USERS = {
    u.strip()
    for u in os.environ.get("TSMOE_IGNORE_FOREIGN_USERS", "dl-box").split(",")
    if u.strip()
}
MAX_OUR_PER_GPU_FREE = int(os.environ.get("TSMOE_MAX_PER_GPU", "6"))
# Default 5: camob ~17GiB/GPU leaves ~28GiB free; user 2026-11-24 asked to fill
# unused capacity while shared. Override via TSMOE_MAX_PER_GPU_SHARED.
MAX_OUR_PER_GPU_SHARED = int(os.environ.get("TSMOE_MAX_PER_GPU_SHARED", "5"))


def proc_user(pid: int) -> str | None:
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
        return pwd.getpwuid(uid).pw_name
    except Exception:
        return None


def uuid_map() -> dict[str, int]:
    uu: dict[str, int] = {}
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
    ).strip().splitlines():
        i, u = [x.strip() for x in line.split(",")]
        uu[u] = int(i)
    return uu


def scan_gpus(city: str):
    """Return (by_our_all, by_our_city, live_cmds, limit_by_gpu, foreign_by_gpu)."""
    uu = uuid_map()
    by_all: dict[int, int] = defaultdict(int)
    by_city: dict[int, int] = defaultdict(int)
    foreign: dict[int, bool] = defaultdict(bool)
    live: set[str] = set()

    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if out:
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 2:
                continue
            g = uu.get(parts[0])
            if g is None:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            user = proc_user(pid)
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(
                    errors="ignore"
                )
            except Exception:
                cmd = ""

            # Foreign contention: anyone except us and ignored users (dl-box).
            if user and user != OUR_USER and user not in IGNORE_FOREIGN_USERS:
                foreign[g] = True

            if "tsmoe" not in cmd:
                continue
            # Only count *our* tsmoe jobs toward the cap.
            if user and user != OUR_USER:
                continue
            by_all[g] += 1
            if f"configs/{city}/" in cmd or f"runs/{city}/" in cmd:
                by_city[g] += 1
            live.add(cmd)

    limits: dict[int, int] = {}
    for g in range(4):
        limits[g] = (
            MAX_OUR_PER_GPU_SHARED if foreign[g] else MAX_OUR_PER_GPU_FREE
        )
    return by_all, by_city, live, limits, dict(foreign)


def peer_climb_packers_alive(self_city: str) -> bool:
    """True if another ``*_climb_pack.py`` process is running (not this city)."""
    try:
        out = subprocess.check_output(["ps", "-eo", "args="], text=True, errors="ignore")
    except Exception:
        return False
    needle_self = f"{self_city}_climb_pack.py"
    for line in out.splitlines():
        if "_climb_pack.py" not in line:
            continue
        if needle_self in line:
            continue
        # ignore editors / grep noise
        if "python" not in line:
            continue
        return True
    return False


def city_cap(
    gpu_limit: int,
    other_city_on_gpu: int = 0,
    peer_packers: bool = False,
) -> int:
    """Per-city share so multi-city packers can run in parallel under the GPU cap.

    About half of the per-GPU our-job limit (ceil so odd limits are usable;
    at least 1). Full ``gpu_limit`` only when this packer is alone *and* no
    peer climb packer is alive — otherwise a deep queue monopolizes forever
    and newer cities never get a foothold. Total our jobs still enforced via
    all_counts; never kills over-limit runners.
    """
    if other_city_on_gpu <= 0 and not peer_packers:
        return max(1, gpu_limit)
    return max(1, (gpu_limit + 1) // 2)
