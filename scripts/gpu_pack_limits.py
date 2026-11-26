"""Per-GPU launch caps for multi-city climb packers.

Rules (prefs 2026-11-25; raised 2026-11-25 evening; large-city 2026-11-26):
- Default: up to ``TSMOE_MAX_PER_GPU`` (default **4**) of *our* ``tsmoe`` jobs per GPU
  when the GPU is free of foreign users (ignore ``dl-box``).
- If another user (e.g. camob) is on that GPU, cap is
  ``TSMOE_MAX_PER_GPU_SHARED`` (default **3**).
- Bandung-or-larger datasets: ``TSMOE_MAX_PER_GPU_LARGE`` (default **2**) per GPU
  (see ``LARGE_CITIES``).
- Caps only gate **new** launches. Never kill running jobs (ours or foreign).
- Also gate on host RAM, per-GPU free VRAM, and disk — these matter more than the
  integer cap when cards have large VRAM headroom.
- Packers should keep filling the box autonomously; agents harvest sparsely
  (token thrift) and only wake on Acc lift / real resource blocker.
"""
from __future__ import annotations

import getpass
import os
import pwd
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


OUR_USER = os.environ.get("TSMOE_OUR_USER") or os.environ.get("USER") or getpass.getuser()
IGNORE_FOREIGN_USERS = {
    u.strip()
    for u in os.environ.get("TSMOE_IGNORE_FOREIGN_USERS", "dl-box").split(",")
    if u.strip()
}
# Prefs: pack more when GPUs are free; VRAM/RAM/disk gates still apply.
MAX_OUR_PER_GPU_FREE = int(os.environ.get("TSMOE_MAX_PER_GPU", "4"))
MAX_OUR_PER_GPU_SHARED = int(os.environ.get("TSMOE_MAX_PER_GPU_SHARED", "3"))
# Bandung-or-larger Massive-STEPS cities (raw data/ CSV ≥ Bandung ≈57MB): tighter
# per-GPU cap to avoid host RAM / VRAM pressure. Override with TSMOE_MAX_PER_GPU_LARGE.
# Raised to 4 (2026-11-27): pack prior-city + f63 densify when VRAM allows;
# host RAM / free-VRAM gates still apply. Override with TSMOE_MAX_PER_GPU_LARGE.
MAX_OUR_PER_GPU_LARGE = int(os.environ.get("TSMOE_MAX_PER_GPU_LARGE", "4"))
try:
    # Single source of truth with train-time large_city_fast gating.
    from tsmoe.city_scale import LARGE_CITIES as _LARGE_CITIES  # type: ignore

    LARGE_CITIES = set(_LARGE_CITIES)
except Exception:
    LARGE_CITIES = {
        # ≥ Bandung by data/ CSV footprint on the GPU box (2026-11-26).
        "bandung",
        "kuwait_city",
        "sao_paulo",
        "new_york",
        "jakarta",
        "petaling_jaya",
        "istanbul",
    }
# Leave headroom so a new train does not OOM the host / GPU.
MIN_HOST_RAM_GIB = float(os.environ.get("TSMOE_MIN_HOST_RAM_GIB", "24"))
MIN_FREE_VRAM_MIB = int(os.environ.get("TSMOE_MIN_FREE_MIB", "5000"))
MIN_DISK_FREE_GIB = float(os.environ.get("TSMOE_MIN_DISK_FREE_GIB", "8"))


def max_our_per_gpu(city: str, *, shared: bool = False) -> int:
    """Per-GPU our-job launch cap for ``city`` (never kills already-running jobs)."""
    if city in LARGE_CITIES:
        return max(1, MAX_OUR_PER_GPU_LARGE)
    return MAX_OUR_PER_GPU_SHARED if shared else MAX_OUR_PER_GPU_FREE


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
        limits[g] = max_our_per_gpu(city, shared=bool(foreign[g]))
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


def host_ram_available_gib() -> float | None:
    try:
        meminfo = Path("/proc/meminfo").read_text()
    except Exception:
        return None
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    return None


def host_ram_ok(min_gib: float | None = None) -> bool:
    """True when host MemAvailable is above the launch floor."""
    floor = MIN_HOST_RAM_GIB if min_gib is None else float(min_gib)
    avail = host_ram_available_gib()
    if avail is None:
        return True
    return avail >= floor


def disk_free_gib(path: str | Path) -> float | None:
    try:
        usage = shutil.disk_usage(str(path))
    except Exception:
        return None
    return usage.free / (1024 ** 3)


def disk_ok(path: str | Path, min_gib: float | None = None) -> bool:
    """True when ``path``'s filesystem has enough free space for a new train."""
    floor = MIN_DISK_FREE_GIB if min_gib is None else float(min_gib)
    free = disk_free_gib(path)
    if free is None:
        return True
    return free >= floor


def gpu_free_mib() -> dict[int, int]:
    free: dict[int, int] = {}
    for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines():
        i, m = [x.strip() for x in line.split(",")]
        free[int(i)] = int(float(m))
    return free


def resource_snapshot(runs_path: str | Path | None = None) -> str:
    """One-line host RAM / VRAM / disk snapshot for packer logs and reports."""
    ram = host_ram_available_gib()
    ram_s = f"{ram:.1f}GiB" if ram is not None else "?"
    try:
        free = gpu_free_mib()
        vram_s = ",".join(str(free.get(g, 0)) for g in range(4))
    except Exception:
        vram_s = "?"
    disk_s = "?"
    if runs_path is not None:
        d = disk_free_gib(runs_path)
        if d is not None:
            disk_s = f"{d:.1f}GiB"
    return (
        f"ram_avail={ram_s} (floor={MIN_HOST_RAM_GIB:g}GiB) "
        f"vram_free_mib=[{vram_s}] (floor={MIN_FREE_VRAM_MIB}) "
        f"disk_free={disk_s} (floor={MIN_DISK_FREE_GIB:g}GiB)"
    )
