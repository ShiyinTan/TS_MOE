"""Sample peak host/GPU memory, GPU util, and wall time during train/eval."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Any


def _read_rss_mib() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # kB
                    kb = float(line.split()[1])
                    return kb / 1024.0
    except Exception:
        return None
    return None


def _visible_gpu_indices() -> list[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() == "":
        return [0]
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            # UUID form — fall back to ordinal 0
            out.append(0)
    return out or [0]


def _nvidia_query(query: str) -> list[list[str]]:
    try:
        text = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return []
    rows = []
    for line in text.strip().splitlines():
        rows.append([c.strip() for c in line.split(",")])
    return rows


def _nvidia_compute_apps() -> list[tuple[int, int, float]]:
    """Return (gpu_index, pid, used_mib) for running compute apps."""
    try:
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return []
    # Map uuid → index
    uuid_rows = _nvidia_query("index,uuid")
    uuid_to_idx = {}
    for row in uuid_rows:
        if len(row) >= 2:
            try:
                uuid_to_idx[row[1]] = int(row[0])
            except ValueError:
                continue
    out: list[tuple[int, int, float]] = []
    for line in text.strip().splitlines():
        parts = [c.strip() for c in line.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid_s, mem_s = parts[0], parts[1], parts[2]
        try:
            pid = int(pid_s)
            mem = float(mem_s)
        except ValueError:
            continue
        idx = uuid_to_idx.get(uuid, -1)
        out.append((idx, pid, mem))
    return out


class ResourceMonitor:
    """Background sampler for peak RSS / GPU mem / GPU util + wall time."""

    def __init__(self, *, interval_s: float = 5.0):
        self.interval_s = max(1.0, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.t0 = 0.0
        self.t1 = 0.0
        self.peak_rss_mib = 0.0
        self.peak_gpu_mem_mib = 0.0  # this process used_gpu_memory
        self.peak_gpu_util_pct = 0.0  # device utilization.gpu
        self.peak_gpu_board_mem_mib = 0.0  # whole-board memory.used
        self.samples = 0
        self.pid = os.getpid()
        self.gpu_indices = _visible_gpu_indices()

    def start(self) -> "ResourceMonitor":
        self.t0 = time.time()
        self._stopped = False
        self._final: dict[str, Any] | None = None
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name="resource-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        if getattr(self, "_stopped", False) and self._final is not None:
            return self._final
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2.0)
            self._thread = None
        self.t1 = time.time()
        self._sample_once()
        self._stopped = True
        self._final = self.snapshot()
        return self._final

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample_once()

    def _sample_once(self) -> None:
        self.samples += 1
        rss = _read_rss_mib()
        if rss is not None:
            self.peak_rss_mib = max(self.peak_rss_mib, rss)

        # Process GPU memory (nvidia-smi compute apps)
        for gpu_idx, pid, mem in _nvidia_compute_apps():
            if pid != self.pid:
                continue
            self.peak_gpu_mem_mib = max(self.peak_gpu_mem_mib, mem)

        # Device util / board memory for physical GPU indices in CUDA_VISIBLE_DEVICES.
        util_rows = _nvidia_query("index,utilization.gpu,memory.used")
        for row in util_rows:
            if len(row) < 3:
                continue
            try:
                idx = int(row[0])
                util = float(row[1])
                board = float(row[2])
            except ValueError:
                continue
            if idx not in self.gpu_indices:
                continue
            self.peak_gpu_util_pct = max(self.peak_gpu_util_pct, util)
            self.peak_gpu_board_mem_mib = max(self.peak_gpu_board_mem_mib, board)

        # torch CUDA allocator peaks (logical device 0 under CUDA_VISIBLE_DEVICES)
        try:
            import torch

            if torch.cuda.is_available():
                alloc = torch.cuda.max_memory_allocated(0) / (1024.0 ** 2)
                reserved = torch.cuda.max_memory_reserved(0) / (1024.0 ** 2)
                self.peak_gpu_mem_mib = max(self.peak_gpu_mem_mib, alloc)
                # stash for snapshot
                self._torch_alloc_mib = max(getattr(self, "_torch_alloc_mib", 0.0), alloc)
                self._torch_reserved_mib = max(getattr(self, "_torch_reserved_mib", 0.0), reserved)
        except Exception:
            pass

    def snapshot(self) -> dict[str, Any]:
        wall = (self.t1 or time.time()) - (self.t0 or time.time())
        return {
            "wall_time_s": round(wall, 3),
            "wall_time_hms": _fmt_hms(wall),
            "peak_rss_mib": round(self.peak_rss_mib, 1),
            "peak_gpu_mem_mib": round(self.peak_gpu_mem_mib, 1),
            "peak_gpu_util_pct": round(self.peak_gpu_util_pct, 1),
            "peak_gpu_board_mem_mib": round(self.peak_gpu_board_mem_mib, 1),
            "torch_peak_alloc_mib": round(getattr(self, "_torch_alloc_mib", 0.0), 1),
            "torch_peak_reserved_mib": round(getattr(self, "_torch_reserved_mib", 0.0), 1),
            "samples": int(self.samples),
            "gpu_indices": list(self.gpu_indices),
            "pid": int(self.pid),
        }


def _fmt_hms(seconds: float) -> str:
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:d}:{sec:02d}"
