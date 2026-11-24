#!/usr/bin/env python3
"""Beijing Acc climb packer (parallel cities; dynamic GPU caps).

Policy: after a city beats STHGCN, start the next city but keep optimizing the
winner. Per-GPU our-job caps via ``gpu_pack_limits``:
  - 4 when GPU has no foreign users (``dl-box`` ignored)
  - 4 when another user (e.g. camob) is on that GPU
Total our jobs per GPU are capped; cities share via ``all_counts``.
Over-limit only blocks new launches — never kills already-running jobs.

Never touches foreign processes. Prefer configs under configs/beijing/climb_*.yaml.
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

CITY = "beijing"
PY = os.environ.get("TSMOE_PYTHON", "/home/shiyintan/miniconda3/envs/tsmoe/bin/python")
MIN_FREE_MIB = int(os.environ.get("TSMOE_MIN_FREE_MIB", "2500"))
LOG = ROOT / "runs" / "packer_logs"
PIDDIR = ROOT / "logs" / "multicity"
STOP_SENTINEL = Path(
    os.environ.get("TSMOE_STOP_SENTINEL", str(ROOT / "logs" / "multicity" / "STOP"))
)
PACK_LOCK = PIDDIR / "packer_launch.lock"

QUEUE = [line.strip() for line in """
climb_s5g20_all_gauss020_wd005_w34_tgt_soft0034_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_soft0039_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_soft0043_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_soft0049_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_soft0051_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_soft0059_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_bpr0095_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_bpr0115_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_bpr0145_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_bpr0195_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_bpr0225_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_bpr0265_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_explore007_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_explore009_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_explore011_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_explore0125_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_tsb173_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_tsb187_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_tsb203_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_tsb218_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_seed38_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_seed62_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_seed63_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_seed64_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_soft0049_session_ep3
climb_s5g20_all_gauss020_wd005_w34_tgt_soft0051_session_ep3
climb_s5g20_all_gauss020_wd005_w34_abl_soft0049_ep3
climb_s5g20_all_gauss020_wd005_w34_abl_soft0051_ep3
climb_s5g20_all_gauss020_wd005_w34_abl_bpr225_ep3
climb_s5g20_all_gauss020_wd005_w34_abl_explore09_ep3
climb_s5g20_all_gauss020_wd005_w34_filt_soft0049_ep3
climb_s5g20_all_gauss020_wd005_w34_filt_soft0051_ep3
climb_s5g20_all_gauss020_wd005_w34_filt_bpr225_ep3
climb_s5g20_all_gauss020_wd005_w34_filt_explore09_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_soft0033_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_soft0037_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_soft0041_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_soft0047_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_soft0053_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_soft0057_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_bpr0105_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_bpr0135_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_bpr0165_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_bpr0185_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_bpr0215_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_bpr0245_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_explore0065_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_explore008_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_explore01_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_explore012_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_tsb17_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_tsb182_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_tsb198_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_tsb212_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_seed39_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_seed59_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_seed60_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_seed61_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_soft0047_session_ep3
climb_s5g20_all_gauss020_wd005_w33_tgt_soft0053_session_ep3
climb_s5g20_all_gauss020_wd005_w33_abl_soft0047_ep3
climb_s5g20_all_gauss020_wd005_w33_abl_soft0053_ep3
climb_s5g20_all_gauss020_wd005_w33_abl_bpr215_ep3
climb_s5g20_all_gauss020_wd005_w33_abl_explore08_ep3
climb_s5g20_all_gauss020_wd005_w33_filt_soft0047_ep3
climb_s5g20_all_gauss020_wd005_w33_filt_soft0053_ep3
climb_s5g20_all_gauss020_wd005_w33_filt_bpr215_ep3
climb_s5g20_all_gauss020_wd005_w33_filt_explore08_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_soft0032_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_soft0036_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_soft0044_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_soft0046_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_soft0054_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_soft0062_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_bpr009_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_bpr0125_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_bpr0155_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_bpr0175_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_bpr0205_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_bpr0255_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_bpr029_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_explore0055_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_explore0075_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_explore0095_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_explore0115_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_explore0135_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_tsb168_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_tsb178_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_tsb192_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_tsb208_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_tsb225_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_seed40_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_seed54_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_seed56_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_seed57_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_seed58_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_wd0002_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_wd0007_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_wd001_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_soft0044_session_ep3
climb_s5g20_all_gauss020_wd005_w32_tgt_soft0054_session_ep3
climb_s5g20_all_gauss020_wd005_w32_abl_soft0044_ep3
climb_s5g20_all_gauss020_wd005_w32_abl_soft0054_ep3
climb_s5g20_all_gauss020_wd005_w32_abl_bpr205_ep3
climb_s5g20_all_gauss020_wd005_w32_abl_explore095_ep3
climb_s5g20_all_gauss020_wd005_w32_abl_session_ep3
climb_s5g20_all_gauss020_wd005_w32_filt_soft0044_ep3
climb_s5g20_all_gauss020_wd005_w32_filt_soft0054_ep3
climb_s5g20_all_gauss020_wd005_w32_filt_bpr205_ep3
climb_s5g20_all_gauss020_wd005_w32_filt_explore095_ep3
climb_s5g20_all_gauss020_wd005_w32_filt_session_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_soft0038_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_soft0042_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_soft0048_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_soft0052_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_soft0058_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_bpr011_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_bpr013_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_bpr017_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_bpr019_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_bpr023_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_bpr027_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_explore007_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_explore0085_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_explore0105_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_explore013_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_tsb165_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_tsb172_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_tsb188_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_tsb215_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_seed41_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_seed49_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_seed51_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_seed52_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_seed53_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_soft0048_session_ep3
climb_s5g20_all_gauss020_wd005_w31_tgt_soft0052_session_ep3
climb_s5g20_all_gauss020_wd005_w31_abl_soft0048_ep3
climb_s5g20_all_gauss020_wd005_w31_abl_soft0052_ep3
climb_s5g20_all_gauss020_wd005_w31_abl_bpr023_ep3
climb_s5g20_all_gauss020_wd005_w31_abl_explore085_ep3
climb_s5g20_all_gauss020_wd005_w31_filt_soft0048_ep3
climb_s5g20_all_gauss020_wd005_w31_filt_soft0052_ep3
climb_s5g20_all_gauss020_wd005_w31_filt_bpr023_ep3
climb_s5g20_all_gauss020_wd005_w31_filt_explore085_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_soft0035_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_soft0045_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_soft0055_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_soft0065_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_soft007_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_bpr01_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_bpr014_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_bpr016_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_bpr02_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_bpr024_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_bpr026_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_bpr03_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_explore006_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_explore009_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_explore011_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_explore014_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_tsb17_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_tsb185_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_tsb195_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_tsb205_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_tsb22_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_ssb01_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_ssb015_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_ssb02_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_ssb025_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_ssb03_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_seed46_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_seed47_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_seed48_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_seed50_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_seed55_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_wd0003_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_wd0004_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_wd0006_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_wd0008_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_soft0045_session_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_soft0055_session_ep3
climb_s5g20_all_gauss020_wd005_w30_tgt_soft0065_session_ep3
climb_s5g20_all_gauss020_wd005_w30_abl_soft0045_ep3
climb_s5g20_all_gauss020_wd005_w30_abl_soft0055_ep3
climb_s5g20_all_gauss020_wd005_w30_abl_bpr02_ep3
climb_s5g20_all_gauss020_wd005_w30_abl_bpr026_ep3
climb_s5g20_all_gauss020_wd005_w30_abl_explore011_ep3
climb_s5g20_all_gauss020_wd005_w30_abl_session_ep3
climb_s5g20_all_gauss020_wd005_w30_filt_soft0045_ep3
climb_s5g20_all_gauss020_wd005_w30_filt_soft0055_ep3
climb_s5g20_all_gauss020_wd005_w30_filt_bpr02_ep3
climb_s5g20_all_gauss020_wd005_w30_filt_bpr026_ep3
climb_s5g20_all_gauss020_wd005_w30_filt_explore011_ep3
climb_s5g20_all_gauss020_wd005_w30_filt_session_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_soft003_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_soft004_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_soft005_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_soft006_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_soft008_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_bpr012_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_bpr015_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_bpr018_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_bpr022_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_bpr028_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_bpr031_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_explore008_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_explore01_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_explore012_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_explore015_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_tsb18_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_tsb19_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_tsb20_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_tsb21_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_seed42_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_seed43_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_seed44_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_seed45_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_soft004_session_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_soft005_session_ep3
climb_s5g20_all_gauss020_wd005_w29_tgt_soft006_session_ep3
climb_s5g20_all_gauss020_wd005_w29_abl_soft004_ep3
climb_s5g20_all_gauss020_wd005_w29_abl_soft005_ep3
climb_s5g20_all_gauss020_wd005_w29_abl_soft006_ep3
climb_s5g20_all_gauss020_wd005_w29_abl_explore010_ep3
climb_s5g20_all_gauss020_wd005_w29_abl_bpr28_ep3
climb_s5g20_all_gauss020_wd005_w29_filt_soft004_ep3
climb_s5g20_all_gauss020_wd005_w29_filt_soft005_ep3
climb_s5g20_all_gauss020_wd005_w29_filt_soft006_ep3
climb_s5g20_all_gauss020_wd005_w29_filt_explore010_ep3
climb_s5g20_all_gauss020_wd005_w29_filt_bpr28_ep3
climb_s5g20_all_gauss020_wd005_w28_allf_seed101
climb_s5g20_all_gauss020_wd005_w28_allf_seed102
climb_s5g20_all_gauss020_wd005_w28_allf_seed103_ep3
climb_s5g20_all_gauss020_wd005_w28_allf_soft048
climb_s5g20_all_gauss020_wd005_w28_allf_soft052
climb_s5g20_all_gauss020_wd005_w28_allf_explore010_ep3
climb_s5g20_all_gauss020_wd005_w28_allf_session_ep3
climb_s5g20_all_gauss020_wd005_w28_allf_hour012_ep3
climb_s5g20_all_gauss020_wd005_w28_filt_session_ep3
climb_s5g20_all_gauss020_wd005_w28_filt_soft050_ep3
climb_s5g20_all_gauss020_wd005_w28_tgt_explore012_ep3
climb_s5g20_all_gauss020_wd005_w28_tgt_session_ep3
climb_s5g20_all_gauss020_wd005_w28_allf_tsb19_ep3
climb_s5g20_all_gauss020_wd005_w28_allf_bpr31_ep3
climb_s5g20_all_gauss020_wd005_w28_p80_soft050_session
climb_s5g20_all_gauss020_wd005_w28_p70_session_ep3
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
