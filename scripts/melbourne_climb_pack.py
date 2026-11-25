#!/usr/bin/env python3
"""Melbourne Acc climb packer (parallel cities; dynamic GPU caps).

Policy: after a city beats STHGCN, start the next city but keep optimizing the
winner. Per-GPU our-job caps via ``gpu_pack_limits``:
  - **2** when GPU has no foreign users (``dl-box`` ignored)
  - **2** when another user (e.g. camob) is on that GPU
Total our jobs per GPU are capped; cities share via ``all_counts``.
Over-limit only blocks new launches — never kills already-running jobs.

Never touches foreign processes. Prefer configs under configs/melbourne/climb_*.yaml.
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
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr200_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr205_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr208_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr210_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr212_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr215_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr218_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr220_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump085
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump090
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump095
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump098
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump100
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump102
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump105
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump110
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump115
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed44_bpr021_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed45_bpr021_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed47_bpr021_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed48_bpr021_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed50_bpr021_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed52_bpr021_jump010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_pmi08
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_pmi10
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_pmi12
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_pmi14
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_pmi15
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_pmi16
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_pmi18
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_pop05
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_pop08
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_pop10
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_pop12
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_pop15
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_wd005
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_wd008
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_jump010_wd010
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_j08p10_pmi12
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_j08p10_pmi15
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_j08p10_pmi18
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_j09p10_pmi12
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_j09p10_pmi15
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_j09p10_pmi18
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_j10p10_pmi12
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_j10p10_pmi15
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr021_j10p10_pmi18
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j07p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j10p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p10_pmi11
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p10_pmi11
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p10_pmi12
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p10_pmi12
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p10_pmi13
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p10_pmi13
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p10_pmi14
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p10_pmi14
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p10_pmi15
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p10_pmi15
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p10_pmi16
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p10_pmi16
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p10_pmi17
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p10_pmi17
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p10_pmi18
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p10_pmi18
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p06_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p06_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p08_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p08_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p09_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p09_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p11_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p11_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p12_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p12_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j08p14_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p14_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed44_j09p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed44_bpr021_j09p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed45_j09p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed45_bpr021_j09p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed47_j09p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed47_bpr021_j09p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed48_j09p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_seed48_bpr021_j09p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr20_j09p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr21_j09p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr22_j09p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_bpr23_j09p10_pmi015
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p10_pmi015_wd005
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p10_pmi015_wd008
climb_bpr020_soft010_w11_tgt_seed46_soft0078_session_j09p10_pmi015_wd010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr195_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr198_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr200_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr202_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr203_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr204_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr206_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr207_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr208_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr210_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr212_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump080
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump085
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump090
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump092
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump095
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump098
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump100
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump102
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump105
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump108
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump110
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump115
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_seed42_bpr205_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_seed43_bpr205_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_seed44_bpr205_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_seed45_bpr205_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_seed47_bpr205_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_seed48_bpr205_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_seed49_bpr205_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_seed50_bpr205_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_seed52_bpr205_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_seed55_bpr205_jump010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_pmi05
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_pmi08
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_pmi10
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_pmi12
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_pmi15
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_pmi18
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_pop05
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_pop08
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_pop10
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_pop12
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_pop15
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_wd003
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_wd005
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_wd008
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_wd010
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_j08p10_pmi10
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_j08p10_pmi12
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_j08p10_pmi15
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_j09p10_pmi10
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_j09p10_pmi12
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_j09p10_pmi15
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_j10p10_pmi10
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_j10p10_pmi12
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_j10p10_pmi15
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_ep2
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_ep3
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_jump010_prior_ema
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_j10p08_pmi15
climb_bpr020_soft010_w12_tgt_seed46_soft0078_session_bpr205_j10p12_pmi15
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump010_pmi02
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump010_pmi025
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump010_pmi03
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump010_pmi035
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump010_pmi04
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump010_pmi045
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump010_pmi055
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump010_pmi06
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump010_pmi065
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump010_pmi07
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr195_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr198_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr200_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr202_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr203_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr204_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr206_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr207_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr208_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr210_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr212_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump080_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump085_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump090_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump092_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump095_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump098_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump102_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump105_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump108_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump110_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump115_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr200_jump010_pmi03
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr200_jump010_pmi04
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr200_jump010_pmi06
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr210_jump010_pmi03
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr210_jump010_pmi04
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr210_jump010_pmi06
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump090_pmi03
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump090_pmi04
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump090_pmi06
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump100_pmi03
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump100_pmi04
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump100_pmi06
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump110_pmi03
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump110_pmi04
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump110_pmi06
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_seed44_bpr205_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_seed45_bpr205_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_seed47_bpr205_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_seed48_bpr205_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_seed49_bpr205_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_seed50_bpr205_jump010_pmi05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pop03
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pop05
climb_bpr020_soft010_w12b_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_pop08
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo005
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo012
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo018
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo020
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo025
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg005
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg015
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg020
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg025
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gsig04
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gsig05
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gsig06
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gsig075
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rev005
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rev008
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rev012
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rev015
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rev020
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hh005
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hh008
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hh010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hh015
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrec005
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrec008
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrec010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrec015
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrec020
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hour005
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hour008
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hour010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hour012
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hour015
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ssb020
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ssb022
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ssb024
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ssb028
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ssb030
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_tsb15
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_tsb18
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_tsb22
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_tsb25
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_exg005
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_exg008
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_exg010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_exr005
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_exr008
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_exr010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_prior_ema
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_prior_tsb
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ep2
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ep3
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo018_rev012
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo015_rev010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo020_rev015
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg020_rev010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg015_rev012
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hour010_rev010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hour008_hh010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hh010_rrec010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo015_hour010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg020_hour010
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ssb028_rev012
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_geo018_ep2
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg020_ep2
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rev012_ep2
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hour010_ep2
climb_bpr020_soft010_w13_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_userpref
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_retgate_exg005
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_retgate_exg008
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_retgate_exg010
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_retgate_exg015
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_retgate_exr008
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_retgate_exr012
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_famret010
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_famret015
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_famret020
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_famret025
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_famex010
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_famex015
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_famex020
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_famrad05
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_famrad10
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_famrad15
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_famrad20
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_famexarea010
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_cstage005
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_cstage010
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_cstage015
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_cstage020
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_cstage025
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_catsc005
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_catsc010
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_catsc015
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_catsc020
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_catsc025
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_wd001
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_wd002
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_wd003
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_wd004
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_wd005
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_wd006
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_wd008
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ema90
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ema95
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ema99
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ema995
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau6
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau12
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau24
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau48
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrec015_tau6
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrec015_tau12
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrec015_tau24
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hhk3
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hhk4
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hhk7
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_hhk10
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_temp08
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_temp09
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_temp11
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_temp12
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_drop05
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_drop15
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_drop20
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gsig070
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gsig080
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gsig085
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gsig090
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg018_gsig075
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg022_gsig075
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg028_gsig075
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_retgate_famret015
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_cstage015_catsc010
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_wd003_cstage015
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_wd004_famret020
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau12_hhk7
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gsig075_wd003
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gsig075_cstage010
climb_bpr020_soft010_w14_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_retgate_wd003
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_trgate
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_trbias
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_troff
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_cand_last
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_cand_causal
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_cand_off
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_cand_ctx_off
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_surd_unique
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_surd_conf
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_surd_umass
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_surd_smip
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_surd_smipc
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_surd_off
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_tsbg_res
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_tsbg_mul
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_tsbg_off
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_xattn_hyb
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_xattn_full
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_xattn_hyb_sparse
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_xattn_hyb_dense
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_userid
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau8_hhk5
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau10_hhk5
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau12_hhk5
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau12_hhk3
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau12_hhk7
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau12_hhk10
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau16_hhk7
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau20_hhk7
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg016_gsig075
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg018_gsig070
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg018_gsig075
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg018_gsig080
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg020_gsig075
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg022_gsig075
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ema90_s44
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ema90_s45
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ema90_s47
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ema90_s48
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_trgate_cand_causal
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_trgate_surd_conf
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_trgate_tsbg_mul
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_xattn_hyb_userid
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_xattn_hyb_cand_causal
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_surd_conf_tsbg_mul
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_surd_smip_tsbg_res
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_cand_causal_userid
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg018_gsig075_trgate
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_gg018_gsig075_xattn
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau12_hhk7_trgate
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_rrtau12_hhk7_xattn
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ema90_trgate
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_ema90_xattn
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_userid_trgate
climb_bpr020_soft010_w15_tgt_seed46_soft0078_session_bpr205_jump010_pmi05_xattn_full_cand_last
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
