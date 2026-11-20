#!/bin/bash
# TM-35 Round-1 wave A: new score-level graph/feature priors on LOCKED WIN.
# Wave B (E1 / session / transition-lambda) → scripts/tm35_round1b_structure.sh
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/tm35
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/tm35/round1.launch.txt

WIN_FLAGS=(
  --cache "$CACHE"
  --seed 42 --device cuda
  --router-mode hard_prior --dim 256 --heads 8
  --lambda-bpr 0.29
  --transition-score-bias 2.0
  --surd-bias-gate unique_p --tsb-gate-mode residual --surd-score-bias 0.26
  --candidate-context on --candidate-context-time last_slot
  --epochs 1 --checkpoint-metric acc1
)

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name}" | tee -a logs/tm35/round1.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${WIN_FLAGS[@]}" "$@" \
    >"logs/tm35/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/tm35/round1.launch.txt
  echo $! > "logs/tm35/${name}.pid"
}

run_one 0 hc_win_csb05 --category-score-bias 0.5
run_one 1 hc_win_psb03 --pop-score-bias 0.3
run_one 2 hc_win_gsb05 --geo-score-bias 0.5
cat logs/tm35/round1.launch.txt
