#!/bin/bash
# TM-35 Round-2: weaker geo-score-bias on LOCKED WIN (Acc≥WIN + keep R@k lift).
# gsb=0.5 Acc 0.258 / R5·R20↑ → try 0.15 / 0.20 / 0.25
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/tm35
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/tm35/round2.launch.txt

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
  echo "[$(date -Iseconds)] GPU${gpu} ${name}" | tee -a logs/tm35/round2.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${WIN_FLAGS[@]}" "$@" \
    >"logs/tm35/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/tm35/round2.launch.txt
  echo $! > "logs/tm35/${name}.pid"
}

run_one 0 hc_win_gsb015 --geo-score-bias 0.15
run_one 1 hc_win_gsb020 --geo-score-bias 0.20
run_one 2 hc_win_gsb025 --geo-score-bias 0.25
cat logs/tm35/round2.launch.txt
