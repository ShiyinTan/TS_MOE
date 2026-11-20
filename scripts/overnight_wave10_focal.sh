#!/bin/bash
# Ladder D step 1: focal CE γ∈{1,1.5,2,0.5} on locked peak (revisit0.10 + geo0.15).
# One loss lever only — do not pair with CE-τ / BPR-max / soft-target.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave10.launch.txt

PEAK=(
  --cache "$CACHE"
  --seed 42 --device cuda
  --router-mode hard_prior --dim 256 --heads 8
  --lambda-bpr 0.29
  --transition-score-bias 2.0
  --surd-bias-gate unique_p --tsb-gate-mode residual --surd-score-bias 0.26
  --candidate-context on --candidate-context-time last_slot
  --transition-attn gate
  --geo-score-bias 0.15
  --revisit-score-bias 0.10
  --epochs 1 --checkpoint-metric acc1
)

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave10.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave10.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

run_one 0 on_rev_foc10 --ce-focal-gamma 1.0
run_one 1 on_rev_foc15 --ce-focal-gamma 1.5
run_one 2 on_rev_foc20 --ce-focal-gamma 2.0
run_one 3 on_rev_foc05 --ce-focal-gamma 0.5
cat logs/overnight/wave10.launch.txt
