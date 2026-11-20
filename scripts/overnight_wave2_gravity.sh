#!/bin/bash
# Overnight Wave-2: gravity geo power α on peak λ=0.15 (+ one soft α with λ=0.12).
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave2.launch.txt

PEAK=(
  --cache "$CACHE"
  --seed 42 --device cuda
  --router-mode hard_prior --dim 256 --heads 8
  --lambda-bpr 0.29
  --transition-score-bias 2.0
  --surd-bias-gate unique_p --tsb-gate-mode residual --surd-score-bias 0.26
  --candidate-context on --candidate-context-time last_slot
  --transition-attn gate
  --epochs 1 --checkpoint-metric acc1
)

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave2.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave2.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

# α≠1 only matters when geo-score-bias>0; α=1 ≡ peak (skip).
run_one 0 on_gsb015_p05 --geo-score-bias 0.15 --geo-score-power 0.5
run_one 1 on_gsb015_p15 --geo-score-bias 0.15 --geo-score-power 1.5
run_one 2 on_gsb015_p20 --geo-score-bias 0.15 --geo-score-power 2.0
run_one 3 on_gsb012_p05 --geo-score-bias 0.12 --geo-score-power 0.5
cat logs/overnight/wave2.launch.txt
