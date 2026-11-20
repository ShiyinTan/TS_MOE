#!/bin/bash
# Overnight Wave-4: gauss geo (replace flat geo) + adaptive geo λ on tr_gate.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave4.launch.txt

BASE=(
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
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave4.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${BASE[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave4.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

# Replace geo0.15 with gauss; plus adaptive flat geo at λ=0.25 (safe when sparse).
run_one 0 on_gauss05 --gauss-geo-bias 0.20 --gauss-sigma-km 0.5
run_one 1 on_gauss10 --gauss-geo-bias 0.20 --gauss-sigma-km 1.0
run_one 2 on_adapt025 --geo-score-bias 0.25 --geo-adapt sparse --geo-adapt-temp 1.0
run_one 3 on_adapt015 --geo-score-bias 0.15 --geo-adapt sparse --geo-adapt-temp 1.0
cat logs/overnight/wave4.launch.txt
