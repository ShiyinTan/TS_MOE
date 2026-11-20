#!/bin/bash
# Wave-4b (ideas v4 Ladder A): spatial replace geo on LOCKED peak stack
# = WIN + tr_gate + revisit0.10 + {gauss | adaptive}  (no flat geo when gauss).
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave4b.launch.txt

PEAK=(
  --cache "$CACHE"
  --seed 42 --device cuda
  --router-mode hard_prior --dim 256 --heads 8
  --lambda-bpr 0.29
  --transition-score-bias 2.0
  --surd-bias-gate unique_p --tsb-gate-mode residual --surd-score-bias 0.26
  --candidate-context on --candidate-context-time last_slot
  --transition-attn gate
  --revisit-score-bias 0.10
  --epochs 1 --checkpoint-metric acc1
)

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave4b.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave4b.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

# Replace geo0.15 with gauss; adaptive keeps geo λ but sparse-scaled; all keep revisit0.10.
run_one 0 on_rev_gauss05 --gauss-geo-bias 0.20 --gauss-sigma-km 0.5
run_one 1 on_rev_gauss10 --gauss-geo-bias 0.20 --gauss-sigma-km 1.0
run_one 2 on_rev_adapt025 --geo-score-bias 0.25 --geo-adapt sparse --geo-adapt-temp 1.0
run_one 3 on_rev_adapt015 --geo-score-bias 0.15 --geo-adapt sparse --geo-adapt-temp 1.0
cat logs/overnight/wave4b.launch.txt
