#!/bin/bash
# Ideas v9 #41 top-K KD + #40 geo-laplace on soft10r (revisit score bias locked).
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
TEACHER=runs/tokyo/on_rev_soft10r/best.pt
: > logs/overnight/wave21.launch.txt

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
  --soft-target-beta 0.10
  --soft-target-prior revisit
  --negatives 63
  --epochs 1 --checkpoint-metric acc1
)

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave21.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave21.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

run_one 0 on_s10r_kd05 --kd-teacher "$TEACHER" --lambda-kd 0.5 --kd-top-k 20 --kd-temp 2
run_one 1 on_s10r_kd10 --kd-teacher "$TEACHER" --lambda-kd 1.0 --kd-top-k 20 --kd-temp 2
run_one 2 on_s10r_lap001 --poi-geo-laplace 0.01 --geo-knn 8
run_one 3 on_s10r_kd05_lap --kd-teacher "$TEACHER" --lambda-kd 0.5 --kd-top-k 20 --kd-temp 2 --poi-geo-laplace 0.01 --geo-knn 8
cat logs/overnight/wave21.launch.txt
