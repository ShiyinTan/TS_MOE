#!/bin/bash
# Ideas v7 #33: tsb session-gap gate on locked soft10r peak (Acc 0.267650).
# Keep soft β=0.10 prior=revisit + revisit0.10 + geo0.15. Sweep τ∈{6,12,24}h + soft sigmoid 12h.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave14.launch.txt

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
  --epochs 1 --checkpoint-metric acc1
)

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave14.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave14.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

run_one 0 on_s10r_sgap06 --tsb-session-gap 6
run_one 1 on_s10r_sgap12 --tsb-session-gap 12
run_one 2 on_s10r_sgap24 --tsb-session-gap 24
# If Wave-13 Keep adds hit/recency, restack later; this wave is gap-only on soft10r.
run_one 3 on_s10r_sgap18 --tsb-session-gap 18
cat logs/overnight/wave14.launch.txt
