#!/bin/bash
# Ideas v9 #43: micro retune ssb/tsb under soft10r (one factor at a time).
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave20.launch.txt

PEAK=(
  --cache "$CACHE"
  --seed 42 --device cuda
  --router-mode hard_prior --dim 256 --heads 8
  --lambda-bpr 0.29
  --surd-bias-gate unique_p --tsb-gate-mode residual
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
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave20.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave20.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

# Peak is ssb=0.26 tsb=2.0 — micro offsets only
run_one 0 on_s10r_ssb22 --surd-score-bias 0.22 --transition-score-bias 2.0
run_one 1 on_s10r_ssb30 --surd-score-bias 0.30 --transition-score-bias 2.0
run_one 2 on_s10r_tsb18 --surd-score-bias 0.26 --transition-score-bias 1.8
run_one 3 on_s10r_tsb22 --surd-score-bias 0.26 --transition-score-bias 2.2
cat logs/overnight/wave20.launch.txt
