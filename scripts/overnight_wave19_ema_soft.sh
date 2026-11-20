#!/bin/bash
# Ideas v9 #39: EMA self-distill soft labels on soft10r stack (revisit score bias locked).
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave19.launch.txt

BASE=(
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
  --negatives 63
  --epochs 1 --checkpoint-metric acc1
)

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave19.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${BASE[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave19.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

# Pure EMA soft (replaces revisit soft prior)
run_one 0 on_s10r_ema999 --soft-target-beta 0.10 --soft-target-prior ema --ema-decay 0.999
run_one 1 on_s10r_ema99 --soft-target-beta 0.10 --soft-target-prior ema --ema-decay 0.99
# Blend EMA + locked revisit soft
run_one 2 on_s10r_ema_rev --soft-target-beta 0.10 --soft-target-prior ema,revisit --ema-decay 0.999
# Slightly stronger EMA soft
run_one 3 on_s10r_ema15 --soft-target-beta 0.15 --soft-target-prior ema --ema-decay 0.999
cat logs/overnight/wave19.launch.txt
