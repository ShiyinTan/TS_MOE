#!/bin/bash
# Ladder A step 7: soft cat-stage bias on locked peak (revisit0.10 + geo0.15).
# Soft only — no hard top-C filter.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave9.launch.txt

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
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave9.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave9.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

run_one 0 on_rev_cat15 --cat-stage-bias 0.15
run_one 1 on_rev_cat30 --cat-stage-bias 0.30
run_one 2 on_rev_cat45 --cat-stage-bias 0.45
run_one 3 on_rev_cat20 --cat-stage-bias 0.20
cat logs/overnight/wave9.launch.txt
