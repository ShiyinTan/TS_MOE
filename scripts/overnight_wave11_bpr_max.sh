#!/bin/bash
# Ladder D step 2: BPR-max / SimCE (one loss lever; no focal). Peak revisit0.10+geo0.15.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave11.launch.txt

PEAK=(
  --cache "$CACHE"
  --seed 42 --device cuda
  --router-mode hard_prior --dim 256 --heads 8
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
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave11.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave11.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

# Peak already uses --lambda-bpr 0.29 --bpr-mode max; try weighting modes.
run_one 0 on_rev_bprmax29 --lambda-bpr 0.29 --bpr-mode bpr_max
run_one 1 on_rev_bprmax45 --lambda-bpr 0.45 --bpr-mode bpr_max
run_one 2 on_rev_bprmax15 --lambda-bpr 0.15 --bpr-mode bpr_max
run_one 3 on_rev_simce29 --lambda-bpr 0.29 --bpr-mode simce
cat logs/overnight/wave11.launch.txt
