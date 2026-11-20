#!/bin/bash
# Ladder A step 5: multi-hop tsb on locked peak (revisit0.10 + geo0.15).
# Keep total tsb scale 2.0; K∈{2,3}, decay κ∈{0.5,1.0}. Skip plain tsb-decay.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave7.launch.txt

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
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave7.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave7.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

run_one 0 on_rev_mh2_d05 --tsb-history-k 2 --tsb-history-decay 0.5
run_one 1 on_rev_mh3_d05 --tsb-history-k 3 --tsb-history-decay 0.5
run_one 2 on_rev_mh2_d10 --tsb-history-k 2 --tsb-history-decay 1.0
run_one 3 on_rev_mh3_d10 --tsb-history-k 3 --tsb-history-decay 1.0
cat logs/overnight/wave7.launch.txt
