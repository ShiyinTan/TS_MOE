#!/bin/bash
# Ladder A step 3: radiation / jump-length replace geo0.15 on locked peak
# = WIN + tr_gate + revisit0.10 + {radiation | jump}  (no flat geo).
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave5.launch.txt

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
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave5.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave5.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

# Replace geo0.15; keep revisit0.10. Radiation λ start ≤0.1; jump λ≈0.2.
run_one 0 on_rev_rad010 --radiation-score-bias 0.10 --radiation-kernel hard
run_one 1 on_rev_rad005 --radiation-score-bias 0.05 --radiation-kernel hard
run_one 2 on_rev_jump02 --jump-length-bias 0.20 --jump-dist lognormal
run_one 3 on_rev_jump_tp --jump-length-bias 0.20 --jump-dist trunc_power --jump-beta 1.5 --jump-d0 0.1
cat logs/overnight/wave5.launch.txt
