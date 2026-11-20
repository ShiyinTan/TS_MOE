#!/bin/bash
# Ladder A step 4: return-vs-explore gate on locked revisit0.10 peak.
# Do NOT raise flat revisit λ. Mix π·revisit + (1−π)·explore spatial.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave6.launch.txt

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
  --return-gate on
  --epochs 1 --checkpoint-metric acc1
)

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave6.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave6.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

# Keep geo0.15 (peak spatial) + gated revisit/explore; also try replace-geo explore-only.
run_one 0 on_rev_rgate_g20 --geo-score-bias 0.15 --explore-spatial-bias gauss --explore-score-bias 0.20 --gauss-sigma-km 0.5
run_one 1 on_rev_rgate_g15 --geo-score-bias 0.15 --explore-spatial-bias gauss --explore-score-bias 0.15 --gauss-sigma-km 0.5
run_one 2 on_rev_rgate_rad --geo-score-bias 0.15 --explore-spatial-bias radiation --explore-score-bias 0.05
run_one 3 on_rev_rgate_nogeo --explore-spatial-bias gauss --explore-score-bias 0.20 --gauss-sigma-km 0.5
cat logs/overnight/wave6.launch.txt
