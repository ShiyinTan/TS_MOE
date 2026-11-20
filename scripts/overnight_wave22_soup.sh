#!/bin/bash
# Ideas v9 #42: soft10r seed ensemble / greedy soup (seeds 42–45).
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave22.launch.txt

PEAK=(
  --cache "$CACHE"
  --device cuda
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
  local gpu="$1" seed="$2" name="$3"
  echo "[$(date -Iseconds)] GPU${gpu} ${name} seed=${seed}" | tee -a logs/overnight/wave22.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" --seed "$seed" "${PEAK[@]}" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave22.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

# seed 42 already = on_rev_soft10r peak; train 43–45 + re-eval 42 for soup
run_one 0 43 on_s10r_seed43
run_one 1 44 on_s10r_seed44
run_one 2 45 on_s10r_seed45
# seed42 clone for fair soup slot (or reuse peak ckpt in soup script)
run_one 3 42 on_s10r_seed42
cat logs/overnight/wave22.launch.txt
