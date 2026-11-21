#!/bin/bash
# Leftover #45: next-category MTL aux on soft10r stack (η ∈ {0.1, 0.3}).
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave25.launch.txt

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

# Pick free GPUs (memory.used < 2GiB); never touch camob/mobedit.
mapfile -t FREE < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < 2048) print $1}')
if ((${#FREE[@]} < 1)); then
  echo "No free GPUs (all busy). Abort launch." | tee -a logs/overnight/wave25.launch.txt
  exit 1
fi

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave25.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave25.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

# η=0.1 then η=0.3 on first two free GPUs
run_one "${FREE[0]}" on_s10r_cat01 --lambda-cat-aux 0.1
if ((${#FREE[@]} >= 2)); then
  run_one "${FREE[1]}" on_s10r_cat03 --lambda-cat-aux 0.3
else
  echo "Only 1 free GPU — launched cat01 only; cat03 deferred" | tee -a logs/overnight/wave25.launch.txt
fi
cat logs/overnight/wave25.launch.txt
