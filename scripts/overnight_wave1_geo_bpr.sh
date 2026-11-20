#!/bin/bash
# Overnight Wave-1: Acc@1 climb from peak stack (tr_gate + geo).
# Peak: hc_win_trgate_gsb015 Acc=0.263435 (geo λ=0.15).
# Pack onto free VRAM even if mobedit holds ~18GB/GPU.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave1.launch.txt

PEAK=(
  --cache "$CACHE"
  --seed 42 --device cuda
  --router-mode hard_prior --dim 256 --heads 8
  --lambda-bpr 0.29
  --transition-score-bias 2.0
  --surd-bias-gate unique_p --tsb-gate-mode residual --surd-score-bias 0.26
  --candidate-context on --candidate-context-time last_slot
  --transition-attn gate
  --epochs 1 --checkpoint-metric acc1
)

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave1.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave1.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

# Fine geo λ around 0.15 + mild BPR on peak geo.
run_one 0 on_gsb010 --geo-score-bias 0.10
run_one 1 on_gsb012 --geo-score-bias 0.12
run_one 2 on_gsb018 --geo-score-bias 0.18
run_one 3 on_gsb015_bpr032 --geo-score-bias 0.15 --lambda-bpr 0.32
cat logs/overnight/wave1.launch.txt
