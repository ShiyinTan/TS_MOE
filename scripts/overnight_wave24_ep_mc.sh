#!/bin/bash
# Ideas v10 #47 ep2–3 Acc1 ckpt + #46 MC-dropout eval on soft10r.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
CKPT=runs/tokyo/on_rev_soft10r/best.pt
: > logs/overnight/wave24.launch.txt

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
  --negatives 63
  --checkpoint-metric acc1
)

echo "[$(date -Iseconds)] GPU0 on_s10r_ep2 --epochs 2" | tee -a logs/overnight/wave24.launch.txt
CUDA_VISIBLE_DEVICES=0 nohup python -m tsmoe train --out runs/tokyo/on_s10r_ep2 "${PEAK[@]}" --epochs 2 \
  >logs/overnight/on_s10r_ep2.log 2>&1 &
echo $! > logs/overnight/on_s10r_ep2.pid

echo "[$(date -Iseconds)] GPU1 on_s10r_ep3 --epochs 3" | tee -a logs/overnight/wave24.launch.txt
CUDA_VISIBLE_DEVICES=1 nohup python -m tsmoe train --out runs/tokyo/on_s10r_ep3 "${PEAK[@]}" --epochs 3 \
  >logs/overnight/on_s10r_ep3.log 2>&1 &
echo $! > logs/overnight/on_s10r_ep3.pid

# #46 MC-dropout eval (training-free) on frozen soft10r
echo "[$(date -Iseconds)] GPU2 mc10 evaluate" | tee -a logs/overnight/wave24.launch.txt
CUDA_VISIBLE_DEVICES=2 nohup python -m tsmoe evaluate \
  --cache "$CACHE" --checkpoint "$CKPT" --split test --device cuda \
  --mc-dropout 10 --mc-keep-train-dropout \
  --out results/tokyo/on_s10r_mc10 \
  >logs/overnight/on_s10r_mc10.log 2>&1 &
echo $! > logs/overnight/on_s10r_mc10.pid

echo "[$(date -Iseconds)] GPU3 mc20 evaluate" | tee -a logs/overnight/wave24.launch.txt
CUDA_VISIBLE_DEVICES=3 nohup python -m tsmoe evaluate \
  --cache "$CACHE" --checkpoint "$CKPT" --split test --device cuda \
  --mc-dropout 20 --mc-keep-train-dropout \
  --out results/tokyo/on_s10r_mc20 \
  >logs/overnight/on_s10r_mc20.log 2>&1 &
echo $! > logs/overnight/on_s10r_mc20.pid

cat logs/overnight/wave24.launch.txt
