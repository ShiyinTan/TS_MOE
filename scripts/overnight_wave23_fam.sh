#!/bin/bash
# Ideas v9 #44: familiar/unfamiliar area soft blend on soft10r (revisit λ locked).
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/overnight/wave23.launch.txt

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
  --epochs 1 --checkpoint-metric acc1
  --fam-area-blend on
)

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name} $*" | tee -a logs/overnight/wave23.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${PEAK[@]}" "$@" \
    >"logs/overnight/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/overnight/wave23.launch.txt
  echo $! > "logs/overnight/${name}.pid"
}

run_one 0 on_s10r_fam05 --fam-radius-km 0.5 --fam-explore-bias 0.05 --fam-return-bias 0.0
run_one 1 on_s10r_fam10 --fam-radius-km 0.5 --fam-explore-bias 0.10 --fam-return-bias 0.0
run_one 2 on_s10r_fam_ret --fam-radius-km 0.5 --fam-explore-bias 0.05 --fam-return-bias 0.05
run_one 3 on_s10r_fam1km --fam-radius-km 1.0 --fam-explore-bias 0.05 --fam-return-bias 0.0
cat logs/overnight/wave23.launch.txt
