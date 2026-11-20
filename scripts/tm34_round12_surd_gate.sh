
#!/bin/bash
# TM-34 Round-12: weaken ssb / flat tsb+ssb / tsb=3 probe
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tsmoe
mkdir -p logs/tm34
CACHE=runs/tokyo/cache_abl_tgt_last
BASE=(--cache "$CACHE" --seed 42 --router-mode hard_prior --dim 256 --heads 8 \
  --lambda-bpr 0.25 --epochs 1 --checkpoint-metric acc1)
: > logs/tm34/round12.launch.txt

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} start ${name}" | tee -a logs/tm34/round12.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${BASE[@]}" "$@" \
    >"logs/tm34/${name}.log" 2>&1 &
  local pid=$!
  echo "GPU${gpu} PID=${pid} run=${name}" | tee -a logs/tm34/round12.launch.txt
  echo "$pid" > "logs/tm34/${name}.pid"
}

# unique_p residual + weak ssb
run_one 0 hc_sg_up_res_ssb025 \
  --transition-score-bias 2.0 --surd-bias-gate unique_p --tsb-gate-mode residual --surd-score-bias 0.25

# unique_p additive only (flat tsb)
run_one 1 hc_sg_up_flat_ssb1 \
  --transition-score-bias 2.0 --surd-bias-gate unique_p --surd-score-bias 1.0

# unique_p residual only (no additive SURD score)
run_one 2 hc_sg_up_res_ssb0 \
  --transition-score-bias 2.0 --surd-bias-gate unique_p --tsb-gate-mode residual

# stronger flat tsb (no SURD gate) — R@20 probe
MEM3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')
if [ "${MEM3:-99999}" -lt 2000 ]; then
  run_one 3 hc_sg_tsb30 --transition-score-bias 3.0
else
  sleep 8
  run_one 0 hc_sg_tsb30 --transition-score-bias 3.0
fi

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a logs/tm34/round12.launch.txt
cat logs/tm34/round12.launch.txt
