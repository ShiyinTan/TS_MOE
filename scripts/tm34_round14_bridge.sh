
#!/bin/bash
# TM-34 Round-14: bridge ssb 0.20 recall ↔ 0.25 Acc; softer BPR; A+time
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tsmoe
mkdir -p logs/tm34
CACHE=runs/tokyo/cache_abl_tgt_last
COMMON=(--cache "$CACHE" --seed 42 --router-mode hard_prior --dim 256 --heads 8 \
  --epochs 1 --checkpoint-metric acc1 --transition-score-bias 2.0 \
  --surd-bias-gate unique_p --tsb-gate-mode residual)
: > logs/tm34/round14.launch.txt

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} start ${name}" | tee -a logs/tm34/round14.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${COMMON[@]}" "$@" \
    >"logs/tm34/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! run=${name}" | tee -a logs/tm34/round14.launch.txt
  echo $! > "logs/tm34/${name}.pid"
}

run_one 0 hc_sg_up_res_ssb022 --lambda-bpr 0.25 --surd-score-bias 0.22
run_one 1 hc_sg_up_res_ssb024 --lambda-bpr 0.25 --surd-score-bias 0.24
run_one 2 hc_sg_up_res_ssb025_bpr02 --lambda-bpr 0.20 --surd-score-bias 0.25

MEM3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')
if [ "${MEM3:-99999}" -lt 2000 ]; then
  run_one 3 hc_sg_up_res_ssb025_Atime --lambda-bpr 0.25 --surd-score-bias 0.25 \
    --candidate-context on --candidate-context-time last_slot
else
  sleep 8
  run_one 0 hc_sg_up_res_ssb025_Atime --lambda-bpr 0.25 --surd-score-bias 0.25 \
    --candidate-context on --candidate-context-time last_slot
fi

cat logs/tm34/round14.launch.txt
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv
