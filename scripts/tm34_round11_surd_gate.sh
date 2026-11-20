
#!/bin/bash
# TM-34 Round-11: SURD/SMI-gated biases vs WIN tsb=2.0
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tsmoe
mkdir -p logs/tm34
CACHE=runs/tokyo/cache_abl_tgt_last
COMMON=(--cache "$CACHE" --seed 42 --router-mode hard_prior --dim 256 --heads 8 \
  --lambda-bpr 0.25 --transition-score-bias 2.0 --epochs 1 --checkpoint-metric acc1)

# Free leftover hillclimb if still holding GPU1
if ps -p 29808 >/dev/null 2>&1; then
  echo "Stopping leftover PID 29808 (hc_filter_ctx)"
  kill 29808 || true
  sleep 2
fi

run_one() {
  local gpu="$1" name="$2"; shift 2
  local log="logs/tm34/${name}.log"
  echo "[$(date -Iseconds)] GPU${gpu} start ${name}" | tee -a logs/tm34/round11.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${COMMON[@]}" "$@" \
    >"$log" 2>&1 &
  local pid=$!
  echo "GPU${gpu} PID=${pid} run=${name}" | tee -a logs/tm34/round11.launch.txt
  echo "$pid" > "logs/tm34/${name}.pid"
}

# GPU0: unique_p residual + mild surd score bias
run_one 0 hc_sg_up_res_ssb1 \
  --surd-bias-gate unique_p --tsb-gate-mode residual --surd-score-bias 1.0

# GPU1: smi_p residual + surd score bias
run_one 1 hc_sg_smip_res_ssb1 \
  --surd-bias-gate smi_p --tsb-gate-mode residual --surd-score-bias 1.0

# GPU2: unique_p mul (focus) + higher surd score bias
run_one 2 hc_sg_up_mul_ssb2 \
  --surd-bias-gate unique_p --tsb-gate-mode mul --surd-score-bias 2.0

# GPU0 also packs attn residual if free later; use GPU3 only if free.
MEM3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')
if [ "${MEM3:-99999}" -lt 2000 ]; then
  run_one 3 hc_sg_up_res_attn \
    --surd-bias-gate unique_p --tsb-gate-mode residual --surd-attn-gate residual --surd-score-bias 1.0
else
  # Pack on GPU0 (ep1 ~1GB) — second job after tiny delay so first allocates
  sleep 8
  run_one 0 hc_sg_up_res_attn \
    --surd-bias-gate unique_p --tsb-gate-mode residual --surd-attn-gate residual --surd-score-bias 1.0
fi

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | tee -a logs/tm34/round11.launch.txt
echo "Launched:" | tee -a logs/tm34/round11.launch.txt
cat logs/tm34/round11.launch.txt | tail -20
