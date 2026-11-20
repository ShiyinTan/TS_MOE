
#!/bin/bash
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/tm34
CACHE=runs/tokyo/cache_abl_tgt_last
COMMON=(--cache "$CACHE" --seed 42 --router-mode hard_prior --dim 256 --heads 8       --epochs 1 --checkpoint-metric acc1 --transition-score-bias 2.0       --surd-bias-gate unique_p --tsb-gate-mode residual       --candidate-context on --candidate-context-time last_slot)
: > logs/tm34/round16.launch.txt
run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name}" | tee -a logs/tm34/round16.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train --out "runs/tokyo/${name}" "${COMMON[@]}" "$@"         >"logs/tm34/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/tm34/round16.launch.txt
  echo $! > "logs/tm34/${name}.pid"
}
# Acc recovery on A+time (R20=0.593 but Acc=0.250)
run_one 0 hc_sg_Atime_ssb025_bpr03 --lambda-bpr 0.30 --surd-score-bias 0.25
run_one 1 hc_sg_Atime_ssb028_bpr025 --lambda-bpr 0.25 --surd-score-bias 0.28
run_one 2 hc_sg_Atime_ssb025_bpr035 --lambda-bpr 0.35 --surd-score-bias 0.25
MEM3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')
if [ "${MEM3:-99999}" -lt 2000 ]; then
  run_one 3 hc_sg_Atime_flat_ssb025 --lambda-bpr 0.25 --surd-score-bias 0.25 --tsb-gate-mode off
else
  sleep 8
  # flat tsb (no residual) + Atime — Acc may recover
  # override: need to not pass residual; COMMON has residual — use separate call
  echo "[$(date -Iseconds)] GPU0 hc_sg_Atime_flat_ssb025" | tee -a logs/tm34/round16.launch.txt
  CUDA_VISIBLE_DEVICES=0 nohup python -m tsmoe train         --out runs/tokyo/hc_sg_Atime_flat_ssb025 --cache "$CACHE" --seed 42         --router-mode hard_prior --dim 256 --heads 8 --epochs 1 --checkpoint-metric acc1         --transition-score-bias 2.0 --lambda-bpr 0.25         --surd-bias-gate unique_p --surd-score-bias 0.25         --candidate-context on --candidate-context-time last_slot         >logs/tm34/hc_sg_Atime_flat_ssb025.log 2>&1 &
  echo "GPU0 PID=$! hc_sg_Atime_flat_ssb025" | tee -a logs/tm34/round16.launch.txt
  echo $! > logs/tm34/hc_sg_Atime_flat_ssb025.pid
fi
cat logs/tm34/round16.launch.txt
