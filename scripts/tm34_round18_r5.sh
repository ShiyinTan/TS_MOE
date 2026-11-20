
#!/bin/bash
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/tm34
CACHE=runs/tokyo/cache_abl_tgt_last
: > logs/tm34/round18.launch.txt
run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name}" | tee -a logs/tm34/round18.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train --out "runs/tokyo/${name}"         --cache "$CACHE" --seed 42 --router-mode hard_prior --dim 256 --heads 8         --epochs 1 --checkpoint-metric acc1 --transition-score-bias 2.0         --surd-bias-gate unique_p --tsb-gate-mode residual         --candidate-context on --candidate-context-time last_slot         "$@" >"logs/tm34/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/tm34/round18.launch.txt
  echo $! > "logs/tm34/${name}.pid"
}
run_one 0 hc_sg_Atime_bpr029_ssb026 --lambda-bpr 0.29 --surd-score-bias 0.26
run_one 1 hc_sg_Atime_bpr031_ssb026 --lambda-bpr 0.31 --surd-score-bias 0.26
run_one 2 hc_sg_Atime_bpr03_ssb026 --lambda-bpr 0.30 --surd-score-bias 0.26
sleep 8
run_one 0 hc_sg_Atime_bpr03_ssb029 --lambda-bpr 0.30 --surd-score-bias 0.29
cat logs/tm34/round18.launch.txt
