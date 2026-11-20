
#!/bin/bash
# TM-34 Round-13: close R@5 / R@20 without losing Acc@1 peak (~0.258)
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tsmoe
mkdir -p logs/tm34
CACHE=runs/tokyo/cache_abl_tgt_last
BASE=(--cache "$CACHE" --seed 42 --router-mode hard_prior --dim 256 --heads 8 \
  --lambda-bpr 0.25 --epochs 1 --checkpoint-metric acc1 \
  --surd-bias-gate unique_p --tsb-gate-mode residual)
: > logs/tm34/round13.launch.txt

run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} start ${name}" | tee -a logs/tm34/round13.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train \
    --out "runs/tokyo/${name}" "${BASE[@]}" "$@" \
    >"logs/tm34/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! run=${name}" | tee -a logs/tm34/round13.launch.txt
  echo $! > "logs/tm34/${name}.pid"
}

# Peak neighborhood: weaker/stronger ssb
run_one 0 hc_sg_up_res_ssb015 --transition-score-bias 2.0 --surd-score-bias 0.15
run_one 1 hc_sg_up_res_ssb020 --transition-score-bias 2.0 --surd-score-bias 0.20
run_one 2 hc_sg_up_res_ssb030 --transition-score-bias 2.0 --surd-score-bias 0.30

# R@20 probe: peak ssb=0.25 + slightly stronger transition prior
MEM3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')
if [ "${MEM3:-99999}" -lt 2000 ]; then
  run_one 3 hc_sg_up_res_ssb025_tsb25 --transition-score-bias 2.5 --surd-score-bias 0.25
else
  sleep 8
  run_one 0 hc_sg_up_res_ssb025_tsb25 --transition-score-bias 2.5 --surd-score-bias 0.25
fi

nvidia-smi --query-gpu=index,memory.used --format=csv | tee -a logs/tm34/round13.launch.txt
cat logs/tm34/round13.launch.txt
