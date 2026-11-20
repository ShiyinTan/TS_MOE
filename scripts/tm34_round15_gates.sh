
#!/bin/bash
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/tm34
CACHE=runs/tokyo/cache_abl_tgt_last
COMMON=(--cache "$CACHE" --seed 42 --router-mode hard_prior --dim 256 --heads 8   --lambda-bpr 0.25 --epochs 1 --checkpoint-metric acc1 --transition-score-bias 2.0   --tsb-gate-mode residual --surd-score-bias 0.25)
: > logs/tm34/round15.launch.txt
run_one() {
  local gpu="$1" name="$2"; shift 2
  echo "[$(date -Iseconds)] GPU${gpu} ${name}" | tee -a logs/tm34/round15.launch.txt
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m tsmoe train --out "runs/tokyo/${name}" "${COMMON[@]}" "$@"     >"logs/tm34/${name}.log" 2>&1 &
  echo "GPU${gpu} PID=$! ${name}" | tee -a logs/tm34/round15.launch.txt
  echo $! > "logs/tm34/${name}.pid"
}
# Alt gates at Acc-peak ssb
run_one 0 hc_sg_umass_res_ssb025 --surd-bias-gate u_mass
run_one 1 hc_sg_conf_res_ssb025 --surd-bias-gate confidence
run_one 2 hc_sg_smipc_res_ssb025 --surd-bias-gate smi_pc
# Peak + stronger transition-attn lambda (deeper hist bias)
MEM3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')
if [ "${MEM3:-99999}" -lt 2000 ]; then
  run_one 3 hc_sg_up_res_ssb025_tal15 --surd-bias-gate unique_p --transition-lambda 1.5
else
  sleep 8
  run_one 0 hc_sg_up_res_ssb025_tal15 --surd-bias-gate unique_p --transition-lambda 1.5
fi
cat logs/tm34/round15.launch.txt
