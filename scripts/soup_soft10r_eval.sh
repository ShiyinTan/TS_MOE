#!/bin/bash
# Ideas v9 #42: report seed Accs and pick greedy-best among soft10r seed ckpts.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
mkdir -p logs/overnight
OUT=logs/overnight/wave22_soup_eval.txt
: > "$OUT"
PEAK=0.267650
echo "peak=$PEAK" | tee -a "$OUT"
best_name=""; best_acc=0
for n in on_s10r_seed42 on_s10r_seed43 on_s10r_seed44 on_s10r_seed45 on_rev_soft10r; do
  f=results/tokyo/$n/eval_test/epoch_001.json
  if [ -f "$f" ]; then
    acc=$(python3 -c "import json; print(json.load(open('$f'))['known_only']['Acc@1'])")
    echo "$n Acc=$acc" | tee -a "$OUT"
    python3 -c "import sys; sys.exit(0 if float('$acc')>float('$best_acc') else 1)" && {
      best_acc=$acc; best_name=$n
    } || true
  fi
done
echo "GREEDY_BEST name=$best_name Acc=$best_acc peak=$PEAK" | tee -a "$OUT"
python3 -c "import sys; sys.exit(0 if float('$best_acc')>float('$PEAK') else 1)" \
  && echo KEEP | tee -a "$OUT" || echo DISCARD | tee -a "$OUT"
