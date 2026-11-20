#!/bin/bash
# Free Acc@1 check: eval-time score temperature on peak ckpt (no train).
# Sweep T on validation Acc@1, then report test for best T (+ T=1 baseline).
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight results/tokyo/on_calib_peak
CACHE=runs/tokyo/cache_abl_tgt_last
CKPT=runs/tokyo/hc_win_trgate_gsb015/best.pt
GPU="${1:-0}"
OUT=logs/overnight/calib_peak_T.txt
: > "$OUT"
echo "calib peak ckpt=$CKPT GPU=$GPU" | tee -a "$OUT"

best_t=1.0
best_acc=-1
for T in 0.7 0.8 0.9 1.0 1.1 1.2 1.5; do
  echo "=== val T=$T ===" | tee -a "$OUT"
  CUDA_VISIBLE_DEVICES="$GPU" python -m tsmoe evaluate \
    --cache "$CACHE" --checkpoint "$CKPT" --split validation \
    --score-temperature "$T" --device cuda \
    --out "results/tokyo/on_calib_peak/val_T${T}" \
    2>&1 | tee -a "$OUT" | tr '\r' '\n' | grep -E 'Acc@1=|primary' | tee -a "$OUT" || true
  ACC=$(python3 -c "import json; m=json.load(open('results/tokyo/on_calib_peak/val_T${T}/metrics.json')); print(m['known_only']['Acc@1'])" 2>/dev/null || echo -1)
  echo "parsed_val_acc=$ACC T=$T" | tee -a "$OUT"
  python3 -c "import sys; a=float('$ACC'); b=float('$best_acc'); sys.exit(0 if a>b else 1)" && best_acc=$ACC && best_t=$T || true
done
echo "BEST_VAL T=$best_t Acc=$best_acc" | tee -a "$OUT"
echo "=== test T=$best_t ===" | tee -a "$OUT"
CUDA_VISIBLE_DEVICES="$GPU" python -m tsmoe evaluate \
  --cache "$CACHE" --checkpoint "$CKPT" --split test \
  --score-temperature "$best_t" --device cuda \
  --out "results/tokyo/on_calib_peak/test_T${best_t}" \
  2>&1 | tee -a "$OUT"
python3 -c "import json; m=json.load(open('results/tokyo/on_calib_peak/test_T${best_t}/metrics.json')); ko=m['known_only']; print('TEST', {k:ko[k] for k in ['Acc@1','Recall@5','Recall@10','Recall@20']})" | tee -a "$OUT"
