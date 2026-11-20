#!/bin/bash
# Ladder A free Acc: score-temperature sweep on frozen on_rev010 peak ckpt.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight results/tokyo/on_calib_rev010
CACHE=runs/tokyo/cache_abl_tgt_last
CKPT=runs/tokyo/on_rev010/best.pt
GPU="${1:-0}"
OUT=logs/overnight/calib_rev010_T.txt
: > "$OUT"
echo "calib on_rev010 ckpt=$CKPT GPU=$GPU $(date -Iseconds)" | tee -a "$OUT"

best_t=1.0
best_acc=-1
for T in 0.7 0.8 0.9 1.0 1.1 1.2 1.5; do
  echo "=== val T=$T ===" | tee -a "$OUT"
  CUDA_VISIBLE_DEVICES="$GPU" python -m tsmoe evaluate \
    --cache "$CACHE" --checkpoint "$CKPT" --split validation \
    --score-temperature "$T" --device cuda \
    --out "results/tokyo/on_calib_rev010/val_T${T}" \
    2>&1 | tee -a "$OUT" | tr '\r' '\n' | grep -E 'Acc@1=|primary' | tee -a "$OUT" || true
  ACC=$(python3 -c "import json;print(json.load(open('results/tokyo/on_calib_rev010/val_T${T}/metrics.json'))['known_only']['Acc@1'])" 2>/dev/null || echo -1)
  echo "parsed_val_acc=$ACC T=$T" | tee -a "$OUT"
  python3 -c "import sys; a=float('$ACC'); b=float('$best_acc'); sys.exit(0 if a>b else 1)" && best_acc=$ACC && best_t=$T || true
done
echo "BEST_VAL T=$best_t Acc=$best_acc" | tee -a "$OUT"
echo "=== test T=$best_t and T=1.0 ===" | tee -a "$OUT"
for T in "$best_t" 1.0; do
  CUDA_VISIBLE_DEVICES="$GPU" python -m tsmoe evaluate \
    --cache "$CACHE" --checkpoint "$CKPT" --split test \
    --score-temperature "$T" --device cuda \
    --out "results/tokyo/on_calib_rev010/test_T${T}" \
    2>&1 | tee -a "$OUT"
  python3 -c "import json; m=json.load(open('results/tokyo/on_calib_rev010/test_T${T}/metrics.json')); ko=m['known_only']; print('TEST_T', '$T', {k:ko[k] for k in ['Acc@1','Recall@5','Recall@10','Recall@20']})" | tee -a "$OUT"
done
