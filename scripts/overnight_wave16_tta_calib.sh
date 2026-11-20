#!/bin/bash
# Ideas v8 #34/#36: Acc-aware gauss calib + TTA on frozen soft10r (Acc 0.267650).
# Val select → test. Parallel 4-GPU first wave.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight results/tokyo/on_s10r_w16
CACHE=runs/tokyo/cache_abl_tgt_last
CKPT=runs/tokyo/on_rev_soft10r/best.pt
OUT=logs/overnight/wave16_tta_calib.txt
: > "$OUT"
: > logs/overnight/wave16.launch.txt
echo "wave16 #34/#36 ckpt=$CKPT $(date -Iseconds)" | tee -a "$OUT"

run_eval() {
  local name="$1" gpu="$2" split="$3"; shift 3
  local dest="results/tokyo/on_s10r_w16/${split}_${name}"
  echo "[$(date -Iseconds)] GPU${gpu} ${split} ${name} $*" \
    | tee -a logs/overnight/wave16.launch.txt "$OUT"
  CUDA_VISIBLE_DEVICES="$gpu" python -m tsmoe evaluate \
    --cache "$CACHE" --checkpoint "$CKPT" --split "$split" \
    --device cuda --out "$dest" "$@" \
    >"logs/overnight/w16_${split}_${name}.log" 2>&1
  python3 -c "
import json
m=json.load(open('${dest}/metrics.json'))
ko=m['known_only']
print('PARSED', '${split}', '${name}', 'Acc@1', ko['Acc@1'],
      'R@5', ko.get('Recall@5'), 'tta', m.get('tta'),
      'calib', m.get('score_calib'), m.get('calib_lambda'))
" | tee -a "$OUT"
}

# Wave A: 4 configs in parallel on val
run_eval tta_mask 0 validation --tta mask --tta-views 2 --tta-fuse mean &
run_eval tta_ms 1 validation --tta mask,substitute --tta-views 1 --tta-fuse mean &
run_eval gpop05 2 validation --score-calib gauss_pop --calib-lambda 0.05 &
run_eval grev05 3 validation --score-calib gauss_rev --calib-lambda 0.05 &
wait
run_eval tta_rrf 0 validation --tta mask,substitute --tta-views 1 --tta-fuse rrf &
run_eval gpop10 1 validation --score-calib gauss_pop --calib-lambda 0.10 &
run_eval grev10 2 validation --score-calib gauss_rev --calib-lambda 0.10 &
run_eval tta_mask_rrf 3 validation --tta mask --tta-views 2 --tta-fuse rrf &
wait

python3 - <<'PY' | tee -a "$OUT"
import json
from pathlib import Path
root = Path("results/tokyo/on_s10r_w16")
best_name, best_acc = None, -1.0
best_args = ""
# Map name → CLI fragment for test replay
CLI = {
  "tta_mask": "--tta mask --tta-views 2 --tta-fuse mean",
  "tta_ms": "--tta mask,substitute --tta-views 1 --tta-fuse mean",
  "tta_rrf": "--tta mask,substitute --tta-views 1 --tta-fuse rrf",
  "tta_mask_rrf": "--tta mask --tta-views 2 --tta-fuse rrf",
  "gpop05": "--score-calib gauss_pop --calib-lambda 0.05",
  "gpop10": "--score-calib gauss_pop --calib-lambda 0.10",
  "grev05": "--score-calib gauss_rev --calib-lambda 0.05",
  "grev10": "--score-calib gauss_rev --calib-lambda 0.10",
}
for p in sorted(root.glob("validation_*/metrics.json")):
    name = p.parent.name.replace("validation_", "", 1)
    acc = float(json.load(p.open())["known_only"]["Acc@1"])
    print(f"val {name} Acc@1={acc:.6f}")
    if acc > best_acc:
        best_acc, best_name = acc, name
print(f"BEST_VAL name={best_name} Acc={best_acc}")
Path("logs/overnight/wave16_best.txt").write_text(
    f"{best_name}\t{best_acc}\t{CLI[best_name]}\n")
PY
IFS=$'\t' read -r BNAME BACC BCLI < logs/overnight/wave16_best.txt
echo "selected $BNAME Acc=$BACC cli=$BCLI" | tee -a "$OUT"
# shellcheck disable=SC2086
run_eval "$BNAME" 0 test $BCLI
# Always report locked baseline on test for comparison
run_eval base 1 test
echo "wave16 done best=${BNAME} valAcc=${BACC} $(date -Iseconds)" | tee -a "$OUT"
