#!/bin/bash
# Ideas v7 #38: locked-bias residual micro-mix on frozen soft10r (Acc 0.267650).
# Grid (geo, revisit) within ±0.02 of locked (0.15, 0.10). Val select → test.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight results/tokyo/on_s10r_eps
CACHE=runs/tokyo/cache_abl_tgt_last
CKPT=runs/tokyo/on_rev_soft10r/best.pt
OUT=logs/overnight/wave15_eps.txt
: > "$OUT"
: > logs/overnight/wave15.launch.txt
echo "wave15 #38 locked-eps ckpt=$CKPT $(date -Iseconds)" | tee -a "$OUT"

# (name geo rev gpu) — 8 cells + locked baseline on round-robin GPUs
CELLS=(
  "base 0.15 0.10 0"
  "g17 0.17 0.10 1"
  "g13 0.13 0.10 2"
  "r12 0.15 0.12 3"
  "r08 0.15 0.08 0"
  "g17r12 0.17 0.12 1"
  "g13r08 0.13 0.08 2"
  "g17r08 0.17 0.08 3"
  "g13r12 0.13 0.12 0"
)

run_eval() {
  local name="$1" geo="$2" rev="$3" gpu="$4" split="$5"
  local dest="results/tokyo/on_s10r_eps/${split}_${name}"
  echo "[$(date -Iseconds)] GPU${gpu} ${split} ${name} geo=${geo} rev=${rev}" \
    | tee -a logs/overnight/wave15.launch.txt "$OUT"
  CUDA_VISIBLE_DEVICES="$gpu" python -m tsmoe evaluate \
    --cache "$CACHE" --checkpoint "$CKPT" --split "$split" \
    --geo-score-bias "$geo" --revisit-score-bias "$rev" \
    --device cuda --out "$dest" \
    >"logs/overnight/eps_${split}_${name}.log" 2>&1
  python3 -c "
import json
m=json.load(open('${dest}/metrics.json'))
ko=m['known_only']
print('PARSED', '${split}', '${name}', 'geo=${geo}', 'rev=${rev}',
      'Acc@1', ko['Acc@1'], 'R@5', ko.get('Recall@5'))
" | tee -a "$OUT"
}

# Phase 1: validation grid (parallel waves of 4)
pids=()
for cell in "${CELLS[@]}"; do
  set -- $cell
  name=$1; geo=$2; rev=$3; gpu=$4
  run_eval "$name" "$geo" "$rev" "$gpu" validation &
  pids+=($!)
  # keep at most 4 concurrent
  if ((${#pids[@]} >= 4)); then
    for p in "${pids[@]}"; do wait "$p"; done
    pids=()
  fi
done
for p in "${pids[@]:-}"; do wait "$p"; done

# Pick best val Acc → write wave15_best.txt
python3 - <<'PY' | tee -a "$OUT"
import json
from pathlib import Path
root = Path("results/tokyo/on_s10r_eps")
best_name, best_acc = "base", -1.0
best_geo, best_rev = 0.15, 0.10
rows = []
for p in sorted(root.glob("validation_*/metrics.json")):
    name = p.parent.name.replace("validation_", "", 1)
    meta = json.load(p.open())
    acc = float(meta["known_only"]["Acc@1"])
    geo = float(meta.get("eval_geo_score_bias", 0.15))
    rev = float(meta.get("eval_revisit_score_bias", 0.10))
    rows.append((acc, name, geo, rev))
    print(f"val {name} Acc@1={acc:.6f} geo={geo} rev={rev}")
    if acc > best_acc:
        best_acc, best_name, best_geo, best_rev = acc, name, geo, rev
print(f"BEST_VAL name={best_name} Acc={best_acc} geo={best_geo} rev={best_rev}")
Path("logs/overnight/wave15_best.txt").write_text(
    f"{best_name} {best_geo} {best_rev} {best_acc}\n")
PY
read -r BNAME BGEO BREV BACC < logs/overnight/wave15_best.txt

# Phase 2: test on best + locked baseline (skip duplicate if best==base)
run_eval "$BNAME" "$BGEO" "$BREV" 0 test
if [[ "$BNAME" != "base" ]]; then
  run_eval "base" 0.15 0.10 1 test
fi
echo "wave15 done best=${BNAME} geo=${BGEO} rev=${BREV} valAcc=${BACC} $(date -Iseconds)" | tee -a "$OUT"
