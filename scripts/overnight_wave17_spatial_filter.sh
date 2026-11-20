#!/bin/bash
# Ideas v8 #10: coverage-safe spatial top-M pool shrink on frozen soft10r.
# Val select → test. Abort note if natural GT coverage < 0.98.
set -euo pipefail
cd /home/shiyintan/Code/TS_MOE
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tsmoe
mkdir -p logs/overnight results/tokyo/on_s10r_w17
CACHE=runs/tokyo/cache_abl_tgt_last
CKPT=runs/tokyo/on_rev_soft10r/best.pt
OUT=logs/overnight/wave17_filter.txt
: > "$OUT"
: > logs/overnight/wave17.launch.txt
echo "wave17 #10 spatial-filter ckpt=$CKPT $(date -Iseconds)" | tee -a "$OUT"

run_eval() {
  local name="$1" gpu="$2" split="$3"; shift 3
  local dest="results/tokyo/on_s10r_w17/${split}_${name}"
  echo "[$(date -Iseconds)] GPU${gpu} ${split} ${name} $*" \
    | tee -a logs/overnight/wave17.launch.txt "$OUT"
  CUDA_VISIBLE_DEVICES="$gpu" python -m tsmoe evaluate \
    --cache "$CACHE" --checkpoint "$CKPT" --split "$split" \
    --device cuda --out "$dest" "$@" \
    >"logs/overnight/w17_${split}_${name}.log" 2>&1
  python3 -c "
import json
m=json.load(open('${dest}/metrics.json'))
ko=m['known_only']
st=m.get('candidate_prior_stats') or {}
print('PARSED', '${split}', '${name}', 'Acc@1', ko['Acc@1'],
      'R@5', ko.get('Recall@5'),
      'cov', st.get('natural_gt_coverage'),
      'size', st.get('size_after_mean'))
" | tee -a "$OUT"
}

# 8 cells: geo/gravity × top-M
run_eval geo64 0 validation --candidate-spatial-prior geo --candidate-prior-top-m 64 &
run_eval geo128 1 validation --candidate-spatial-prior geo --candidate-prior-top-m 128 &
run_eval geo256 2 validation --candidate-spatial-prior geo --candidate-prior-top-m 256 &
run_eval geo512 3 validation --candidate-spatial-prior geo --candidate-prior-top-m 512 &
wait
run_eval grav64 0 validation --candidate-spatial-prior gravity --candidate-prior-top-m 64 --candidate-prior-gravity-beta 1.0 &
run_eval grav128 1 validation --candidate-spatial-prior gravity --candidate-prior-top-m 128 --candidate-prior-gravity-beta 1.0 &
run_eval grav256 2 validation --candidate-spatial-prior gravity --candidate-prior-top-m 256 --candidate-prior-gravity-beta 1.0 &
run_eval rad128 3 validation --candidate-spatial-prior radiation --candidate-prior-top-m 128 &
wait

python3 - <<'PY' | tee -a "$OUT"
import json
from pathlib import Path
root = Path("results/tokyo/on_s10r_w17")
CLI = {
  "geo64": "--candidate-spatial-prior geo --candidate-prior-top-m 64",
  "geo128": "--candidate-spatial-prior geo --candidate-prior-top-m 128",
  "geo256": "--candidate-spatial-prior geo --candidate-prior-top-m 256",
  "geo512": "--candidate-spatial-prior geo --candidate-prior-top-m 512",
  "grav64": "--candidate-spatial-prior gravity --candidate-prior-top-m 64 --candidate-prior-gravity-beta 1.0",
  "grav128": "--candidate-spatial-prior gravity --candidate-prior-top-m 128 --candidate-prior-gravity-beta 1.0",
  "grav256": "--candidate-spatial-prior gravity --candidate-prior-top-m 256 --candidate-prior-gravity-beta 1.0",
  "rad128": "--candidate-spatial-prior radiation --candidate-prior-top-m 128",
}
best_name, best_acc, best_cov = None, -1.0, None
for p in sorted(root.glob("validation_*/metrics.json")):
    name = p.parent.name.replace("validation_", "", 1)
    m = json.load(p.open())
    acc = float(m["known_only"]["Acc@1"])
    cov = (m.get("candidate_prior_stats") or {}).get("natural_gt_coverage")
    print(f"val {name} Acc@1={acc:.6f} cov={cov}")
    if acc > best_acc:
        best_acc, best_name, best_cov = acc, name, cov
print(f"BEST_VAL name={best_name} Acc={best_acc} cov={best_cov}")
Path("logs/overnight/wave17_best.txt").write_text(
    f"{best_name}\t{best_acc}\t{best_cov}\t{CLI[best_name]}\n")
PY
IFS=$'\t' read -r BNAME BACC BCOV BCLI < logs/overnight/wave17_best.txt
echo "selected $BNAME Acc=$BACC cov=$BCOV cli=$BCLI" | tee -a "$OUT"
# shellcheck disable=SC2086
run_eval "$BNAME" 0 test $BCLI
run_eval base 1 test
echo "wave17 done best=${BNAME} valAcc=${BACC} cov=${BCOV} $(date -Iseconds)" | tee -a "$OUT"
