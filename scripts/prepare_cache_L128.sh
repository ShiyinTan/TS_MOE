#!/usr/bin/env bash
# Re-prepare train_all/eval_last caches with --max-length 128 for history>50 climb.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${TSMOE_PYTHON:-/home/shiyintan/miniconda3/envs/tsmoe/bin/python}"
cd "$ROOT"
cities=("$@")
if [[ ${#cities[@]} -eq 0 ]]; then
  cities=(bandung melbourne beijing palembang)
fi
for city in "${cities[@]}"; do
  out="runs/${city}/cache_tgt_all_eval_last_L128"
  echo "=== prepare ${city} → ${out} ==="
  "$PY" -m tsmoe prepare \
    --data-root data --city "$city" \
    --out "$out" \
    --max-length 128 \
    --train-targets all --eval-targets last
done
echo "done"
