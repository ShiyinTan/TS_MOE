# Results layout

Final evaluation **metrics and reports** are stored under a top-level `results/` directory. Checkpoints, caches, and training history stay under `runs/`.

## What goes where

| Location | Contents |
| --- | --- |
| `runs/<city>/cache/` | `prepare` cache (`manifest.json`, split `.pt` files, SURD tables) |
| `runs/<city>/model/` | `train` checkpoint `best.pt` and per-epoch `history.json` |
| `results/<run_name>/eval_validation/` | Best-checkpoint **validation** metrics after `train` |
| `results/<run_name>/eval_test/` | Final **test** metrics after `train`, or default output of `evaluate --split test` |

Each `eval_*` directory contains:

- `metrics.json` — full payload (Acc/Hit/Recall, NDCG, MRR, MAP @1/5/10/20, groups, device, cache id, timestamp, model config, …)
- `report.txt` — same summary printed to the terminal (Acc@1 headline + tables)

Optional: `evaluate --output path.json` still mirrors the JSON to an extra path.

## How `<run_name>` is chosen

Defaults derive from the train `--out` / checkpoint directory by stripping a leading `runs/` segment under the working directory:

| Train / checkpoint path | Default results root |
| --- | --- |
| `runs/tokyo/model` (or `…/best.pt`) | `results/tokyo/model/` |
| `runs/demo/model` | `results/demo/model/` |

So:

```text
results/tokyo/model/eval_validation/{metrics.json,report.txt}
results/tokyo/model/eval_test/{metrics.json,report.txt}
```

Override the eval destination with `evaluate --out <dir>` (writes `metrics.json` + `report.txt` directly in that directory).

## Finding Acc@1 after train

```bash
# From repo root, after:
#   python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model
python -c "import json; m=json.load(open('results/tokyo/model/eval_test/metrics.json')); print(m['all_targets']['Acc@1'])"
# or read the human report:
grep -E 'summary|Acc@1' results/tokyo/model/eval_test/report.txt
```

Primary headline metrics in reports are **Acc@1**, **NDCG@1**, and full **MRR** (`all_targets`). Checkpoint selection during training still uses validation **NDCG@10**.

## Relation to `runs/`

- **Do not** look under `runs/.../model/eval_*` for metrics — those paths are no longer used.
- `runs/` remains the place for heavy artifacts you may want to keep or delete independently of published numbers.
- Both `runs/` and `results/` are gitignored; only code and docs are tracked.

## Example commands

```bash
python -m tsmoe prepare --data-root data --city tokyo --out runs/tokyo/cache
python -m tsmoe train --cache runs/tokyo/cache --out runs/tokyo/model
# → checkpoints: runs/tokyo/model/best.pt
# → results:     results/tokyo/model/eval_validation/ and .../eval_test/

python -m tsmoe evaluate --cache runs/tokyo/cache --checkpoint runs/tokyo/model/best.pt --split test
# → results/tokyo/model/eval_test/{metrics.json,report.txt}
```
