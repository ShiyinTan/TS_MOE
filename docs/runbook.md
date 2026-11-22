# Runbook — runnable commands

Install / data / full CLI catalog: [`../README.md`](../README.md).  
Settings → [`model-settings.md`](model-settings.md) · results → [`experiments.md`](experiments.md).

No overnight/leftover wave shell scripts. Prefer **config files** for Base / WIN / Peak.

---

## Env (GPU server)

```bash
cd TS_MOE
conda activate tsmoe   # or: conda env create -f environment.yml
pip install 'pyyaml>=6,<7'   # if env predates PyYAML
python -c "import torch; assert torch.cuda.is_available()"
```

---

## Config recipes (preferred)

Committed under `configs/`:

| Path | Meaning |
| --- | --- |
| `configs/recipes/{base,win,peak}.yaml` | Shared knobs (WIN/Peak `extends` Base) |
| `configs/<city>/{base,win,peak}.yaml` | City `cache` + `out` + `extends` recipe |

Keys use argparse dest names (`lambda_bpr`, `transition_score_bias`, …). Meta: `recipe`, `recipe_id`, `city`, `description`, `extends`.

```bash
# LOCKED WIN (Tokyo)
python -m tsmoe train --config configs/tokyo/win.yaml

# Acc peak
python -m tsmoe train --config configs/tokyo/peak.yaml

# Multi-city Base / WIN / Peak
python -m tsmoe train --config configs/beijing/base.yaml
python -m tsmoe train --config configs/shanghai/win.yaml
python -m tsmoe train --config configs/melbourne/peak.yaml

# CLI overrides win over the file (last flag wins)
python -m tsmoe train --config configs/tokyo/win.yaml --device cuda:1

# Evaluate: config supplies cache/device only; pass --checkpoint
python -m tsmoe evaluate \
  --config configs/tokyo/peak.yaml \
  --checkpoint runs/tokyo/on_rev_soft10r/best.pt \
  --split test
```

→ metrics under `results/<run>/eval_test/` (`known_only` Acc@1).

One-factor Acc climb: start from a recipe config, override `--out` and **one** Extra flag.

```bash
python -m tsmoe train --config configs/tokyo/peak.yaml \
  --out runs/tokyo/my_abl --geo-score-power 1.5
```

Pack free GPU memory; ≤8 of our jobs/GPU; do not kill foreign jobs.

---

## Prepare Tokyo Acc-climb cache

```bash
python -m tsmoe prepare \
  --data-root data --city tokyo \
  --out runs/tokyo/cache_abl_tgt_last \
  --train-targets last --eval-targets last
```

Reuse this `--cache` for WIN / soft10r / leftover trains (no re-prepare for score-bias / soft-target / dim changes). Same pattern for other Massive-STEPS cities (`runs/<city>/cache_abl_tgt_last`).

---

## Train LOCKED WIN (equivalent flag soup)

```bash
python -m tsmoe train \
  --cache runs/tokyo/cache_abl_tgt_last \
  --out runs/tokyo/hc_sg_Atime_bpr029_ssb026 \
  --seed 42 --device cuda \
  --router-mode hard_prior --dim 256 --heads 8 \
  --lambda-bpr 0.29 \
  --transition-score-bias 2.0 \
  --surd-bias-gate unique_p --tsb-gate-mode residual --surd-score-bias 0.26 \
  --candidate-context on --candidate-context-time last_slot \
  --epochs 1 --checkpoint-metric acc1
```

Prefer: `python -m tsmoe train --config configs/tokyo/win.yaml`

→ `results/tokyo/hc_sg_Atime_bpr029_ssb026/eval_test/metrics.json` (`known_only`)

---

## Train Acc peak (`on_rev_soft10r`)

Prefer: `python -m tsmoe train --config configs/tokyo/peak.yaml`

Equivalent flags:

```bash
python -m tsmoe train \
  --cache runs/tokyo/cache_abl_tgt_last \
  --out runs/tokyo/on_rev_soft10r \
  --seed 42 --device cuda \
  --router-mode hard_prior --dim 256 --heads 8 \
  --lambda-bpr 0.29 \
  --transition-score-bias 2.0 \
  --surd-bias-gate unique_p --tsb-gate-mode residual --surd-score-bias 0.26 \
  --candidate-context on --candidate-context-time last_slot \
  --transition-attn gate \
  --geo-score-bias 0.15 \
  --revisit-score-bias 0.10 \
  --soft-target-beta 0.10 --soft-target-prior revisit \
  --epochs 1 --checkpoint-metric acc1
```

---

## Read Acc@1

```bash
python -c "import json; m=json.load(open('results/tokyo/on_rev_soft10r/eval_test/metrics.json')); print(m['known_only']['Acc@1'])"
# Hit@1 / Recall@1 equal Acc@1 in known_only reports
```

Independent eval:

```bash
python -m tsmoe evaluate \
  --cache runs/tokyo/cache_abl_tgt_last \
  --checkpoint runs/tokyo/on_rev_soft10r/best.pt \
  --split test --device cuda
```

Ablation master table:

```bash
python -m tsmoe ablation-summary --results results
```
