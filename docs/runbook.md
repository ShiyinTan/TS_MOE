# Runbook — runnable commands

Install / data / full CLI catalog: [`../README.md`](../README.md).  
Settings → [`model-settings.md`](model-settings.md) · results → [`experiments.md`](experiments.md).

No overnight/leftover wave shell scripts. Use `python -m tsmoe …` directly.

---

## Env (GPU server)

```bash
cd TS_MOE
conda activate tsmoe   # or: conda env create -f environment.yml
python -c "import torch; assert torch.cuda.is_available()"
```

---

## Prepare Tokyo Acc-climb cache

```bash
python -m tsmoe prepare \
  --data-root data --city tokyo \
  --out runs/tokyo/cache_abl_tgt_last \
  --train-targets last --eval-targets last
```

Reuse this `--cache` for WIN / soft10r / leftover trains (no re-prepare for score-bias / soft-target / dim changes).

---

## Train LOCKED WIN

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

→ `results/tokyo/hc_sg_Atime_bpr029_ssb026/eval_test/metrics.json` (`known_only`)

---

## Train Acc peak (`on_rev_soft10r`)

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

## One leftover on soft10r (example)

Same flags as peak, change `--out` and add **one** Extra, e.g.:

```bash
# #4 2-hop (0.3 or 0.5)
... soft10r flags ... --out runs/tokyo/on_s10r_2hop03 --transition-2hop-bias 0.3

# #8 hour-pop
... --out runs/tokyo/on_s10r_hpop10 --hour-pop-score-bias 0.10

# #45 cat MTL
... --out runs/tokyo/on_s10r_cat01 --lambda-cat-aux 0.1
```

Pack free GPU memory; do not kill foreign jobs.

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
