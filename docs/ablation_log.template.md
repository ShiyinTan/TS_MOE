# Ablation results log (template)

`train` / `evaluate` **auto-append** matching rows to gitignored `results/ablation_log.md` and `results/ablation_log.csv` (created if missing). Copy this file only if you want a blank checklist or to reset:

```bash
cp docs/ablation_log.template.md results/ablation_log.md
```

Source of numbers: latest test `all_targets.Acc@1` / `NDCG@1` / `MRR` (train also notes best-val).
Protocol: see [`ablation.md`](ablation.md).

| date | run_name | cache | key flags | Acc@1 | NDCG@1 | MRR | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | `abl_router_hard` | `runs/tokyo/cache` | `--router-mode hard_prior` |  |  |  |  |
|  | `abl_router_surd_mlp` | `runs/tokyo/cache` | `--router-mode surd_mlp` |  |  |  |  |
|  | `abl_router_hybrid` | `runs/tokyo/cache` | `--router-mode hybrid --lambda-prior 1 --hybrid-lambda 0.1` |  |  |  |  |
|  | `abl_neg_63` | `runs/tokyo/cache` | `--negatives 63 --max-train-candidates -1` |  |  |  |  |
|  | `abl_neg_all_max256` | `runs/tokyo/cache` | `--negatives -1 --max-train-candidates 256` |  |  |  |  |
|  | `abl_causal_on` | `runs/tokyo/cache` | `--causal-counts --transition-attn bias` |  |  |  |  |
|  | `abl_causal_off` | `runs/tokyo/cache` | `--no-causal-counts` |  |  |  |  |
|  | `abl_tr_bias` | `runs/tokyo/cache` | `--transition-attn bias` |  |  |  |  |
|  | `abl_tr_gate` | `runs/tokyo/cache` | `--transition-attn gate` |  |  |  |  |
|  | `abl_tr_rel_key` | `runs/tokyo/cache` | `--transition-attn rel_key` |  |  |  |  |
|  | `abl_cand_baseline` | `runs/tokyo/cache` | `--candidate-radius p90` + default multi-source |  |  |  | §5a baseline |
|  | `abl_cand_random` | `runs/tokyo/cache_abl_random` | `--candidate-radius random --random-negatives 50` |  |  |  | §5a; re-prepare |
|  | `abl_cand_p80` | `runs/tokyo/cache_abl_p80` | `--candidate-radius p80` |  |  |  | re-prepare |
|  | `abl_cand_p70` | `runs/tokyo/cache_abl_p70` | `--candidate-radius p70` |  |  |  | re-prepare |
|  | `abl_cand_rnd0` | `runs/tokyo/cache_abl_rnd0` | `--random-negatives 0` |  |  |  | re-prepare |
|  | `abl_cand_no_pop` | `runs/tokyo/cache_abl_no_pop` | `--popular-top-k 0` |  |  |  | re-prepare |
|  | `abl_cand_no_uf` | `runs/tokyo/cache_abl_no_uf` | `--user-frequent-top-k 0` |  |  |  | re-prepare |
|  | `abl_cand_nbhd` | `runs/tokyo/cache_abl_nbhd` | pop=0 uf=0 rnd=50 |  |  |  | re-prepare |
|  | `abl_cand_full` | `runs/tokyo/cache_abl_full` | `--candidate-radius full` |  |  |  | re-prepare |
|  | `abl_tgt_all` | `runs/tokyo/cache_abl_tgt_all` | `--train-targets all --eval-targets all` |  |  |  | re-prepare |
|  | `abl_tgt_last` | `runs/tokyo/cache_abl_tgt_last` | `--train-targets last --eval-targets last` |  |  |  | re-prepare |
|  | `abl_tgt_train_all_eval_last` | `runs/tokyo/cache_abl_tgt_all_eval_last` | `--train-targets all --eval-targets last` |  |  |  | re-prepare |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
