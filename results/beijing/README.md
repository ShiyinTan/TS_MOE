# results/beijing/ — Acc climb bundle

| File | Content |
| --- | --- |
| [`climb_path.md`](climb_path.md) | Locked Acc@1 climb ladder |
| [`settings_summary.md`](settings_summary.md) / [`.csv`](settings_summary.csv) | Per-setting known_only summary (keep) |
| [`mean_std.md`](mean_std.md) / [`.csv`](mean_std.csv) | Multi-run mean±std (keep) |

**Retention:** after a city is STOPPED/locked, keep these summary files + WIN eval dirs; individual non-WIN config result folders may be deleted to save disk. Do **not** delete summary/stat files.

**Protocol:** known_only Acc@1 · eval_test · seed 42 · hard_prior  
**STHGCN:** 0.453 · **Locked WIN Acc@1:** **0.465909**

Regenerate summaries (needs run dirs present):

```bash
python scripts/city_results_bundle.py --city beijing --sthgcn 0.453
```

Summarize then prune non-kept run dirs **on the GPU box** (avoids agent token burn):

```bash
python scripts/summarize_and_prune_results.py --city beijing --sthgcn 0.453 --dry-run
python scripts/summarize_and_prune_results.py --city beijing --sthgcn 0.453
# or: python scripts/summarize_and_prune_results.py --all-stopped
```
