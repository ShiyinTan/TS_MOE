# results/beijing/ — Acc climb bundle

| File | Content |
| --- | --- |
| [`climb_path.md`](climb_path.md) | Acc@1 提升路径（表格） |
| [`settings_summary.md`](settings_summary.md) / [`.csv`](settings_summary.csv) | 各设定 known_only 结果 |
| [`mean_std.md`](mean_std.md) / [`.csv`](mean_std.csv) | 同设定多次运行的 mean±std |

**Protocol:** known_only Acc@1 · eval_test · seed 42 · hard_prior  
**STHGCN:** 0.453 · **runs with metrics:** 0  
**Best:** —

Regenerate:

```bash
python scripts/city_results_bundle.py --city beijing --sthgcn 0.453
```
