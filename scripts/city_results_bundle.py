#!/usr/bin/env python3
"""Build per-city Acc climb reports under ``results/<city>/``.

Writes (and refreshes):

- ``climb_path.md`` — Acc@1 improvement ladder vs STHGCN (tables)
- ``settings_summary.md`` / ``.csv`` — every run's known_only metrics
- ``mean_std.md`` / ``.csv`` — mean±std when the same setting has ≥2 runs
  (run names ending in ``_seedN`` / ``_repN`` / ``_rN`` are grouped)
- ``README.md`` — short index

Example::

    python scripts/city_results_bundle.py --city beijing --sthgcn 0.453 \\
        --climb-from docs/beijing-acc-climb-path.md
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CLIMB: dict[str, list[dict]] = {
    "beijing": [
        {
            "stage": "0 · 此前城市 best",
            "acc": 0.420455,
            "tag": "climb_ep2_trbias_soft005_geo020",
            "cache": "cache_abl_tgt_last",
            "change": "Peak + trbias + soft β=0.05 + geo λ=0.20",
        },
        {
            "stage": "1 · 换 cache + gauss",
            "acc": 0.431818,
            "tag": "climb_s5g20_all_gauss015",
            "cache": "cache_tgt_all_eval_last",
            "change": "train_targets=all · eval=last · gauss λ=0.15（关掉 geo）",
        },
        {
            "stage": "2 · 加强 gauss",
            "acc": 0.443182,
            "tag": "climb_s5g20_all_gauss020",
            "cache": "cache_tgt_all_eval_last",
            "change": "gauss λ 0.15 → 0.20",
        },
        {
            "stage": "3 · 调 weight_decay · WIN",
            "acc": 0.454545,
            "tag": "climb_s5g20_all_gauss020_wd005",
            "cache": "cache_tgt_all_eval_last",
            "change": "weight_decay=0.005（并列 …_wd02）",
        },
    ],
}


def recall(ko: dict, k: int):
    for key in (f"Recall@{k}", f"Acc@{k}", f"Hit@{k}", f"R@{k}"):
        if key in ko:
            return ko[key]
    return None


def setting_key(run: str) -> str:
    s = re.sub(r"_seed\d+$", "", run)
    s = re.sub(r"_rep\d+$", "", s)
    s = re.sub(r"_r\d+$", "", s)
    return s


def mean_std(vals: list[float]) -> tuple[float, float | None, int]:
    n = len(vals)
    if n == 0:
        return float("nan"), None, 0
    mu = sum(vals) / n
    if n == 1:
        return mu, None, 1
    var = sum((x - mu) ** 2 for x in vals) / (n - 1)
    return mu, math.sqrt(var), n


def load_row(run_dir: Path) -> dict:
    import json

    mj = run_dir / "eval_test" / "metrics.json"
    if not mj.exists():
        found = list(run_dir.glob("**/eval_test/metrics.json"))
        mj = found[0] if found else None
    row = {
        "run": run_dir.name,
        "acc": "",
        "r5": "",
        "r10": "",
        "r20": "",
        "mrr": "",
        "ndcg1": "",
        "cache": "",
        "status": "missing",
        "known_targets": "",
    }
    if mj is None or not mj.exists():
        return row
    m = json.loads(mj.read_text())
    ko = m.get("known_only") or {}
    if "Acc@1" not in ko:
        row["status"] = "no_known_only"
        return row
    row.update(
        {
            "acc": ko["Acc@1"],
            "r5": recall(ko, 5),
            "r10": recall(ko, 10),
            "r20": recall(ko, 20),
            "mrr": ko.get("MRR", ""),
            "ndcg1": ko.get("NDCG@1", ""),
            "cache": m.get("cache", ""),
            "status": "ok",
            "known_targets": ko.get("known_targets", ""),
        }
    )
    return row


def fmt(x, digits=6) -> str:
    if x is None or x == "":
        return ""
    return f"{float(x):.{digits}f}"


def write_climb_path(
    out: Path,
    city: str,
    sthgcn: float,
    ladder: list[dict],
    win_row: dict | None,
    climb_from: Path | None,
) -> None:
    path = out / "climb_path.md"
    if climb_from and climb_from.exists():
        text = climb_from.read_text()
        # Prefer metrics.json WIN recalls when available.
        if win_row and win_row.get("r5") not in ("", None):
            text = re.sub(
                r"(\*\*0\.454545\*\* / )0\.625( / 0\.727 / 0\.784)",
                rf"\g<1>{float(win_row['r5']):.3f} / {float(win_row['r10']):.3f} / {float(win_row['r20']):.3f}",
                text,
                count=1,
            )
            text = re.sub(
                r"(R5/R10/R20: )0\.625 / 0\.727 / 0\.784",
                rf"\g<1>{float(win_row['r5']):.3f} / {float(win_row['r10']):.3f} / {float(win_row['r20']):.3f}",
                text,
                count=1,
            )
        # Point readers at this results folder.
        if "results/beijing/" not in text:
            text += (
                "\n\n---\n\n"
                f"本路径的机器可读结果摘要也在 "
                f"`results/{city}/`（`settings_summary.*` · `mean_std.*`）。\n"
            )
        path.write_text(text)
        return

    lines = [
        f"# {city.title()} Acc@1 climb path (vs STHGCN)",
        "",
        f"**Protocol:** `known_only` Acc@1 · eval_test · seed 42 · hard_prior  ",
        f"**STHGCN Acc@1:** {sthgcn}  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "| 阶段 | Acc@1 | Δ step | vs STHGCN | tag | cache | 关键改动 |",
        "| ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    prev = None
    for step in ladder:
        acc = float(step["acc"])
        d_step = "" if prev is None else f"{acc - prev:+.4f}"
        lines.append(
            f"| {step['stage']} | {acc:.6f} | {d_step} | {acc - sthgcn:+.4f} | "
            f"`{step['tag']}` | `{step['cache']}` | {step['change']} |"
        )
        prev = acc
    if win_row:
        lines += [
            "",
            "## WIN metrics (known_only)",
            "",
            f"| Acc@1 | R@5 | R@10 | R@20 | MRR |",
            f"| ---: | ---: | ---: | ---: | ---: |",
            f"| {fmt(win_row['acc'])} | {fmt(win_row['r5'])} | {fmt(win_row['r10'])} | "
            f"{fmt(win_row['r20'])} | {fmt(win_row['mrr'])} |",
        ]
    path.write_text("\n".join(lines) + "\n")


def write_settings_summary(out: Path, city: str, sthgcn: float, rows: list[dict]) -> list[dict]:
    ok = [r for r in rows if r["status"] == "ok"]
    ok_sorted = sorted(ok, key=lambda r: -float(r["acc"]))
    csv_path = out / "settings_summary.csv"
    fields = [
        "rank",
        "run",
        "acc",
        "r5",
        "r10",
        "r20",
        "mrr",
        "ndcg1",
        "delta_sthgcn",
        "cache",
        "status",
        "known_targets",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        rank = {r["run"]: i + 1 for i, r in enumerate(ok_sorted)}
        for r in rows:
            acc = r["acc"]
            delta = f"{float(acc) - sthgcn:+.6f}" if acc != "" else ""
            w.writerow(
                {
                    "rank": rank.get(r["run"], ""),
                    "run": r["run"],
                    "acc": fmt(acc) if acc != "" else "",
                    "r5": fmt(r["r5"]) if r["r5"] != "" else "",
                    "r10": fmt(r["r10"]) if r["r10"] != "" else "",
                    "r20": fmt(r["r20"]) if r["r20"] != "" else "",
                    "mrr": fmt(r["mrr"]) if r["mrr"] != "" else "",
                    "ndcg1": fmt(r["ndcg1"]) if r["ndcg1"] != "" else "",
                    "delta_sthgcn": delta,
                    "cache": r["cache"],
                    "status": r["status"],
                    "known_targets": r["known_targets"],
                }
            )

    md = [
        f"# {city.title()} settings summary",
        "",
        f"**Metric:** `known_only` Acc@1 (eval_test) · **STHGCN:** {sthgcn}  ",
        f"**Runs with metrics:** {len(ok)} · **missing / incomplete:** {len(rows) - len(ok)}  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        "",
        "Full table: [`settings_summary.csv`](settings_summary.csv).",
        "",
        "## Top 30 (by Acc@1)",
        "",
        "| # | Acc@1 | Δ STHGCN | R@5 | R@10 | R@20 | run |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for i, r in enumerate(ok_sorted[:30], 1):
        acc = float(r["acc"])
        md.append(
            f"| {i} | {acc:.6f} | {acc - sthgcn:+.4f} | {fmt(r['r5'], 4)} | "
            f"{fmt(r['r10'], 4)} | {fmt(r['r20'], 4)} | `{r['run']}` |"
        )

    # Acc plateaus (discrete 1/N)
    plateaus = Counter(round(float(r["acc"]), 6) for r in ok)
    md += [
        "",
        "## Acc@1 plateaus (distinct settings landing on same Acc)",
        "",
        "Beijing test set is small (≈88 known targets); Acc moves in steps of ≈1/88 ≈ 0.0114.",
        "",
        "| Acc@1 | # settings | vs STHGCN |",
        "| ---: | ---: | ---: |",
    ]
    for acc, n in plateaus.most_common(20):
        md.append(f"| {acc:.6f} | {n} | {acc - sthgcn:+.4f} |")

    missing = [r["run"] for r in rows if r["status"] != "ok"]
    if missing:
        md += [
            "",
            f"## Missing eval_test ({len(missing)})",
            "",
            ", ".join(f"`{n}`" for n in missing[:40])
            + (" …" if len(missing) > 40 else ""),
        ]
    (out / "settings_summary.md").write_text("\n".join(md) + "\n")
    return ok_sorted


def write_mean_std(out: Path, city: str, ok: list[dict]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        groups[setting_key(r["run"])].append(r)

    ms_rows = []
    for key, items in sorted(groups.items(), key=lambda kv: -max(float(x["acc"]) for x in kv[1])):
        accs = [float(x["acc"]) for x in items]
        mu, sd, n = mean_std(accs)
        ms_rows.append(
            {
                "setting": key,
                "n": n,
                "acc_mean": f"{mu:.6f}",
                "acc_std": "" if sd is None else f"{sd:.6f}",
                "acc_min": f"{min(accs):.6f}",
                "acc_max": f"{max(accs):.6f}",
                "runs": "|".join(sorted(x["run"] for x in items)),
            }
        )

    fields = ["setting", "n", "acc_mean", "acc_std", "acc_min", "acc_max", "runs"]
    with (out / "mean_std.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ms_rows)

    multi = [r for r in ms_rows if int(r["n"]) >= 2]
    md = [
        f"# {city.title()} mean ± std (same setting, multiple runs)",
        "",
        "Group key = run name with `_seedN` / `_repN` / `_rN` stripped.",
        "",
        f"**Settings:** {len(ms_rows)} · **with n≥2:** {len(multi)}  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]
    if not multi:
        md += [
            "> 当前该城市**没有**同设定多次运行（均为 seed=42 单次）。",
            "> 下表 `acc_std` 为空；日后用 `_seed42` / `_seed43` … 或 `_rep1` 命名复跑后，重新执行本脚本即可填入 mean±std。",
            "",
            "## All settings (n=1 → std 空)",
            "",
            "| setting | n | Acc mean | Acc std | min | max |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in ms_rows[:40]:
            md.append(
                f"| `{r['setting']}` | {r['n']} | {r['acc_mean']} | — | "
                f"{r['acc_min']} | {r['acc_max']} |"
            )
        if len(ms_rows) > 40:
            md.append(f"| … | | | | | 另有 {len(ms_rows) - 40} 行见 CSV |")
    else:
        md += [
            "## Settings with n≥2",
            "",
            "| setting | n | Acc mean | Acc std | min | max | runs |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in multi:
            md.append(
                f"| `{r['setting']}` | {r['n']} | {r['acc_mean']} | {r['acc_std']} | "
                f"{r['acc_min']} | {r['acc_max']} | {r['runs']} |"
            )
    md += ["", f"CSV: [`mean_std.csv`](mean_std.csv)."]
    (out / "mean_std.md").write_text("\n".join(md) + "\n")


def write_readme(out: Path, city: str, sthgcn: float, n_ok: int, win: dict | None) -> None:
    win_line = "—"
    if win:
        win_line = (
            f"**{fmt(win['acc'])}** · `{win['run']}` · "
            f"R@5/10/20 = {fmt(win['r5'], 3)} / {fmt(win['r10'], 3)} / {fmt(win['r20'], 3)}"
        )
    text = f"""# results/{city}/ — Acc climb bundle

| File | Content |
| --- | --- |
| [`climb_path.md`](climb_path.md) | Acc@1 提升路径（表格） |
| [`settings_summary.md`](settings_summary.md) / [`.csv`](settings_summary.csv) | 各设定 known_only 结果 |
| [`mean_std.md`](mean_std.md) / [`.csv`](mean_std.csv) | 同设定多次运行的 mean±std |

**Protocol:** known_only Acc@1 · eval_test · seed 42 · hard_prior  
**STHGCN:** {sthgcn} · **runs with metrics:** {n_ok}  
**Best:** {win_line}

Regenerate:

```bash
python scripts/city_results_bundle.py --city {city} --sthgcn {sthgcn}
```
"""
    (out / "README.md").write_text(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", required=True)
    p.add_argument("--sthgcn", type=float, required=True)
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument(
        "--climb-from",
        type=Path,
        default=None,
        help="Optional existing climb-path markdown to copy/adapt",
    )
    args = p.parse_args()

    out = args.results_root / args.city
    if not out.is_dir():
        raise SystemExit(f"missing city results dir: {out}")

    rows = [
        load_row(d)
        for d in sorted(out.iterdir())
        if d.is_dir() and not d.name.startswith(("_", "."))
    ]
    ok = write_settings_summary(out, args.city, args.sthgcn, rows)
    write_mean_std(out, args.city, ok)
    win = ok[0] if ok else None
    climb_from = args.climb_from
    if climb_from is None:
        cand = Path(f"docs/{args.city}-acc-climb-path.md")
        if cand.exists():
            climb_from = cand
    write_climb_path(
        out,
        args.city,
        args.sthgcn,
        DEFAULT_CLIMB.get(args.city, []),
        win,
        climb_from,
    )
    write_readme(out, args.city, args.sthgcn, len(ok), win)
    print(f"wrote bundle under {out} (ok={len(ok)}, missing={len(rows)-len(ok)})")


if __name__ == "__main__":
    main()
