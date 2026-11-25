#!/usr/bin/env python3
"""Finalize a finished climb city on the GPU box.

1. Scan ``results/<city>/*/eval_test/metrics.json``
2. Build an Acc ledger of every **effective** run (new personal-best along the
   climb, beat STHGCN, or match/exceed locked milestones)
3. Register those tags in ``configs/climb_retention.yaml`` so prune never drops them
4. Refresh summaries via ``sync_climb_retention`` / summarize scripts
5. Append a short section to ``results/<city>/climb_path.md``

Important configs/results are **kept**; only non-effective densify debris may be pruned.

    python scripts/city_climb_finalize.py --city melbourne --sthgcn 0.168 --lock-acc 0.173108
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from climb_retention import load_retention, save_retention, sthgcn_map  # noqa: E402

PY = sys.executable


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def scan_runs(city: str) -> list[dict]:
    root = ROOT / "results" / city
    rows: list[dict] = []
    if not root.is_dir():
        return rows
    for metrics in root.glob("*/eval_test/metrics.json"):
        tag = metrics.parent.parent.name
        try:
            blob = json.loads(metrics.read_text())
            ko = blob.get("known_only") or {}
            acc = ko.get("Acc@1")
        except Exception:
            continue
        if acc is None:
            continue
        mtime = metrics.stat().st_mtime
        rows.append(
            {
                "tag": tag,
                "acc": float(acc),
                "mrr": float(ko.get("MRR") or 0.0),
                "ndcg10": float(ko.get("NDCG@10") or ko.get("NDCG@5") or 0.0),
                "mtime": mtime,
                "metrics": str(metrics.relative_to(ROOT)),
            }
        )
    rows.sort(key=lambda r: (r["mtime"], r["tag"]))
    return rows


def effective_runs(
    rows: list[dict],
    *,
    sthgcn: float,
    lock_acc: float | None,
    milestones: list[float] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return (ledger_steps, keep_rows).

    A run is effective if it:
    - sets a new running best Acc (chronological personal best), or
    - first beats STHGCN, or
    - reaches/exceeds a milestone / final lock Acc (6dp).
    """
    milestones = list(milestones or [])
    if lock_acc is not None:
        milestones.append(float(lock_acc))
    milestones = sorted({round(m, 6) for m in milestones})

    best = -1.0
    beat_sthgcn = False
    hit_ms: set[float] = set()
    steps: list[dict] = []
    keep_tags: set[str] = set()

    for r in rows:
        acc = float(r["acc"])
        reasons: list[str] = []
        if round(acc, 6) > round(best, 6):
            reasons.append(f"new_best:{best:.6f}->{acc:.6f}" if best >= 0 else f"first:{acc:.6f}")
            best = acc
        if not beat_sthgcn and round(acc, 6) > round(sthgcn, 6):
            reasons.append(f"beat_sthgcn:{sthgcn}")
            beat_sthgcn = True
        for ms in milestones:
            if ms not in hit_ms and round(acc, 6) >= ms:
                reasons.append(f"milestone>={ms:.6f}")
                hit_ms.add(ms)
        if reasons:
            step = {**r, "reasons": reasons}
            steps.append(step)
            keep_tags.add(r["tag"])

    # Always keep absolute top-5 by Acc even if not on the PB path (near-miss / TIE@lock)
    for r in sorted(rows, key=lambda x: -x["acc"])[:5]:
        keep_tags.add(r["tag"])

    keep_rows = [r for r in rows if r["tag"] in keep_tags]
    keep_rows.sort(key=lambda r: -r["acc"])
    return steps, keep_rows


def write_ledger(city: str, steps: list[dict], keep_rows: list[dict], meta: dict) -> Path:
    out_dir = ROOT / "results" / city
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "city": city,
        "written_at": _now(),
        **meta,
        "n_steps": len(steps),
        "n_keep": len(keep_rows),
        "steps": [
            {
                "tag": s["tag"],
                "acc": s["acc"],
                "mrr": s["mrr"],
                "reasons": s["reasons"],
                "metrics": s["metrics"],
            }
            for s in steps
        ],
        "keep_tags": [r["tag"] for r in keep_rows],
    }
    path = out_dir / "acc_ledger.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        f"# {city.title()} Acc ledger (effective changes)",
        "",
        f"Written: {payload['written_at']}",
        f"STHGCN: {meta.get('sthgcn')} · lock Acc: {meta.get('lock_acc')}",
        f"Steps recorded: {len(steps)} · keep tags: {len(keep_rows)}",
        "",
        "| Acc@1 | Tag | Why kept |",
        "| ---: | --- | --- |",
    ]
    for s in steps:
        lines.append(f"| {s['acc']:.6f} | `{s['tag']}` | {', '.join(s['reasons'])} |")
    lines.append("")
    lines.append("## All keep tags (incl. top-5)")
    lines.append("")
    for r in keep_rows:
        lines.append(f"- `{r['tag']}` · Acc={r['acc']:.6f}")
    (out_dir / "acc_ledger.md").write_text("\n".join(lines) + "\n")
    return path


def update_climb_path(city: str, steps: list[dict], lock_acc: float | None, win_tag: str | None) -> None:
    path = ROOT / "results" / city / "climb_path.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    block = [
        "",
        f"<!-- auto:city_climb_finalize {_now()} -->",
        f"## Acc ledger (auto)",
        "",
        f"Final lock Acc@1: **{lock_acc:.6f}**" if lock_acc is not None else "Final lock Acc@1: —",
        f"WIN tag: `{win_tag}`" if win_tag else "",
        "",
        "Effective Acc changes kept (see `acc_ledger.md`):",
        "",
    ]
    for s in steps[-12:]:
        block.append(f"- **{s['acc']:.6f}** · `{s['tag']}` · {', '.join(s['reasons'])}")
    block.append("")
    text = path.read_text() if path.is_file() else f"# {city.title()} Acc@1 climb path\n"
    # replace prior auto block if present
    marker = "<!-- auto:city_climb_finalize"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    path.write_text(text.rstrip() + "\n" + "\n".join(block))


def register_keeps(
    city: str,
    *,
    keep_tags: list[str],
    lock_config: str | None,
    lock_acc: float | None,
    status: str = "STOPPED",
) -> None:
    data = load_retention()
    sthgcn = sthgcn_map(data)
    # ensure sthgcn entry exists when provided via CLI later
    cities = data.setdefault("cities", {})
    c = cities.setdefault(city, {})
    keeps_c = set(c.get("keep_configs") or [])
    keeps_c.update({"base.yaml", "peak.yaml", "win.yaml"})
    if lock_config:
        keeps_c.add(Path(lock_config).name)
        c["lock_config"] = (
            lock_config if str(lock_config).startswith("configs/") else f"configs/{city}/{Path(lock_config).name}"
        )
    # keep foundation climb yamls that are not wave densify
    for p in (ROOT / "configs" / city).glob("*.yaml"):
        if "_w" in p.stem and any(ch.isdigit() for ch in p.stem.split("_w", 1)[-1][:3]):
            # wave densify — only keep if matching an effective result tag
            if p.stem in keep_tags:
                keeps_c.add(p.name)
            continue
        if p.name.startswith("climb_") or p.name in {"base.yaml", "peak.yaml", "win.yaml"}:
            # keep non-wave climb foundations + any effective stem yaml
            if "_w" not in p.stem or p.stem in keep_tags:
                keeps_c.add(p.name)
    for tag in keep_tags:
        # if a non-gitignored win yaml exists with this stem, keep it
        yp = ROOT / "configs" / city / f"{tag}.yaml"
        if yp.is_file() and "_w" not in tag:
            keeps_c.add(yp.name)
        # also keep promoted win that may differ in name
    c["keep_configs"] = sorted(keeps_c)
    tags = set(c.get("keep_results") or [])
    tags.update(keep_tags)
    if lock_config:
        tags.add(Path(lock_config).stem)
    c["keep_results"] = sorted(tags)
    # Exact tags only — do NOT widen to whole-wave prefixes (would keep
    # every densify sibling). Dead densify may still be pruned later.
    c["keep_results_prefixes"] = []
    if lock_acc is not None:
        c["lock_acc"] = float(lock_acc)
    c["status"] = status
    c["aggressive_configs"] = status == "STOPPED"
    c["keep_wave"] = None
    cities[city] = c
    data["cities"] = cities
    if city not in sthgcn and city == "melbourne":
        data.setdefault("sthgcn", {})["melbourne"] = 0.168
    if city == "bandung":
        data.setdefault("sthgcn", {})["bandung"] = 0.219
    save_retention(data)


def run_summaries(city: str, sthgcn: float, keep_tags: list[str], *, prune: bool) -> None:
    # configs index (+ optional prune of obsolete wave yamls not in keep)
    cfg_cmd = [PY, "scripts/summarize_and_prune_configs.py", "--city", city]
    for name in ("base.yaml", "peak.yaml", "win.yaml"):
        cfg_cmd += ["--keep", name]
    # keep config files whose stem is an effective tag
    for tag in keep_tags:
        cfg_cmd += ["--keep", f"{tag}.yaml"]
    if not prune:
        cfg_cmd.append("--summarize-only")
    print("+", " ".join(cfg_cmd))
    subprocess.call(cfg_cmd, cwd=ROOT)

    res_cmd = [
        PY,
        "scripts/summarize_and_prune_results.py",
        "--city",
        city,
        "--sthgcn",
        str(sthgcn),
        "--keep-top",
        "5",
    ]
    for tag in keep_tags:
        res_cmd += ["--keep", tag]
    if not prune:
        res_cmd.append("--summarize-only")
    print("+", " ".join(res_cmd))
    subprocess.call(res_cmd, cwd=ROOT)
    # Do NOT call city_results_bundle here — it rewrites climb_path.md and
    # would clobber the Acc ledger section we just appended.


def finalize(
    city: str,
    *,
    sthgcn: float | None = None,
    lock_acc: float | None = None,
    lock_config: str | None = None,
    prune: bool = True,
) -> dict:
    data = load_retention()
    if sthgcn is None:
        sthgcn = float(sthgcn_map(data).get(city) or 0.0)
    rows = scan_runs(city)
    # historical MEL milestones (harmless for other cities if unused)
    milestones = [0.168277, 0.169887, 0.170692]
    steps, keep_rows = effective_runs(
        rows, sthgcn=sthgcn, lock_acc=lock_acc, milestones=milestones
    )
    keep_tags = [r["tag"] for r in keep_rows]
    win_tag = keep_rows[0]["tag"] if keep_rows else None
    if lock_acc is None and keep_rows:
        lock_acc = float(keep_rows[0]["acc"])

    write_ledger(
        city,
        steps,
        keep_rows,
        {"sthgcn": sthgcn, "lock_acc": lock_acc, "n_scanned": len(rows)},
    )
    update_climb_path(city, steps, lock_acc, win_tag)
    register_keeps(
        city,
        keep_tags=keep_tags,
        lock_config=lock_config,
        lock_acc=lock_acc,
        status="STOPPED",
    )
    run_summaries(city, float(sthgcn), keep_tags, prune=prune)

    out = {
        "event": "city_finalized",
        "city": city,
        "sthgcn": sthgcn,
        "lock_acc": lock_acc,
        "n_scanned": len(rows),
        "n_steps": len(steps),
        "n_keep": len(keep_tags),
        "win_tag": win_tag,
        "ledger": f"results/{city}/acc_ledger.md",
        "prune": prune,
        "written_at": _now(),
    }
    (ROOT / "runs" / "climb_state").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "climb_state" / f"finalize_{city}.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    ap.add_argument("--sthgcn", type=float, default=None)
    ap.add_argument("--lock-acc", type=float, default=None)
    ap.add_argument("--lock-config", default=None)
    ap.add_argument(
        "--summarize-only",
        action="store_true",
        help="Refresh summaries/ledger/retention; do not delete result dirs",
    )
    args = ap.parse_args()
    out = finalize(
        args.city,
        sthgcn=args.sthgcn,
        lock_acc=args.lock_acc,
        lock_config=args.lock_config,
        prune=not args.summarize_only,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
