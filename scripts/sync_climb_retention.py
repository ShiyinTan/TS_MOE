#!/usr/bin/env python3
"""After each climb Acc lock / new wave: update retention + refresh prune summaries.

This is the one entrypoint that keeps ``summarize_and_prune_configs`` and
``summarize_and_prune_results`` in sync with the latest important climb configs.

Examples (run on GPU box — no chat uploads)::

    # Re-run prune/summary from configs/climb_retention.yaml
    python scripts/sync_climb_retention.py --apply

    # New MEL Acc lock + move active wave to w18
    python scripts/sync_climb_retention.py --city melbourne \\
      --lock-config configs/melbourne/climb_..._win.yaml \\
      --results-tag climb_..._w18_... \\
      --acc 0.1725 --keep-wave w18 --results-prefix climb_bpr020_soft010_w18_ \\
      --apply

    # Mark city STOPPED after clear margin
    python scripts/sync_climb_retention.py --city melbourne --status STOPPED --apply
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from climb_retention import (  # noqa: E402
    RETENTION_PATH,
    load_retention,
    register_lock,
    save_retention,
    sthgcn_map,
)


def run(cmd: list[str], dry_run: bool) -> int:
    print("+", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=ROOT)


def apply_city(city: str, data: dict, *, dry_run: bool, summarize_only: bool) -> None:
    c = (data.get("cities") or {}).get(city) or {}
    sthgcn = sthgcn_map(data).get(city)
    py = sys.executable

    cfg_cmd = [py, "scripts/summarize_and_prune_configs.py", "--city", city]
    wave = c.get("keep_wave")
    if wave:
        cfg_cmd += ["--keep-wave", str(wave)]
    if c.get("aggressive_configs"):
        cfg_cmd.append("--aggressive")
    for name in c.get("keep_configs") or []:
        cfg_cmd += ["--keep", name]
    if summarize_only:
        cfg_cmd.append("--summarize-only")
    if dry_run:
        cfg_cmd.append("--dry-run")
    run(cfg_cmd, dry_run=False)  # dry-run flag already in cmd

    if sthgcn is None:
        print(f"  skip results prune: no sthgcn for {city}")
        return
    res_cmd = [
        py,
        "scripts/summarize_and_prune_results.py",
        "--city",
        city,
        "--sthgcn",
        str(sthgcn),
        "--keep-top",
        "3",
    ]
    for tag in c.get("keep_results") or []:
        res_cmd += ["--keep", tag]
    for pref in c.get("keep_results_prefixes") or []:
        # results tool matches startswith(full dir name)
        if pref.startswith("_"):
            continue
        res_cmd += ["--keep-prefix", pref]
    if summarize_only:
        res_cmd.append("--summarize-only")
    if dry_run:
        res_cmd.append("--dry-run")
    run(res_cmd, dry_run=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", action="append", dest="cities", help="Limit to city (repeatable)")
    ap.add_argument("--all", action="store_true", help="All cities in climb_retention.yaml")
    ap.add_argument("--lock-config", help="New locked WIN yaml path")
    ap.add_argument("--results-tag", help="results/<city>/<tag>/ dir to keep")
    ap.add_argument("--acc", type=float, help="Locked Acc@1")
    ap.add_argument("--keep-wave", help="Active wave id, e.g. w17 (ACTIVE cities)")
    ap.add_argument(
        "--results-prefix",
        action="append",
        default=[],
        help="results dir prefix to keep (e.g. climb_bpr020_soft010_w17_)",
    )
    ap.add_argument("--status", choices=["ACTIVE", "STOPPED"], help="Set city status")
    ap.add_argument("--foundation", action="append", default=[], help="Extra foundation yaml to keep")
    ap.add_argument("--apply", action="store_true", help="Actually prune (else dry-run prune)")
    ap.add_argument("--summarize-only", action="store_true", help="Refresh summaries; do not delete")
    ap.add_argument("--register-only", action="store_true", help="Only edit climb_retention.yaml")
    args = ap.parse_args()

    data = load_retention()

    if args.lock_config or args.keep_wave or args.status or args.results_tag:
        if not args.cities or len(args.cities) != 1:
            ap.error("--lock-config/--keep-wave/--status/--results-tag need exactly one --city")
        city = args.cities[0]
        data = register_lock(
            city,
            lock_config=args.lock_config or (data["cities"][city].get("lock_config") or f"configs/{city}/win.yaml"),
            results_tag=args.results_tag,
            acc=args.acc,
            keep_wave=args.keep_wave,
            status=args.status,
            foundation_configs=args.foundation,
        )
        if args.results_prefix:
            c = data["cities"][city]
            prefs = set(c.get("keep_results_prefixes") or [])
            prefs.update(args.results_prefix)
            # drop placeholder substring prefixes
            prefs = {p for p in prefs if not p.startswith("_")}
            c["keep_results_prefixes"] = sorted(prefs)
            save_retention(data)
        print(f"updated {RETENTION_PATH} for {city}")

    if args.register_only:
        return

    cities = list(args.cities or [])
    if args.all or not cities:
        cities = list((data.get("cities") or {}).keys())
    cities = list(dict.fromkeys(cities))

    dry = not args.apply
    mode = "DRY-RUN" if dry else "APPLY"
    print(f"=== sync_climb_retention ({mode}) cities={cities} ===")
    if dry:
        print("Pass --apply to delete; without it prune tools run with --dry-run.")
    for city in cities:
        print(f"\n##### sync {city} #####", flush=True)
        apply_city(city, data, dry_run=dry, summarize_only=args.summarize_only)


if __name__ == "__main__":
    main()
