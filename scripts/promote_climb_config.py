#!/usr/bin/env python3
"""Promote a climb WIN config into git after a real Acc lock (not densify waves).

Wave / intermediate YAMLs stay GPU-local (gitignored). Only call this when Acc
beats the lock and you want the WIN recipe on main::

    python scripts/promote_climb_config.py \\
      --city melbourne \\
      --config configs/melbourne/climb_..._win.yaml \\
      --results-tag climb_... \\
      --acc 0.1725 \\
      --keep-wave w18

Then commit the promoted YAML + climb_retention.yaml (and run sync on GPU).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from climb_retention import register_lock  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    ap.add_argument("--config", required=True, help="Path to WIN yaml (must exist)")
    ap.add_argument("--results-tag", default=None)
    ap.add_argument("--acc", type=float, required=True)
    ap.add_argument("--keep-wave", default=None)
    ap.add_argument("--results-prefix", action="append", default=[])
    ap.add_argument("--foundation", action="append", default=[])
    ap.add_argument("--status", choices=["ACTIVE", "STOPPED"], default="ACTIVE")
    ap.add_argument("--git-add", action="store_true", help="git add -f the WIN yaml + retention")
    args = ap.parse_args()

    cfg = Path(args.config)
    if not cfg.is_file():
        # allow city-relative
        alt = ROOT / "configs" / args.city / cfg.name
        if alt.is_file():
            cfg = alt
        else:
            raise SystemExit(f"missing config: {args.config}")

    # WIN files should not match wave gitignore; if somehow ignored, -f needed.
    data = register_lock(
        args.city,
        lock_config=str(cfg.relative_to(ROOT)) if cfg.is_absolute() else str(cfg),
        results_tag=args.results_tag,
        acc=args.acc,
        keep_wave=args.keep_wave,
        status=args.status,
        foundation_configs=args.foundation,
    )
    if args.results_prefix:
        c = data["cities"][args.city]
        prefs = set(c.get("keep_results_prefixes") or [])
        prefs.update(p for p in args.results_prefix if not p.startswith("_"))
        c["keep_results_prefixes"] = sorted(prefs)
        from climb_retention import save_retention

        save_retention(data)

    print(f"registered lock {args.city} Acc={args.acc} cfg={cfg.name}")
    if args.git_add:
        rel = str(cfg.relative_to(ROOT)) if cfg.is_absolute() else str(cfg)
        cmds = [
            ["git", "add", "-f", rel],
            ["git", "add", "configs/climb_retention.yaml"],
        ]
        for cmd in cmds:
            print("+", " ".join(cmd))
            subprocess.check_call(cmd, cwd=ROOT)
        print("staged; commit + push when ready (do NOT add *_wNN_* densify yamls)")


if __name__ == "__main__":
    main()
