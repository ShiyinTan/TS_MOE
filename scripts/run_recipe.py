#!/usr/bin/env python3
"""Launch a committed Base/WIN/Peak recipe config.

Examples:
  python scripts/run_recipe.py tokyo win
  python scripts/run_recipe.py beijing peak -- --device cuda:1
  python scripts/run_recipe.py --config configs/sydney/base.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="explicit config path (skips city/recipe)")
    p.add_argument("city", nargs="?", help="Massive-STEPS city, e.g. tokyo")
    p.add_argument("recipe", nargs="?", choices=("base", "win", "peak"),
                   help="recipe name")
    p.add_argument("passthrough", nargs=argparse.REMAINDER,
                   help="extra CLI after -- (overrides config)")
    args = p.parse_args(argv)
    if args.config:
        cfg = Path(args.config)
    else:
        if not args.city or not args.recipe:
            p.error("pass city+recipe or --config")
        cfg = ROOT / "configs" / args.city / f"{args.recipe}.yaml"
    if not cfg.is_file():
        raise SystemExit(f"config not found: {cfg}")
    extra = list(args.passthrough)
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd = [sys.executable, "-m", "tsmoe", "train", "--config", str(cfg), *extra]
    print("+", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
