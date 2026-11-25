#!/usr/bin/env python3
"""Summarize ``results/<city>/<run>/`` metrics, then delete non-kept run dirs.

Run **on the GPU box** (or any machine that holds ``results/``) so agents do not
stream per-folder deletes through the cloud chat.

Keeps forever (never deleted):
  - root files: ``climb_path.md``, ``settings_summary.*``, ``mean_std.*``, ``README.md``, …
  - run dirs named in ``--keep`` / parsed from ``climb_path.md`` / ``--keep-prefix``
  - optionally top-N Acc runs (``--keep-top``)

Example (dry-run first)::

    python scripts/summarize_and_prune_results.py --city beijing --sthgcn 0.453 --dry-run
    python scripts/summarize_and_prune_results.py --city beijing --sthgcn 0.453

    # active MEL wave: summarize all, keep WIN + in-flight w17*
    python scripts/summarize_and_prune_results.py --city melbourne --sthgcn 0.168 \\
        --keep-prefix climb_bpr020_soft010_w17_ --keep-top 5

Does **not** rewrite a hand-written ``climb_path.md`` unless ``--rewrite-climb-path``.
Summaries always refresh from whatever run dirs still exist at summarize time.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from city_results_bundle import (  # noqa: E402
    load_row,
    write_mean_std,
    write_readme,
    write_settings_summary,
)
from climb_retention import (  # noqa: E402
    city_keep_prefixes,
    city_keep_results,
    sthgcn_map,
)

DEFAULT_STHGCN = sthgcn_map()


def default_keep(city: str) -> set[str]:
    try:
        return city_keep_results(city)
    except Exception:
        return set()


def default_prefixes(city: str) -> list[str]:
    try:
        return [p for p in city_keep_prefixes(city) if not p.startswith("_")]
    except Exception:
        return []

KEEP_ROOT_NAMES = {
    "climb_path.md",
    "settings_summary.md",
    "settings_summary.csv",
    "mean_std.md",
    "mean_std.csv",
    "README.md",
}


_SKIP_TAGS = {
    "eval_test",
    "known_only",
    "cache_abl_tgt_last",
    "cache_tgt_all_eval_last",
    "configs",
    "docs",
    "results",
    "runs",
    "scripts",
}


def parse_keep_from_climb_path(path: Path) -> set[str]:
    """Extract keep tags from climb_path backticks (prefer results/<city>/<tag>/…)."""
    if not path.exists():
        return set()
    text = path.read_text(errors="ignore")
    tags: set[str] = set()
    # Prefer explicit results paths: results/<city>/<tag>/...
    for m in re.finditer(r"`results/[A-Za-z0-9_.-]+/([A-Za-z0-9][A-Za-z0-9_.-]{2,})(?:/[^`]*)?`", text):
        tags.add(m.group(1))
    # Also accept long climb_* / mc_* / on_* / hc_* / tune_* / win run-like tags
    for m in re.finditer(
        r"`((?:climb|mc|on|hc|tune|cal|abl)_[A-Za-z0-9_.-]{3,}|win|peak)`",
        text,
    ):
        tags.add(m.group(1))
    return {t for t in tags if t not in _SKIP_TAGS and not t.endswith((".yaml", ".md", ".csv", ".json", ".py"))}


def summarize(city_dir: Path, city: str, sthgcn: float, *, allow_empty: bool = False) -> list[dict]:
    run_dirs = sorted(p for p in city_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    rows = [load_row(d) for d in run_dirs]
    ok_rows = [r for r in rows if r["status"] == "ok"]
    # Do not clobber an existing rich summary with an empty rewrite (e.g. local checkout
    # that only has climb_path.md and no run dirs).
    if not ok_rows and not allow_empty:
        existing = city_dir / "settings_summary.csv"
        if existing.exists() and existing.stat().st_size > 50:
            print(f"  skip summary rewrite: 0 ok dirs but {existing.name} already present")
            return []
    ok = write_settings_summary(city_dir, city, sthgcn, rows)
    write_mean_std(city_dir, city, ok)
    win = ok[0] if ok else None
    write_readme(city_dir, city, sthgcn, len(ok), win)
    return ok


def resolve_keep(
    city: str,
    city_dir: Path,
    ok: list[dict],
    keep: list[str],
    keep_prefix: list[str],
    keep_top: int,
) -> set[str]:
    out: set[str] = set(default_keep(city))
    out |= parse_keep_from_climb_path(city_dir / "climb_path.md")
    out |= set(keep)
    prefs = list(keep_prefix) + default_prefixes(city)
    for pref in prefs:
        for p in city_dir.iterdir():
            if p.is_dir() and p.name.startswith(pref):
                out.add(p.name)
    if keep_top > 0 and ok:
        for r in ok[:keep_top]:
            out.add(r["run"])
    return out


def prune(city_dir: Path, keep: set[str], dry_run: bool) -> tuple[int, int]:
    deleted = 0
    kept = 0
    for p in list(city_dir.iterdir()):
        if p.is_file():
            # never delete root summary / climb files
            continue
        if not p.is_dir() or p.name.startswith("."):
            continue
        if p.name in keep:
            kept += 1
            continue
        if dry_run:
            print(f"  DRY-RUN delete {p}")
            deleted += 1
            continue
        shutil.rmtree(p)
        deleted += 1
    return deleted, kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", action="append", dest="cities", help="City name under results/ (repeatable)")
    ap.add_argument("--all-stopped", action="store_true", help="tokyo shanghai beijing palembang")
    ap.add_argument("--sthgcn", type=float, default=None, help="Override STHGCN Acc@1 (single-city)")
    ap.add_argument("--keep", action="append", default=[], help="Extra run dir name to keep")
    ap.add_argument("--keep-prefix", action="append", default=[], help="Keep run dirs with this prefix")
    ap.add_argument("--keep-top", type=int, default=3, help="Also keep top-N Acc runs (default 3)")
    ap.add_argument("--summarize-only", action="store_true", help="Write summaries; do not delete")
    ap.add_argument("--prune-only", action="store_true", help="Delete without refreshing summaries")
    ap.add_argument("--dry-run", action="store_true", help="Print deletes; write nothing destructive")
    ap.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    cities = list(args.cities or [])
    if args.all_stopped:
        cities.extend(["tokyo", "shanghai", "beijing", "palembang"])
    cities = list(dict.fromkeys(cities))
    if not cities:
        ap.error("pass --city … or --all-stopped")

    for city in cities:
        city_dir = args.results_root / city
        if not city_dir.is_dir():
            print(f"SKIP missing {city_dir}")
            continue
        sthgcn = args.sthgcn if args.sthgcn is not None else DEFAULT_STHGCN.get(city)
        if sthgcn is None:
            ap.error(f"--sthgcn required for city={city}")

        print(f"=== {city} (sthgcn={sthgcn}) ===")
        before = sum(1 for p in city_dir.iterdir() if p.is_dir())
        ok: list[dict] = []
        if args.prune_only:
            rows = [load_row(d) for d in sorted(p for p in city_dir.iterdir() if p.is_dir())]
            ok = sorted([r for r in rows if r["status"] == "ok"], key=lambda r: -float(r["acc"]))
        else:
            ok = summarize(city_dir, city, sthgcn)
            print(f"  summarized {len(ok)} ok / {before} dirs → settings_summary.* mean_std.*")

        if args.summarize_only:
            continue

        keep = resolve_keep(city, city_dir, ok, args.keep, args.keep_prefix, args.keep_top)
        print(f"  keep ({len(keep)}): {sorted(keep)[:12]}{' …' if len(keep) > 12 else ''}")
        deleted, kept = prune(city_dir, keep, dry_run=args.dry_run)
        after = sum(1 for p in city_dir.iterdir() if p.is_dir())
        root_files = sorted(p.name for p in city_dir.iterdir() if p.is_file())
        print(f"  dirs {before} → {after} · deleted={deleted} kept_dirs={kept}")
        print(f"  root_files={root_files}")


if __name__ == "__main__":
    main()
