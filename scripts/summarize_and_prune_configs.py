#!/usr/bin/env python3
"""Summarize climb configs → CSV/MD index, then delete obsolete wave YAMLs.

Run **on the GPU box or locally** (no chat uploads)::

    # dry-run (default safe): only wave configs ``*_w12_*`` etc.
    python scripts/summarize_and_prune_configs.py --all-stopped --dry-run
    python scripts/summarize_and_prune_configs.py --all-stopped

    # Melbourne active wave: keep w17 (+ WIN + foundations via extends)
    python scripts/summarize_and_prune_configs.py --city melbourne \\
        --keep-wave w17 --dry-run
    python scripts/summarize_and_prune_configs.py --city melbourne --keep-wave w17

    # STOPPED city: drop all climb waves; keep locked WIN + extends chain
    python scripts/summarize_and_prune_configs.py --city beijing --aggressive

Writes (kept forever)::

    configs/<city>/_index/configs_summary.csv
    configs/<city>/_index/configs_summary.md
    configs/<city>/_index/pruned_manifest.csv   # what was deleted (append)

Never deletes: ``base.yaml`` / ``peak.yaml`` / ``win.yaml`` / ``recipes/`` /
``_index/``. Active ``--keep-wave`` and locked WIN tags stay.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyYAML required: pip install pyyaml") from e

ROOT = Path(__file__).resolve().parents[1]

CORE_NAMES = {"base.yaml", "peak.yaml", "win.yaml"}

# Locked WIN / ladder configs to retain for STOPPED / active cities.
DEFAULT_KEEP = {
    "tokyo": {"base.yaml", "peak.yaml", "win.yaml"},
    "shanghai": {
        "base.yaml",
        "peak.yaml",
        "win.yaml",
        "climb_soft005.yaml",
        "climb_soft005_filter_last.yaml",
    },
    "beijing": {
        "base.yaml",
        "peak.yaml",
        "win.yaml",
        "climb_s5g20_all_gauss020_wd005_j12p10_pmi015_win.yaml",
        "climb_s5g20_all_gauss020_wd005.yaml",
        "climb_s5g20_all_gauss020.yaml",
        "climb_s5g20_all_gauss015.yaml",
        "climb_ep2_trbias_soft005_geo020.yaml",
    },
    "melbourne": {
        "base.yaml",
        "peak.yaml",
        "win.yaml",
        "climb_bpr020_soft010.yaml",
        "climb_bpr020_soft010_tgt_soft0078_session_bpr205_jump010_pmi05_win.yaml",
        "climb_bpr020_soft010_tgt_soft0078_session_bpr205_jump010_win.yaml",
    },
    "palembang": {
        "base.yaml",
        "peak.yaml",
        "win.yaml",
        "climb_tsb20_bpr020_filter.yaml",
        "climb_tsb20_bpr020_filter_soft0754_bpr025_session_seed55.yaml",
    },
}

WAVE_RE = re.compile(r"_w(\d+[a-z]?)_")

SUMMARY_KEYS = [
    "file",
    "recipe_id",
    "extends",
    "cache",
    "out",
    "description",
    "seed",
    "epochs",
    "soft_target_beta",
    "lambda_bpr",
    "jump_length_bias",
    "pmi_score_bias",
    "pop_score_bias",
    "geo_score_bias",
    "gauss_geo_bias",
    "weight_decay",
    "wave",
    "bytes",
]


def wave_of(name: str) -> str | None:
    m = WAVE_RE.search(name)
    return m.group(1) if m else None


def load_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def row_for(path: Path) -> dict:
    data = load_yaml(path)
    w = wave_of(path.name)
    return {
        "file": path.name,
        "recipe_id": data.get("recipe_id", ""),
        "extends": data.get("extends", ""),
        "cache": data.get("cache", ""),
        "out": data.get("out", ""),
        "description": (data.get("description") or "").replace("\n", " ")[:240],
        "seed": data.get("seed", ""),
        "epochs": data.get("epochs", ""),
        "soft_target_beta": data.get("soft_target_beta", ""),
        "lambda_bpr": data.get("lambda_bpr", ""),
        "jump_length_bias": data.get("jump_length_bias", ""),
        "pmi_score_bias": data.get("pmi_score_bias", ""),
        "pop_score_bias": data.get("pop_score_bias", ""),
        "geo_score_bias": data.get("geo_score_bias", ""),
        "gauss_geo_bias": data.get("gauss_geo_bias", ""),
        "weight_decay": data.get("weight_decay", ""),
        "wave": w or "",
        "bytes": path.stat().st_size,
    }


def write_summary(city_dir: Path, rows: list[dict]) -> Path:
    idx = city_dir / "_index"
    idx.mkdir(parents=True, exist_ok=True)
    csv_path = idx / "configs_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_KEYS)
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["file"]):
            w.writerow({k: r.get(k, "") for k in SUMMARY_KEYS})

    waves: dict[str, int] = {}
    for r in rows:
        key = r["wave"] or "non-wave"
        waves[key] = waves.get(key, 0) + 1
    md = [
        f"# {city_dir.name} configs index",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**YAML files indexed:** {len(rows)}  ",
        "",
        "Full table: [`configs_summary.csv`](configs_summary.csv).",
        "",
        "## Wave counts",
        "",
        "| wave | # configs |",
        "| --- | ---: |",
    ]
    for k, n in sorted(waves.items(), key=lambda kv: (-kv[1], kv[0])):
        md.append(f"| `{k}` | {n} |")
    md += [
        "",
        "## Retention",
        "",
        "After prune: keep `base`/`peak`/`win`, locked WIN tags, `--keep-wave`, "
        "and anything in the extends chain of kept files. "
        "Re-run this script anytime; summaries refresh from remaining YAMLs.",
        "",
    ]
    (idx / "configs_summary.md").write_text("\n".join(md))
    return csv_path


def append_pruned_manifest(city_dir: Path, deleted: list[dict]) -> None:
    if not deleted:
        return
    idx = city_dir / "_index"
    idx.mkdir(parents=True, exist_ok=True)
    path = idx / "pruned_manifest.csv"
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["pruned_at", *SUMMARY_KEYS],
        )
        if write_header:
            w.writeheader()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for r in deleted:
            w.writerow({"pruned_at": ts, **{k: r.get(k, "") for k in SUMMARY_KEYS}})


def extends_target(city_dir: Path, extends: str) -> Path | None:
    if not extends:
        return None
    # ./foo.yaml or foo.yaml or ../recipes/x.yaml
    p = (city_dir / extends).resolve()
    if p.exists():
        return p
    return None


def closure_keep(city_dir: Path, seeds: set[str]) -> set[str]:
    """Expand keep set with extends chain (same-city files only)."""
    keep = set(seeds)
    changed = True
    while changed:
        changed = False
        for name in list(keep):
            path = city_dir / name
            if not path.exists():
                continue
            data = load_yaml(path)
            ext = data.get("extends") or ""
            tgt = extends_target(city_dir, ext)
            if tgt is None:
                continue
            try:
                rel = tgt.relative_to(city_dir.resolve())
            except ValueError:
                continue  # recipes/ etc.
            if len(rel.parts) == 1 and rel.name not in keep:
                keep.add(rel.name)
                changed = True
    return keep


def parse_keep_from_climb_path(repo: Path, city: str) -> set[str]:
    path = repo / "results" / city / "climb_path.md"
    if not path.exists():
        return set()
    text = path.read_text(errors="ignore")
    tags: set[str] = set()
    for m in re.finditer(
        r"`(?:configs/[A-Za-z0-9_.-]+/)?([A-Za-z0-9][A-Za-z0-9_.-]+\.yaml)`",
        text,
    ):
        tags.add(m.group(1))
    return tags


def should_delete(
    name: str,
    *,
    keep: set[str],
    keep_waves: set[str],
    aggressive: bool,
) -> bool:
    if name in CORE_NAMES or name in keep:
        return False
    if name.startswith("_"):
        return False
    w = wave_of(name)
    if w is not None:
        if w in keep_waves:
            return False
        return True  # obsolete wave config
    # non-wave
    if not aggressive:
        return False
    # aggressive: drop climb_* not in keep
    if name.startswith("climb_") or name.startswith("cal_") or name.startswith("tune_"):
        return name not in keep
    return False


def prune_city(
    repo: Path,
    city: str,
    *,
    keep_extra: list[str],
    keep_waves: list[str],
    aggressive: bool,
    dry_run: bool,
    summarize_only: bool,
) -> None:
    city_dir = repo / "configs" / city
    if not city_dir.is_dir():
        print(f"SKIP missing {city_dir}")
        return

    yamls = sorted(p for p in city_dir.glob("*.yaml") if p.is_file())
    rows = [row_for(p) for p in yamls]
    write_summary(city_dir, rows)
    print(f"=== {city}: indexed {len(rows)} yamls → configs/{city}/_index/")

    if summarize_only:
        return

    keep = set(DEFAULT_KEEP.get(city, CORE_NAMES))
    keep |= set(keep_extra)
    keep |= parse_keep_from_climb_path(repo, city)
    keep |= CORE_NAMES
    # Always keep any *current* locked file ending with _win.yaml that is listed in DEFAULT_KEEP
    # plus explicit --keep-wave files' non-wave parents via closure
    keep_waves_set = {w.lstrip("w") if w.startswith("w") and w[1:].isalnum() else w for w in keep_waves}
    # normalize: accept w17 or 17
    keep_waves_norm: set[str] = set()
    for w in keep_waves:
        keep_waves_norm.add(w[1:] if w.startswith("w") else w)

    # seed keep with all non-wave files that kept waves extend? handled by closure after adding wave files
    for p in yamls:
        w = wave_of(p.name)
        if w is not None and w in keep_waves_norm:
            keep.add(p.name)

    keep = closure_keep(city_dir, keep)

    to_delete: list[Path] = []
    for p in yamls:
        if should_delete(
            p.name,
            keep=keep,
            keep_waves=keep_waves_norm,
            aggressive=aggressive,
        ):
            to_delete.append(p)

    print(f"  keep={len(keep)} delete={len(to_delete)} aggressive={aggressive} waves_keep={sorted(keep_waves_norm)}")
    if dry_run:
        for p in to_delete[:12]:
            print(f"  DRY-RUN delete {p.name}")
        if len(to_delete) > 12:
            print(f"  DRY-RUN … +{len(to_delete) - 12} more")
        return

    deleted_rows = []
    for p in to_delete:
        deleted_rows.append(row_for(p))
        p.unlink()
    append_pruned_manifest(city_dir, deleted_rows)
    # refresh summary from remaining
    left = sorted(x for x in city_dir.glob("*.yaml") if x.is_file())
    write_summary(city_dir, [row_for(p) for p in left])
    print(f"  deleted {len(deleted_rows)}; remaining {len(left)} → index refreshed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", action="append", dest="cities")
    ap.add_argument("--all-stopped", action="store_true", help="tokyo shanghai beijing palembang")
    ap.add_argument("--keep", action="append", default=[], help="Extra yaml filename to keep")
    ap.add_argument("--keep-wave", action="append", default=[], help="Keep wave id, e.g. w17 or 17")
    ap.add_argument(
        "--aggressive",
        action="store_true",
        help="Also delete non-wave climb_/cal_/tune_ not in keep/extends",
    )
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--repo", type=Path, default=ROOT)
    args = ap.parse_args()

    cities = list(args.cities or [])
    if args.all_stopped:
        cities.extend(["tokyo", "shanghai", "beijing", "palembang"])
    cities = list(dict.fromkeys(cities))
    if not cities:
        ap.error("pass --city … or --all-stopped")

    for city in cities:
        prune_city(
            args.repo,
            city,
            keep_extra=args.keep,
            keep_waves=args.keep_wave,
            aggressive=args.aggressive,
            dry_run=args.dry_run,
            summarize_only=args.summarize_only,
        )


if __name__ == "__main__":
    main()
