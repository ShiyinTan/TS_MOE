#!/usr/bin/env python3
"""Verify Massive-STEPS check-in CSVs match what `python -m tsmoe prepare` reads.

Does not invent prepare flags. Mirrors the layout resolution in `tsmoe/prepare.py`
and the required columns in `tsmoe/data.py` (stdlib only; no torch import).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SPLITS = ("train", "validation", "test")
REQUIRED = {"trail_id", "user_id", "venue_id", "latitude", "longitude", "timestamp"}
CATEGORY = {"venue_category_id_code", "venue_category_id"}


def resolve_root(data_root: Path, city: str) -> Path:
    """Same rule as prepare.py: prefer flat root, else <data-root>/<city>/."""
    flat = data_root / f"{city}_checkins_train.csv"
    nested = data_root / city / f"{city}_checkins_train.csv"
    if flat.is_file():
        return data_root
    if nested.is_file():
        return data_root / city
    raise FileNotFoundError(
        f"missing {flat.name} under {data_root} or {data_root / city}. "
        "Download with: python examples/download_city.py --city tokyo --data-root data"
    )


def check_split(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing file: {path}"]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED - fields
        if missing:
            errors.append(f"{path.name}: missing columns {sorted(missing)}")
        if not fields & CATEGORY:
            errors.append(
                f"{path.name}: need venue_category_id_code and/or venue_category_id "
                f"(Foursquare-style category fields used by the design doc)"
            )
        row = next(reader, None)
        if row is None:
            errors.append(f"{path.name}: file has a header but no data rows")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="same --data-root you will pass to prepare")
    parser.add_argument("--city", required=True, help="city filename prefix, e.g. tokyo")
    args = parser.parse_args()
    try:
        root = resolve_root(args.data_root, args.city)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    paths = {split: root / f"{args.city}_checkins_{split}.csv" for split in SPLITS}
    errors: list[str] = []
    for path in paths.values():
        errors.extend(check_split(path))
    if errors:
        print("Layout check failed:", file=sys.stderr)
        for item in errors:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("OK: prepare can read these files under", root)
    for split, path in paths.items():
        print(f"  {split}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
