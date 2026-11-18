"""Generate small, synthetic event CSVs in the official Massive-STEPS format."""
import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

FIELDS = ("trail_id", "user_id", "venue_id", "latitude", "longitude", "name", "address",
          "venue_category", "venue_category_id", "venue_category_id_code", "venue_city",
          "venue_city_latitude", "venue_city_longitude", "venue_country", "timestamp")


def generate(root):
    directory = Path(root) / "demo"
    directory.mkdir(parents=True, exist_ok=True)
    for s, split in enumerate(("train", "validation", "test")):
        with (directory / f"demo_checkins_{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for trail in range(24 if split == "train" else 6):
                start = datetime(2018, 1, 1) + timedelta(days=s * 30 + trail, hours=trail % 24)
                for j in range(3 + trail % 4):
                    poi = (trail + j + (trail % 3) * j) % 12
                    if split != "train" and trail == 0 and j == 2:
                        poi = 99  # held-out target, used to verify out-of-catalog reporting
                    cat = poi % 4
                    writer.writerow(dict(trail_id=f"{split}_{trail}", user_id=trail % 4, venue_id=poi,
                                         latitude=35 + poi / 100, longitude=139 + poi / 100,
                                         venue_category_id=f"cat-{cat}", venue_category_id_code=cat,
                                         timestamp=(start + timedelta(minutes=15 * j)).isoformat(sep=" ")))
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    print(generate(parser.parse_args().data_root))
