"""Download the three public Massive-STEPS event CSVs for one city."""
import argparse
import csv
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

CITIES = ("bandung", "beijing", "istanbul", "jakarta", "kuwait_city", "melbourne", "moscow",
          "new_york", "palembang", "petaling_jaya", "sao_paulo", "shanghai", "sydney", "tangerang", "tokyo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=CITIES, default="tokyo")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--revision", default="main", help="Hugging Face revision or commit SHA")
    args = parser.parse_args()
    slug = args.city.replace("_", "-").title()
    directory = args.data_root / args.city
    directory.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        name = f"{args.city}_checkins_{split}.csv"
        target = directory / name
        if target.exists():
            print("Using existing", target)
            continue
        url = (f"https://huggingface.co/datasets/CRUISEResearchGroup/Massive-STEPS-{slug}"
               f"/resolve/{args.revision}/{name}")
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as tmp:
            staged = Path(tmp.name)
        try:
            with urlopen(url, timeout=120) as src, staged.open("wb") as dest:
                shutil.copyfileobj(src, dest)
            with staged.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if "venue_id" not in (reader.fieldnames or ()) or next(reader, None) is None:
                    raise ValueError(f"{url} did not return a check-in CSV")
            staged.replace(target)
            print("Downloaded", target)
        finally:
            staged.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
