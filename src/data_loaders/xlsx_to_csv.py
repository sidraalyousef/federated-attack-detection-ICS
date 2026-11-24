"""Convert the iTrust SWaT ``.xlsx`` downloads to the CSVs that ``prepare_swat`` reads.

iTrust ships the 2015 SWaT release as Excel workbooks. ``preprocessing.py``
expects ``SWaT_Dataset_Normal_v1.csv`` and ``SWaT_Dataset_Attack_v0.csv`` in
``data/raw/swat/``. This script streams each workbook (``read_only`` mode keeps
memory low even though each file has ~500k rows) and writes the CSV with the
same rows, in the same order. Real date cells are written in the text format
``load_swat`` parses (``%d/%m/%Y %I:%M:%S %p``); text cells are left as they are.

Run from the repo root (needs ``openpyxl``; each file takes a few minutes):

    python -m src.data_loaders.xlsx_to_csv "<folder with the .xlsx files>"
    python -m src.data_loaders.xlsx_to_csv "<folder>" --out data/raw/swat

The CSVs go to ``data/raw/swat`` by default, which is gitignored: never commit
the raw data (SWaT is released under the iTrust data agreement).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from .preprocessing import ATTACK_FILE, NORMAL_FILE, TIME_FORMAT

# xlsx name (as downloaded) -> csv name expected by prepare_swat
FILES = {
    "SWaT_Dataset_Normal_v1.xlsx": NORMAL_FILE,
    "SWaT_Dataset_Attack_v0.xlsx": ATTACK_FILE,
}


def _cell(value):
    if isinstance(value, datetime):
        return value.strftime(TIME_FORMAT)
    return "" if value is None else value


def convert(xlsx_path: Path, csv_path: Path, log_every: int = 100_000) -> int:
    """Stream the first sheet of ``xlsx_path`` into ``csv_path``; returns the row count."""
    from openpyxl import load_workbook  # imported here: only needed for this step

    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    n = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in ws.iter_rows(values_only=True):
            writer.writerow([_cell(v) for v in row])
            n += 1
            if n % log_every == 0:
                print(f"  {n:,} rows", flush=True)
    wb.close()
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", type=Path, help="folder that holds the SWaT .xlsx files")
    p.add_argument("--out", type=Path, default=Path("data/raw/swat"))
    p.add_argument("--force", action="store_true", help="overwrite existing CSVs")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for xlsx_name, csv_name in FILES.items():
        src, dst = args.src / xlsx_name, args.out / csv_name
        if not src.exists():
            raise SystemExit(f"missing {src}")
        if dst.exists() and not args.force:
            print(f"{dst} exists, skipping (use --force to overwrite)")
            continue
        print(f"{src.name} -> {dst}", flush=True)
        print(f"  done, {convert(src, dst):,} rows", flush=True)


if __name__ == "__main__":
    main()
