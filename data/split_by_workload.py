#!/usr/bin/env python3
"""Split a trace CSV into per-workload-type CSVs."""

import argparse
import csv
import os
from collections import defaultdict


def split(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    writers = {}
    file_handles = {}

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            wt = row["workload_type"]
            if wt not in writers:
                out_path = os.path.join(output_dir, f"{wt}.csv")
                fh = open(out_path, "w", newline="")
                file_handles[wt] = fh
                writers[wt] = csv.DictWriter(fh, fieldnames=fieldnames)
                writers[wt].writeheader()
            writers[wt].writerow(row)

    for fh in file_handles.values():
        fh.close()

    print(f"Split into {len(writers)} files in {output_dir}:")
    for wt in sorted(writers):
        print(f"  {wt}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split trace CSV by workload_type")
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: same dir as input)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input_csv))
    split(args.input_csv, out_dir)
