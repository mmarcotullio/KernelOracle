"""
Convert TCN trace CSVs to the format expected by baseline/utils.preprocess_data().

TCN trace columns:  timestamp, time_diff, task_name, pid, cpu, prev_state,
                    prev_comm, workload_type, trace_id

Baseline expected:  task_code, time, name, pid
  - task_code: dropped by preprocess_data(), so a dummy value is used
  - time:      raw timestamp; preprocess_data() calls np.diff() on this
  - name:      task name; one-hot encoded by preprocess_data()
  - pid:       dropped by preprocess_data()

Usage:
    python convert_to_baseline.py <input.csv> <output.csv>

Example:
    python convert_to_baseline.py data/traces/train.csv data/train_baseline.csv
"""

import sys
import argparse
import pandas as pd


def convert(input_path: str, output_path: str, chunksize: int = 500_000) -> None:
    print(f"Converting {input_path} -> {output_path}")

    first_chunk = True
    rows_written = 0

    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        out = pd.DataFrame({
            "task_code": 0,
            "time": chunk["timestamp"],
            "name": chunk["task_name"],
            "pid": chunk["pid"],
        })

        out.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        rows_written += len(out)
        first_chunk = False
        print(f"  {rows_written:,} rows written", end="\r")

    print(f"\nDone. {rows_written:,} rows written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TCN trace CSV to baseline-compatible format."
    )
    parser.add_argument("input", help="Path to TCN trace CSV (e.g. data/traces/train.csv)")
    parser.add_argument("output", help="Path for output CSV (e.g. data/train_baseline.csv)")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Rows per chunk for memory-efficient processing (default: 500000)",
    )
    args = parser.parse_args()

    convert(args.input, args.output, chunksize=args.chunksize)


if __name__ == "__main__":
    main()
