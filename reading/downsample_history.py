#!/usr/bin/env python3
"""
Downsample existing history parquet files by keeping every nth row.
This reduces disk usage while preserving data at lower resolution.

Ex: python reading/downsample_history.py --dir /user/home/il22158/work/vEcoli/out/gene_ko_441imported_2seeds/history --n 5
"""

import pandas as pd
from pathlib import Path
import argparse
import subprocess


def print_folder_size(history_dir):
    try:
        result = subprocess.run(
            ["du", "-sh", str(history_dir)], capture_output=True, text=True
        )
        print(f"Actual history folder size: {result.stdout.strip()}")
    except Exception as e:
        print(f"Could not determine folder size: {e}")


def downsample_history(history_dir, n):
    pq_files = list(history_dir.rglob("*.pq"))
    print(f"Found {len(pq_files)} parquet files to process")
    print_folder_size(history_dir)
    print(f"Starting downsampling (keeping every {n}th row)...")
    total_saved = 0
    for i, pq_file in enumerate(pq_files):
        if i % 1000 == 0:
            print(f"Progress: {i}/{len(pq_files)} files processed...")
        try:
            original_size = pq_file.stat().st_size
            df = pd.read_parquet(pq_file)
            df_sampled = df.iloc[::n]
            df_sampled.to_parquet(pq_file)
            new_size = pq_file.stat().st_size
            saved = original_size - new_size
            total_saved += saved
        except Exception as e:
            print(f"Error processing {pq_file}: {e}")
    print(f"\nDone! Estimated space saved: {total_saved / 1024**3:.1f} GB")
    print_folder_size(history_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample all parquet files in a directory by keeping every n-th row."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to history directory containing parquet files.",
    )
    parser.add_argument(
        "--n", type=int, default=5, help="Keep every n-th row (default: 5)"
    )
    args = parser.parse_args()
    downsample_history(Path(args.dir), args.n)
