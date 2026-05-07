#!/usr/bin/env python3
"""
Aggregate gene knockout simulation results into a summary CSV.
Reads success files from the simulation output and creates a format similar to ko_runs_with_success_generation.csv

Usage:
    python aggregate_knockout_results.py --project <project_name> [--output-dir <dir>] [--output-file <name.csv>]

Example:
    python aggregate_knockout_results.py --project gene_knockout_p_list
    python aggregate_knockout_results.py --project gene_knockout_test --output-dir /custom/path --output-file summary.csv
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
from pathlib import Path
from collections import defaultdict


def get_success_generations(base_path, project_name):
    """
    Extract success generation information from success directory structure.
    Args:
        base_path: Root output directory containing success subdirectory
        project_name: Project name (e.g., 'gene_knockout_p_list')
    Returns dict: {(variant, lineage_seed): max_generation}
    """
    success_dir = os.path.join(base_path, "success", f"experiment_id={project_name}")
    success_info = defaultdict(lambda: {"max_gen": 0, "generations": set()})

    # Find all success files
    pattern = os.path.join(
        success_dir, "variant=*", "lineage_seed=*", "generation=*", "agent_id=*", "s.pq"
    )
    success_files = glob.glob(pattern)

    print(f"Found {len(success_files)} success files")

    for filepath in success_files:
        # Extract variant, lineage_seed, generation from path
        parts = filepath.split(os.sep)

        variant_str = [p for p in parts if p.startswith("variant=")]
        lineage_seed_str = [p for p in parts if p.startswith("lineage_seed=")]
        generation_str = [p for p in parts if p.startswith("generation=")]

        if variant_str and lineage_seed_str and generation_str:
            variant = int(variant_str[0].split("=")[1])
            lineage_seed = int(lineage_seed_str[0].split("=")[1])
            generation = int(generation_str[0].split("=")[1])

            key = (variant, lineage_seed)
            success_info[key]["max_gen"] = max(success_info[key]["max_gen"], generation)
            success_info[key]["generations"].add(generation)

    return success_info


def enumerate_runs(base_path):
    """
    Enumerate all variant/lineage_seed runs present under base_path.
    Returns list of tuples: (variant:int, lineage_seed:int, generation_dirs:list)
    """
    runs = []
    variant_pattern = os.path.join(base_path, "variant=*")
    variant_dirs = glob.glob(variant_pattern)

    for vdir in variant_dirs:
        try:
            variant = int(os.path.basename(vdir).split("=")[1])
        except Exception:
            continue

        seed_pattern = os.path.join(vdir, "lineage_seed=*")
        seed_dirs = glob.glob(seed_pattern)
        for sdir in seed_dirs:
            try:
                lineage_seed = int(os.path.basename(sdir).split("=")[1])
            except Exception:
                continue

            gen_pattern = os.path.join(sdir, "generation=*")
            gen_dirs = glob.glob(gen_pattern)
            gen_nums = []
            for g in gen_dirs:
                try:
                    gen_nums.append(int(os.path.basename(g).split("=")[1]))
                except Exception:
                    continue

            runs.append((variant, lineage_seed, sorted(gen_nums)))

    return runs


def find_nextflow_log(start_path, max_up=4):
    """
    Walk up from start_path up to max_up directories and try to find a Nextflow run log
    such as '.nextflow.log' or 'nextflow.log'. Return path or None.
    """
    candidates = [".nextflow.log", "nextflow.log", "pipeline.log"]
    path = os.path.abspath(start_path)
    for _ in range(max_up + 1):
        for c in candidates:
            p = os.path.join(path, c)
            if os.path.exists(p):
                return p
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return None


def read_config_metrics(
    base_path, project_name, variant, lineage_seed, generation, agent_id
):
    """
    Read metrics from config parquet file for a given run.
    Args:
        base_path: Root output directory
        project_name: Project name
        variant, lineage_seed, generation, agent_id: Run identifiers
    """
    config_path = os.path.join(
        base_path,
        "configuration",
        f"experiment_id={project_name}",
        f"variant={variant}",
        f"lineage_seed={lineage_seed}",
        f"generation={generation}",
        f"agent_id={agent_id}",
        "config.pq",
    )

    try:
        if os.path.exists(config_path):
            df_config = pd.read_parquet(config_path)
            # Extract initial dry mass from config
            if len(df_config) > 0:
                return {
                    "config_exists": True,
                    "dry_mass": df_config.iloc[0].get(
                        "parameters__container__dry_mass", np.nan
                    ),
                }
    except Exception as e:
        print(f"Error reading config {config_path}: {e}")

    return {"config_exists": False}


def read_success_state(filepath):
    """
    Read final state from success parquet file.
    """
    try:
        df = pd.read_parquet(filepath)
        if len(df) > 0:
            row = df.iloc[0]
            return {
                "duration_sec": float(row.get("time", 0)),
                "final_dry_mass": float(row.get("cell_dry_mass", 0)),
                "growth_rate": float(row.get("growth_rate", 0)),
                "n_timepoints": len(df),
            }
    except Exception as e:
        print(f"Error reading success file {filepath}: {e}")

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate gene knockout simulation results into a summary CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aggregate_knockout_results.py --project gene_knockout_p_list
  python aggregate_knockout_results.py --project gene_knockout_test --output-dir /path/to/results --output-file p_list_summary.csv
        """,
    )

    parser.add_argument(
        "--project",
        required=True,
        help="Project name (e.g., 'gene_knockout_p_list', 'gene_knockout_test')",
    )
    parser.add_argument(
        "--output-dir",
        default="/user/home/il22158/work/vEcoli/surrogate/results",
        help="Output directory for summary CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output filename (default: {project}_success_summary.csv)",
    )
    parser.add_argument(
        "--base-path",
        default="/user/home/il22158/work/vEcoli/out",
        help="Base output directory containing simulation results (default: %(default)s)",
    )

    args = parser.parse_args()

    # Construct paths
    base_path = os.path.join(args.base_path, args.project)
    output_dir = args.output_dir
    output_file = args.output_file or f"{args.project}_success_summary.csv"
    output_path = os.path.join(output_dir, output_file)

    print(f"Aggregating results for project: {args.project}")
    print(f"Base path: {base_path}")
    print(f"Output: {output_path}\n")

    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Enumerate runs (all variant/seed combinations present)
    runs = enumerate_runs(base_path)
    print(f"\nFound {len(runs)} variant-seed directories to inspect")

    # Also collect success info to speed up lookup (map of (variant,seed) -> info)
    success_info = get_success_generations(base_path, args.project)

    # Build summary data
    summary_data = []

    for variant, lineage_seed, gen_list in sorted(runs):
        info = success_info.get(
            (variant, lineage_seed), {"max_gen": 0, "generations": set()}
        )
        max_gen = info.get("max_gen", 0)

        stop_status = None

        if max_gen > 0:
            # successful run (at least one generation with success)
            stop_status = "success"
        else:
            # No success recorded; try to infer why the run stopped
            if gen_list:
                last_gen = gen_list[-1]
                # look for any log/error files in last generation directory
                gen_dir = os.path.join(
                    base_path,
                    f"variant={variant}",
                    f"lineage_seed={lineage_seed}",
                    f"generation={last_gen}",
                )
                # search for error/log files
                patterns = ["*.err", "*.stderr", "*.out", "*.log", "*error*", "*fail*"]
                found = []
                for pat in patterns:
                    found.extend(
                        glob.glob(os.path.join(gen_dir, "**", pat), recursive=True)
                    )

                if found:
                    # read the first file's tail (safe, small)
                    candidate = found[0]
                    try:
                        with open(candidate, "rb") as f:
                            f.seek(0, os.SEEK_END)
                            size = f.tell()
                            # read last ~4KB
                            tail_size = min(4096, size)
                            f.seek(max(0, size - tail_size))
                            tail = f.read().decode(errors="replace")
                            stop_status = f"no_success; last_generation={last_gen}; log_tail={tail.strip().replace('\n', ' | ')}"
                    except Exception as e:
                        stop_status = f"no_success; last_generation={last_gen}; error_reading_log={e}"
                else:
                    # try to find Nextflow run log upwards from base_path
                    nf_log = find_nextflow_log(base_path)
                    if nf_log:
                        try:
                            with open(nf_log, "rb") as f:
                                f.seek(0, os.SEEK_END)
                                size = f.tell()
                                tail_size = min(8192, size)
                                f.seek(max(0, size - tail_size))
                                tail = f.read().decode(errors="replace")
                                # try to extract lines mentioning this variant/seed/generation
                                marker = f"variant={variant} lineage_seed={lineage_seed} generation={last_gen}"
                                # if marker not found, just include tail
                                if marker in tail:
                                    excerpt = "\n".join(
                                        [
                                            line
                                            for line in tail.splitlines()
                                            if str(variant) in line
                                            or str(lineage_seed) in line
                                            or str(last_gen) in line
                                        ]
                                    )
                                else:
                                    excerpt = tail
                                stop_status = f"no_success; last_generation={last_gen}; nextflow_log_tail={excerpt.strip().replace('\n', ' | ')}"
                        except Exception as e:
                            stop_status = f"no_success; last_generation={last_gen}; error_reading_nextflow_log={e}"
                    else:
                        stop_status = (
                            f"no_success; last_generation={last_gen}; no_log_found"
                        )
            else:
                # no generation folders at all
                # maybe run never started
                stop_status = "no_generations_found; run_maybe_not_started"

        row = {
            "project": args.project,
            "variant": variant,
            "lineage_seed": lineage_seed,
            "max_success_generation": max_gen,
            "stop_status": stop_status,
        }

        summary_data.append(row)
    # If no run directories were discovered but there are success records, fall back to success_info
    if not summary_data and success_info:
        for (variant, lineage_seed), info in sorted(success_info.items()):
            row = {
                "project": args.project,
                "variant": variant,
                "lineage_seed": lineage_seed,
                "max_success_generation": info.get("max_gen", 0),
                "stop_status": "success"
                if info.get("max_gen", 0) > 0
                else "no_success_detected",
            }
            summary_data.append(row)

    # Create DataFrame
    df_summary = pd.DataFrame(summary_data)

    # Sort by variant and seed if possible
    if (
        not df_summary.empty
        and "variant" in df_summary.columns
        and "lineage_seed" in df_summary.columns
    ):
        df_summary = df_summary.sort_values(["variant", "lineage_seed"]).reset_index(
            drop=True
        )

    # Save to CSV
    df_summary.to_csv(output_path, index=False)

    print("\nSummary statistics:")
    print(f"Total runs: {len(df_summary)}")
    if not df_summary.empty:
        print(f"Unique variants: {df_summary['variant'].nunique()}")
        print(f"Unique seeds: {df_summary['lineage_seed'].nunique()}")
        print(
            f"Average max generation: {df_summary['max_success_generation'].mean():.2f}"
        )
    print(f"\nSummary saved to: {output_path}")

    # Print first few rows
    print("\nFirst 10 rows:")
    print(df_summary.head(10).to_string())


if __name__ == "__main__":
    main()
