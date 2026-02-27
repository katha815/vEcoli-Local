#!/usr/bin/env python3
"""
Extract growth rate data from vEcoli simulations.
Based on multi_gen_plot.py approach.

Usage:
    python growth_rate_extract.py --all  # Extract all variants, all 8 generations
    python growth_rate_extract.py --trial  # Extract 2 variants, 2 generations (quick test)
"""

import pandas as pd
import numpy as np
import glob
import json
import argparse
import os


def extract_growth_rate_data(
    project_folder,
    variant_key="condition",
    generation=1,
    read_interval_sec=20,
    max_variants=None,
    save_timeseries=False,
    lineage_seed=0,
):
    """
    Extract growth rate metrics from vEcoli simulation data.

    Parameters:
    -----------
    project_folder : str
        Name of the project folder (e.g., "all_media_conditions1")
    variant_key : str
        Key in metadata for variant information
    generation : int or list
        Generation(s) to extract. If list, data is averaged across generations
    read_interval_sec : int
        Downsample data (take 1 point every N seconds). Default=20
    max_variants : int or None
        Max variants to extract (for testing). None = all
    save_timeseries : bool
        If True, save all timepoints to parquet. If False, return summary only
    lineage_seed : int
        The lineage seed to use for the path.

    Returns:
    --------
    df_growth : pd.DataFrame
        Summary statistics or full timeseries data

    Output Parquet File:
    --------------------
    If save_timeseries is True, the function will output a parquet file containing the full downsampled timeseries for all variants and generations.
    Each row in the parquet file will include the following columns:
        - project (project folder name)
        - variant (variant id)
        - label (variant label, e.g., KO: gene name)
        - time (simulation time)
        - listeners__mass__instantaneous_growth_rate
        - listeners__mass__dry_mass
    The file will be saved to:
        /user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_timeseries_{suffix}_{run_type}.parquet
    """

    # Convert generation to list
    gen_list = [generation] if isinstance(generation, int) else list(generation)

    # Load metadata
    metadata_file = f"/user/home/il22158/work/vEcoli/out/{project_folder}/variant_sim_data/metadata.json"
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        variant_metadata = metadata.get(variant_key, {})
        variant_ids = sorted([int(k) for k in variant_metadata.keys()])
        if max_variants:
            variant_ids = variant_ids[:max_variants]
    except FileNotFoundError:
        print(f"❌ Metadata not found: {project_folder}")
        return pd.DataFrame()

    # Extract data for each variant
    all_data = []

    for variant_id in variant_ids:
        # Get label
        v_info = variant_metadata.get(str(variant_id), f"variant_{variant_id}")
        if isinstance(v_info, str):
            label = v_info
        elif isinstance(v_info, dict):
            genes = v_info.get("genes_to_knockout", [])
            label = (
                f"KO: {', '.join(genes)}"
                if genes
                else v_info.get("condition", str(v_info))
            )
        else:
            label = f"variant_{variant_id}"

        # Load data from all generations
        gen_data = []
        for gen in gen_list:
            agent_id = "0" * gen

            base_path = f"/user/home/il22158/work/vEcoli/out/{project_folder}/history/experiment_id={project_folder}/variant={variant_id}/lineage_seed={lineage_seed}/generation={gen}/agent_id={agent_id}"
            pq_files = sorted(glob.glob(f"{base_path}/*.pq"))

            if not pq_files:
                continue

            # Load and downsample
            dfs = []
            for pq_file in pq_files:
                df_temp = pd.read_parquet(pq_file)
                if read_interval_sec and read_interval_sec > 1:
                    df_temp = df_temp.iloc[::read_interval_sec]
                dfs.append(df_temp)

            df_gen = (
                pd.concat(dfs, ignore_index=True)
                .sort_values("time")
                .reset_index(drop=True)
            )
            gen_data.append(df_gen)

        if not gen_data:
            continue

        # Concatenate generations
        df_all = pd.concat(gen_data, ignore_index=True)

        if save_timeseries:
            # Save all timepoints
            df_all["project"] = project_folder
            df_all["variant"] = variant_id
            df_all["label"] = label
            all_data.append(
                df_all[
                    [
                        "project",
                        "variant",
                        "label",
                        "time",
                        "listeners__mass__instantaneous_growth_rate",
                        "listeners__mass__dry_mass",
                    ]
                ]
            )
        else:
            # Calculate summary statistics
            growth_rate = df_all["listeners__mass__instantaneous_growth_rate"].values
            dry_mass = df_all["listeners__mass__dry_mass"].values
            time_vals = df_all["time"].values

            all_data.append(
                {
                    "project": project_folder,
                    "variant": variant_id,
                    "label": label,
                    "mean_growth_rate": np.mean(growth_rate),
                    "median_growth_rate": np.median(growth_rate),
                    "std_growth_rate": np.std(growth_rate),
                    "max_growth_rate": np.max(growth_rate),
                    "initial_dry_mass": dry_mass[0],
                    "final_dry_mass": dry_mass[-1],
                    "max_dry_mass": np.max(dry_mass),
                    "fold_change_mass": dry_mass[-1] / dry_mass[0],
                    "duration_sec": time_vals[-1] - time_vals[0],
                    "n_timepoints": len(df_all),
                    "generations_used": len(gen_list),
                }
            )

    # Create DataFrame
    if save_timeseries and all_data:
        df_growth = pd.concat(all_data, ignore_index=True)
    else:
        df_growth = pd.DataFrame(all_data)

    if len(df_growth) > 0:
        gen_str = f"{len(gen_list)} gens" if len(gen_list) > 1 else f"gen {gen_list[0]}"
        data_type = "timeseries" if save_timeseries else "summary"
        print(f"✓ {project_folder}: {len(all_data)} variants, {gen_str}, {data_type}")

    return df_growth


def main():
    parser = argparse.ArgumentParser(
        description="Extract growth rate data from vEcoli simulations.\n\n"
        "By default, only summary statistics are saved as CSV.\n"
        "Use --save-timeseries to also save the full downsampled timeseries for all variants and generations as a parquet file.\n"
        "Example: python growth_rate_extract.py --all --projects gene_ko_trial40_seed101 --suffix seed101 --lineage-seed 101 --save-timeseries"
    )
    parser.add_argument(
        "--save-timeseries",
        action="store_true",
        help="If set, save the full timeseries data to a parquet file. If not set, only summary statistics will be saved.",
    )
    parser.add_argument(
        "--all", action="store_true", help="Extract all variants, all 8 generations"
    )
    parser.add_argument(
        "--trial", action="store_true", help="Trial run: 2 variants, 2 generations"
    )
    parser.add_argument(
        "--generations",
        type=int,
        nargs="+",
        default=None,
        help="Specific generations to extract (e.g., --generations 1 2 3)",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Maximum number of variants to extract per project",
    )
    parser.add_argument(
        "--projects",
        type=str,
        nargs="+",
        default=[
            "all_media_conditions1",
            "gene_knockout_metabolic1",
            "gene_knockout_non_metabolic1",
        ],
        help="Full project folder names (e.g., all_media_conditions1_seed100)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="default",
        help='Suffix for output filenames only (e.g., "seed100", "trial1")',
    )
    parser.add_argument(
        "--lineage-seed",
        type=int,
        default=0,
        help="Lineage seed to use for data extraction.",
    )

    args = parser.parse_args()

    # Define projects - use full folder names directly
    projects = []
    for proj_name in args.projects:
        # Determine variant key based on project name
        if "media" in proj_name.lower() or "condition" in proj_name.lower():
            variant_key = "condition"
        else:
            variant_key = "gene_knockout"

        projects.append({"folder": proj_name, "variant_key": variant_key})

    # Determine extraction parameters
    if args.all:
        generations = list(range(1, 9))  # All 8 generations
        max_variants = None
        run_type = "ALL"
    elif args.trial:
        generations = [1, 2]
        max_variants = 2
        run_type = "TRIAL"
    elif args.generations:
        generations = args.generations
        max_variants = args.max_variants
        run_type = "CUSTOM"
    else:
        print("Please specify --all, --trial, or --generations")
        return

    print("=" * 80)
    print(f"GROWTH RATE EXTRACTION - {run_type}")
    print(f"Projects: {[p['folder'] for p in projects]}")
    print(f"Generations: {generations}")
    print(f"Max variants per project: {max_variants if max_variants else 'all'}")
    print(f"Output suffix: {args.suffix}")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = "/user/home/il22158/work/vEcoli/reading/results/growth_rate"
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting data...")
    all_summary = []
    all_timeseries = []
    for proj in projects:
        df = extract_growth_rate_data(
            proj["folder"],
            proj["variant_key"],
            generation=generations,
            max_variants=max_variants,
            save_timeseries=args.save_timeseries,
            lineage_seed=args.lineage_seed,
        )
        if len(df) > 0:
            if args.save_timeseries:
                # If timeseries, append to timeseries list
                all_timeseries.append(df)
            else:
                # If summary, append to summary list
                all_summary.append(df)

    if args.save_timeseries:
        if not all_timeseries:
            print(
                "\nERROR: No data was extracted from any project. Check project names and data paths."
            )
            return
        df_timeseries = pd.concat(all_timeseries, ignore_index=True)
        # Save timeseries
        timeseries_file = f"{output_dir}/growth_rate_timeseries_{args.suffix}_{run_type.lower()}.parquet"
        df_timeseries.to_parquet(timeseries_file, index=False)
        print(
            f"✓ Saved timeseries: {timeseries_file} ({len(df_timeseries)} timepoints)"
        )
        # Also create summary from timeseries
        # Group by project and variant, calculate summary stats
        summary_rows = []
        for (project, variant), group in df_timeseries.groupby(["project", "variant"]):
            growth_rate = group["listeners__mass__instantaneous_growth_rate"].values
            dry_mass = group["listeners__mass__dry_mass"].values
            time_vals = group["time"].values
            label = group["label"].iloc[0]
            summary_rows.append(
                {
                    "project": project,
                    "variant": variant,
                    "label": label,
                    "mean_growth_rate": np.mean(growth_rate),
                    "median_growth_rate": np.median(growth_rate),
                    "std_growth_rate": np.std(growth_rate),
                    "max_growth_rate": np.max(growth_rate),
                    "initial_dry_mass": dry_mass[0],
                    "final_dry_mass": dry_mass[-1],
                    "max_dry_mass": np.max(dry_mass),
                    "fold_change_mass": dry_mass[-1] / dry_mass[0],
                    "duration_sec": time_vals[-1] - time_vals[0],
                    "n_timepoints": len(group),
                    "generations_used": len(generations),
                }
            )
        df_summary = pd.DataFrame(summary_rows)
    else:
        if not all_summary:
            print(
                "\nERROR: No data was extracted from any project. Check project names and data paths."
            )
            return
        df_summary = pd.concat(all_summary, ignore_index=True)

    # Print summary
    print()
    print("=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(df_summary)}")
    print()
    print("Samples per project:")
    print(df_summary.groupby("project")["variant"].count())
    print()

    # Save summary
    summary_file = (
        f"{output_dir}/growth_rate_summary_{args.suffix}_{run_type.lower()}.csv"
    )
    df_summary.to_csv(summary_file, index=False)
    print(f"✓ Saved summary: {summary_file}")

    # Print final statistics
    print()
    print("=" * 80)
    print("STATISTICS BY PROJECT")
    print("=" * 80)

    project_stats = (
        df_summary.groupby("project")
        .agg(
            {
                "variant": "count",
                "mean_growth_rate": ["mean", "std", "min", "max"],
                "max_growth_rate": "max",
                "fold_change_mass": ["mean", "std", "min", "max"],
                "n_timepoints": "sum",
            }
        )
        .round(6)
    )

    print(project_stats)
    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
