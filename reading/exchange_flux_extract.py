#!/usr/bin/env python3
"""
Extract external exchange fluxes (metabolite readings) from vEcoli simulations.
Based on growth_rate_extract.py.

The history files store the fluxes as one vector per timepoint
(listeners__fba_results__external_exchange_fluxes) without names. The names
come from sim_data.external_state.all_external_exchange_molecules (same order);
the compartment suffix (e.g. "[p]") is stripped to match the chemical names in
unique_exchanges.csv.

Usage:
    python exchange_flux_extract.py --all  # all variants, all available generations on disk
    python exchange_flux_extract.py --trial  # 2 variants, 2 generations (quick test)
    python exchange_flux_extract.py --all --lineage-seeds 100 101
    python exchange_flux_extract.py --all --metabolite-list /path/to/list.csv
"""

import argparse
import functools
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT_ROOT = "/user/home/il22158/work/vEcoli/out"
EXCHANGE_COL = "listeners__fba_results__external_exchange_fluxes"
DEFAULT_METABOLITE_LIST = (
    "/user/home/il22158/work/vEcoli/reading/results/colony/unique_exchange_non_inf.csv"
)
REL_TOL = 1e-3  # constant if (max - min) <= REL_TOL * max|flux|


def get_available_generations(variant_base_path):
    """Return generations that have at least one parquet file under the variant path."""
    available_generations = []

    for generation_path in sorted(glob.glob(f"{variant_base_path}/generation=*")):
        generation_name = os.path.basename(generation_path)
        try:
            generation = int(generation_name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue

        pq_files = glob.glob(f"{generation_path}/agent_id=*/*.pq")
        if pq_files:
            available_generations.append(generation)

    return sorted(set(available_generations))


def load_metabolite_list(path):
    """Read the chemical names from a CSV (column 'exchange_chemical', else the first column)."""
    df = pd.read_csv(path)
    col = "exchange_chemical" if "exchange_chemical" in df.columns else df.columns[0]
    return df[col].dropna().astype(str).tolist()


@functools.lru_cache(maxsize=None)
def load_exchange_names(project_folder):
    """Chemical names (no compartment suffix) in the order of the flux vector."""
    sim_data_path = f"{OUT_ROOT}/{project_folder}/parca/kb/simData.cPickle"
    with open(sim_data_path, "rb") as f:
        sim_data = pickle.load(f)
    molecules = list(sim_data.external_state.all_external_exchange_molecules)
    return [m[:-3] for m in molecules]


def summarize_fluxes(df, metabolites):
    """Per-metabolite summary statistics (one row per metabolite) for one variant/seed."""
    rows = []
    for chem in metabolites:
        values = df[chem].to_numpy()
        span = values.max() - values.min()
        scale = np.abs(values).max()
        rows.append(
            {
                "metabolite": chem,
                "mean_flux": np.mean(values),
                "median_flux": np.median(values),
                "std_flux": np.std(values),
                "min_flux": values.min(),
                "max_flux": values.max(),
                "n_nonzero": int(np.count_nonzero(values)),
                "is_constant": bool(span <= REL_TOL * scale),
            }
        )
    return rows


def extract_exchange_flux_data(
    project_folder,
    metabolites,
    variant_key="gene_knockout",
    generation=1,
    read_interval_sec=1,
    max_variants=None,
    save_timeseries=False,
    lineage_seed=0,
):
    """
    Extract exchange fluxes of the selected metabolites from vEcoli simulation data.

    Parameters:
    -----------
    project_folder : str
        Name of the project folder (e.g., "gene_knockout_462KO_basal_operon_on_p1")
    metabolites : list of str
        Chemical names to extract (no compartment suffix).
    variant_key : str
        Key in metadata for variant information
    generation : int, list, or None
        Generation(s) to extract. If None, all available generations for each
        variant are discovered from disk. The summary column max_generation
        reports the highest generation with data for that variant.
    read_interval_sec : int
        Downsample data (take 1 point every N original timepoints). Default=1
    max_variants : int or None
        Max variants to extract (for testing). None = all
    save_timeseries : bool
        If True, return all timepoints (wide: one column per metabolite).
        If False, return per-metabolite summary statistics only.
    lineage_seed : int
        The lineage seed to use for the path.

    Returns:
    --------
    df_flux : pd.DataFrame
        Summary statistics (long format, one row per variant and metabolite) or
        the full timeseries (one row per timepoint, one column per metabolite).

    Output Parquet File (if save_timeseries):
    -----------------------------------------
    Columns: project, variant, label, lineage_seed, generation, time, then one
    column per metabolite (flux in mmol/gDCW/hr). Saved to:
        /user/home/il22158/work/vEcoli/reading/results/exchange_flux/exchange_flux_timeseries_{suffix}_{run_type}.parquet
    """

    gen_list = (
        None
        if generation is None
        else ([generation] if isinstance(generation, int) else list(generation))
    )

    metadata_file = f"{OUT_ROOT}/{project_folder}/variant_sim_data/metadata.json"
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

    # Skip projects that were set up but never produced history (no sim_data needed)
    history_glob = (
        f"{OUT_ROOT}/{project_folder}/history/experiment_id={project_folder}"
        f"/variant=*/lineage_seed={lineage_seed}/generation=*/agent_id=*/*.pq"
    )
    if next(glob.iglob(history_glob), None) is None:
        print(f"❌ No history files: {project_folder} (seed {lineage_seed})")
        return pd.DataFrame()

    # Map chemical name -> position in the flux vector
    try:
        exchange_names = load_exchange_names(project_folder)
    except FileNotFoundError:
        print(f"❌ simData.cPickle not found: {project_folder}")
        return pd.DataFrame()
    name_to_idx = {name: i for i, name in enumerate(exchange_names)}
    unknown = [m for m in metabolites if m not in name_to_idx]
    if unknown:
        print(f"⚠ Not exchange molecules of {project_folder}, skipped: {unknown}")
    metabolites = [m for m in metabolites if m in name_to_idx]
    if not metabolites:
        print("❌ No valid metabolites to extract")
        return pd.DataFrame()
    col_idx = [name_to_idx[m] for m in metabolites]

    all_data = []

    for variant_id in variant_ids:
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

        variant_base_path = (
            f"{OUT_ROOT}/{project_folder}/history/experiment_id={project_folder}"
            f"/variant={variant_id}/lineage_seed={lineage_seed}"
        )
        variant_gen_list = (
            get_available_generations(variant_base_path)
            if gen_list is None
            else gen_list
        )

        gen_data = []
        loaded_generations = []
        for gen in variant_gen_list:
            agent_id = "0" * gen
            pq_files = sorted(
                glob.glob(
                    f"{variant_base_path}/generation={gen}/agent_id={agent_id}/*.pq"
                )
            )
            if not pq_files:
                continue

            dfs = []
            for pq_file in pq_files:
                # Only read the columns needed
                table = pq.read_table(pq_file, columns=["time", EXCHANGE_COL])
                times = table.column("time").to_numpy()
                fluxes = np.vstack(table.column(EXCHANGE_COL).to_pylist())[:, col_idx]
                df_temp = pd.DataFrame(fluxes, columns=metabolites)
                df_temp.insert(0, "time", times)
                if read_interval_sec and read_interval_sec > 1:
                    df_temp = df_temp.iloc[::read_interval_sec]
                dfs.append(df_temp)

            df_gen = (
                pd.concat(dfs, ignore_index=True)
                .sort_values("time")
                .reset_index(drop=True)
            )
            df_gen["generation"] = gen
            gen_data.append(df_gen)
            loaded_generations.append(gen)

        if not gen_data:
            continue

        df_all = pd.concat(gen_data, ignore_index=True)

        if save_timeseries:
            df_all["project"] = project_folder
            df_all["variant"] = variant_id
            df_all["label"] = label
            df_all["lineage_seed"] = lineage_seed
            all_data.append(
                df_all[
                    [
                        "project",
                        "variant",
                        "label",
                        "lineage_seed",
                        "generation",
                        "time",
                    ]
                    + metabolites
                ]
            )
        else:
            for row in summarize_fluxes(df_all, metabolites):
                all_data.append(
                    {
                        "project": project_folder,
                        "variant": variant_id,
                        "label": label,
                        "lineage_seed": lineage_seed,
                        **row,
                        "n_timepoints": len(df_all),
                        "max_generation": max(loaded_generations),
                    }
                )

    if not all_data:
        return pd.DataFrame()

    if save_timeseries:
        df_flux = pd.concat(all_data, ignore_index=True)
    else:
        df_flux = pd.DataFrame(all_data)

    if gen_list is None:
        gen_str = "all available generations"
    else:
        gen_str = f"{len(gen_list)} gens" if len(gen_list) > 1 else f"gen {gen_list[0]}"
    data_type = "timeseries" if save_timeseries else "summary"
    n_variants = df_flux["variant"].nunique()
    print(
        f"✓ {project_folder} (seed {lineage_seed}): {n_variants} variants, "
        f"{len(metabolites)} metabolites, {gen_str}, {data_type}"
    )

    return df_flux


def main():
    parser = argparse.ArgumentParser(
        description="Extract exchange fluxes of selected metabolites from vEcoli "
        "simulations. By default, only per-metabolite summary statistics are saved "
        "as CSV. Example: python exchange_flux_extract.py --all "
        "--projects gene_knockout_462KO_basal_operon_on_p1 --lineage-seeds 100 101",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="all variants, every generation found on disk",
    )
    parser.add_argument(
        "--trial", action="store_true", help="quick test: 2 variants, generations 1-2"
    )
    parser.add_argument(
        "--generations",
        type=int,
        nargs="+",
        default=None,
        help="specific generations to extract (e.g. 1 2 3)",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="max variants per project (with --generations)",
    )
    parser.add_argument(
        "--projects",
        type=str,
        nargs="+",
        default=["gene_knockout_462KO_basal_operon_on_p1"],
        help="project folder names under out/",
    )
    parser.add_argument(
        "--lineage-seeds",
        type=int,
        nargs="+",
        default=[100],
        help="lineage seeds to extract",
    )
    parser.add_argument(
        "--metabolite-list",
        type=str,
        default=DEFAULT_METABOLITE_LIST,
        help="CSV of chemical names (no compartment suffix) in column "
        "'exchange_chemical', or the first column",
    )
    parser.add_argument(
        "--read-interval",
        type=int,
        default=1,
        help="keep one timepoint every N per history file",
    )
    parser.add_argument(
        "--save-timeseries",
        action="store_true",
        help="also save the full timeseries as parquet",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="default",
        help="suffix for output filenames",
    )

    args = parser.parse_args()

    projects = []
    for proj_name in args.projects:
        if "media" in proj_name.lower() or "condition" in proj_name.lower():
            variant_key = "condition"
        else:
            variant_key = "gene_knockout"
        projects.append({"folder": proj_name, "variant_key": variant_key})

    if args.all:
        generations = None
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

    metabolites = load_metabolite_list(args.metabolite_list)

    print("=" * 80)
    print(f"EXCHANGE FLUX EXTRACTION - {run_type}")
    print(f"Projects: {[p['folder'] for p in projects]}")
    print(f"Generations: {'all available' if generations is None else generations}")
    print(f"Max variants per project: {max_variants if max_variants else 'all'}")
    print(f"Lineage seeds: {args.lineage_seeds}")
    print(f"Metabolite list: {args.metabolite_list} ({len(metabolites)} metabolites)")
    print(f"Output suffix: {args.suffix}")
    print("=" * 80)
    print()

    output_dir = "/user/home/il22158/work/vEcoli/reading/results/exchange_flux"
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting data...")
    all_summary = []
    all_timeseries = []
    for proj in projects:
        for lineage_seed in args.lineage_seeds:
            df = extract_exchange_flux_data(
                proj["folder"],
                metabolites,
                proj["variant_key"],
                generation=generations,
                read_interval_sec=args.read_interval,
                max_variants=max_variants,
                save_timeseries=args.save_timeseries,
                lineage_seed=lineage_seed,
            )
            if len(df) > 0:
                if args.save_timeseries:
                    all_timeseries.append(df)
                else:
                    all_summary.append(df)

    if args.save_timeseries:
        if not all_timeseries:
            print(
                "\nERROR: No data was extracted from any project. Check project names and data paths."
            )
            return
        df_timeseries = pd.concat(all_timeseries, ignore_index=True)
        timeseries_file = f"{output_dir}/exchange_flux_timeseries_{args.suffix}_{run_type.lower()}.parquet"
        df_timeseries.to_parquet(timeseries_file, index=False)
        print(
            f"✓ Saved timeseries: {timeseries_file} ({len(df_timeseries)} timepoints)"
        )

        # Summary from the timeseries, one row per variant/seed/metabolite
        meta_cols = [
            "project",
            "variant",
            "label",
            "lineage_seed",
            "generation",
            "time",
        ]
        chem_cols = [c for c in df_timeseries.columns if c not in meta_cols]
        summary_rows = []
        for (project, variant, lineage_seed), group in df_timeseries.groupby(
            ["project", "variant", "lineage_seed"]
        ):
            for row in summarize_fluxes(group, chem_cols):
                summary_rows.append(
                    {
                        "project": project,
                        "variant": variant,
                        "label": group["label"].iloc[0],
                        "lineage_seed": lineage_seed,
                        **row,
                        "n_timepoints": len(group),
                        "max_generation": int(group["generation"].max()),
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

    print()
    print("=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    n_samples = df_summary[["project", "variant", "lineage_seed"]].drop_duplicates()
    print(f"Total samples (variant x seed): {len(n_samples)}")
    print()
    print("Samples per project:")
    print(n_samples.groupby("project")["variant"].count())
    print()

    summary_file = (
        f"{output_dir}/exchange_flux_summary_{args.suffix}_{run_type.lower()}.csv"
    )
    df_summary.to_csv(summary_file, index=False)
    print(f"✓ Saved summary: {summary_file}")

    print()
    print("=" * 80)
    print("STATISTICS BY METABOLITE (across all variants/seeds)")
    print("=" * 80)
    metabolite_stats = (
        df_summary.groupby("metabolite")
        .agg(
            n_samples=("variant", "count"),
            n_constant=("is_constant", "sum"),
            mean_flux=("mean_flux", "mean"),
            min_flux=("min_flux", "min"),
            max_flux=("max_flux", "max"),
        )
        .round(6)
    )
    print(metabolite_stats.to_string())
    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
