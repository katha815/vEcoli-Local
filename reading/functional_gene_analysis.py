#!/usr/bin/env python3
"""
Functional Gene Analysis for vEcoli Simulations

Calculates minimum functional units for protein complexes across simulation data.

*Core Functions*
load_sim_data(project_folder)

Loads simulation data object containing all molecular definitions
Path: out/{project}/kb/simData_Modified.cPickle
load_bulk_data(project_folder, variant, generations, downsample)

Loads protein/molecule counts from parquet files across all generations
Extracts bulk molecule columns and downsamples timepoints
Returns: bulk count arrays (generations × timepoints × molecules) + molecule IDs
calculate_functional_units(sim_data, bulk_data, bulk_ids)

Core algorithm: For each complex, calculates min(monomer_count / stoichiometry) across all subunits and timepoints
Aggregates across 3 complexation processes: complexation, equilibrium, two_component_system
Returns: monomer → min functional units mapping
map_monomers_to_genes(sim_data)

Builds monomer → gene ID mapping via cistron intermediate
Uses transcription and translation data structures

*Key Settings (Defaults)*
Parameter	Default	Description
--project	all_media_conditions1	Simulation project folder
--variant	0	Variant ID to analyze
--generations	[1,2,3,4,5,6,7,8]	Generations to include
--downsample	10	Take every Nth timepoint
--threshold	1	Minimum units to be "functional" with selected metric
--output	results/functional_gene/	Output directory

*Output*
CSV: gene_id, monomer_id, min_functional_units, num_complexes, is_functional
PNG: Histogram of functional units distribution

*Algorithm*
For each complex:
  1. Get subunit stoichiometry (e.g., A₂B₃ means 2×A + 3×B)
  2. For each timepoint: functional_units = metric(countₐ/2, countᵦ/3)
  3. Take minimum across ALL timepoints/generations

For each monomer:
  functional_units = max across all complexes it participates in
"""

import pandas as pd
import numpy as np
import pickle
import glob
from pathlib import Path
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"


def load_sim_data(project_folder, variant=0):
    """Load sim_data object"""
    sim_data_path = f"/user/home/il22158/work/vEcoli/out/{project_folder}/variant_sim_data/{variant}.cPickle"
    print(f"Loading sim_data from: {sim_data_path}")
    with open(sim_data_path, "rb") as f:
        return pickle.load(f)


def load_bulk_data(project_folder, variant, generations, downsample=10):
    """Load bulk molecule counts from all generations"""
    bulk_data = {}
    bulk_ids = None

    for gen in generations:
        agent_id = "0" * gen

        # Load config for bulk metadata
        config_path = f"/user/home/il22158/work/vEcoli/out/{project_folder}/configuration/experiment_id={project_folder}/variant={variant}/lineage_seed=0/generation={gen}/agent_id={agent_id}/config.pq"
        df_config = pd.read_parquet(config_path)
        bulk_ids = df_config["output_metadata__bulk"].iloc[0]
        if isinstance(bulk_ids, np.ndarray):
            bulk_ids = bulk_ids.tolist()

        # Load history
        history_path = f"/user/home/il22158/work/vEcoli/out/{project_folder}/history/experiment_id={project_folder}/variant={variant}/lineage_seed=0/generation={gen}/agent_id={agent_id}"
        pq_files = sorted(glob.glob(f"{history_path}/*.pq"))

        if not pq_files:
            print(f"Warning: No files found for generation {gen}")
            continue

        print(f"Loading gen {gen}: {len(pq_files)} files...")
        df = (
            pd.concat([pd.read_parquet(f) for f in pq_files], ignore_index=True)
            .sort_values("time")
            .reset_index(drop=True)
        )

        # Extract bulk counts using apply
        bulk_counts = np.array([row for row in df["bulk"].values[::downsample]])
        bulk_data[gen] = bulk_counts
        print(f"  Gen {gen}: {len(bulk_data[gen])} timepoints")

    return bulk_data, bulk_ids


def calculate_functional_units(sim_data, bulk_data, bulk_ids, metric="mean"):
    """Calculate functional units for each monomer

    Args:
        metric: 'min', 'mean', or 'median'
    """
    bulk_id_to_idx = {mol_id: idx for idx, mol_id in enumerate(bulk_ids)}

    # Get all complexes
    complexation = sim_data.process.complexation
    equilibrium = sim_data.process.equilibrium
    two_component = sim_data.process.two_component_system

    all_complexes = list(
        set(
            list(complexation.ids_complexes)
            + list(equilibrium.ids_complexes)
            + list(two_component.complex_to_monomer.keys())
        )
    )

    print(f"\nAnalyzing {len(all_complexes)} complexes...")

    monomer_functional_units = defaultdict(lambda: 0)
    monomer_to_complexes = defaultdict(list)

    for complex_id in all_complexes:
        try:
            if complex_id in complexation.molecule_names:
                subunit_info = complexation.get_monomers(complex_id)
            elif complex_id in equilibrium.molecule_names:
                subunit_info = equilibrium.get_monomers(complex_id)
            else:
                subunit_info = two_component.get_monomers(complex_id)
        except Exception:
            continue

        subunit_ids = subunit_info["subunitIds"]
        subunit_stoich = subunit_info["subunitStoich"]

        if len(subunit_ids) == 0 or (
            len(subunit_ids) == 1 and subunit_ids[0] == complex_id
        ):
            continue

        # Find valid subunits in bulk
        subunit_indices = []
        valid_subunits = []
        valid_stoich = []

        for subunit_id, stoich in zip(subunit_ids, subunit_stoich):
            if subunit_id in bulk_id_to_idx:
                subunit_indices.append(bulk_id_to_idx[subunit_id])
                valid_subunits.append(subunit_id)
                valid_stoich.append(stoich)

        if not subunit_indices:
            continue

        # Collect all functional units (across subunits)
        all_functional_units = []
        for counts in bulk_data.values():
            for timestep_counts in counts:
                subunit_counts = timestep_counts[subunit_indices]
                functional_units = np.min(subunit_counts / np.array(valid_stoich))
                all_functional_units.append(functional_units)

        # Apply aggregation metric (across timepoints)
        if metric == "min":
            aggregate_functional = np.min(all_functional_units)
        elif metric == "mean":
            aggregate_functional = np.mean(all_functional_units)
        elif metric == "median":
            aggregate_functional = np.median(all_functional_units)

        # Map to monomers
        for subunit_id in valid_subunits:
            monomer_to_complexes[subunit_id].append(complex_id)
            monomer_functional_units[subunit_id] = max(
                monomer_functional_units[subunit_id], aggregate_functional
            )

    print(f"Analyzed {len(monomer_functional_units)} monomers")
    return dict(monomer_functional_units), dict(monomer_to_complexes)


def map_monomers_to_genes(sim_data):
    """Build monomer -> gene mapping using the same approach as gene_expression_trace.py

    Pattern: monomer_id -> cistron_id -> gene_id

    No corresponding genes reasons?
        Some "monomers" in the complexation system are actually:
            Not real monomers (metabolites, complexes)
            Alternative products from same gene (frameshifts)
            Pseudogene products (no active gene)

    Only 1561 genes code for proteins that participate in complexes (out of 4747)?
        ~3200 genes code for proteins that either:
            Work as monomers (not forming complexes)
            Encode RNAs (rRNA, tRNA, sRNA)
            Are condition-specific (not expressed in minimal media)
            Are pseudogenes or inactive
    """
    monomer_data = sim_data.process.translation.monomer_data
    cistron_data = sim_data.process.transcription.cistron_data

    # Build cistron_id to gene_id mapping (direct field access)
    cistron_to_gene = dict(zip(cistron_data["id"], cistron_data["gene_id"]))

    # Build monomer to gene mapping via cistron intermediate
    monomer_to_gene = {}

    for monomer in monomer_data:
        cistron_id = monomer["cistron_id"]
        monomer_id = monomer[
            "id"
        ]  # Full ID with location suffix e.g. "ACKA-MONOMER[c]"

        if cistron_id in cistron_to_gene:
            gene_id = cistron_to_gene[cistron_id]
            # Store with full ID (includes location suffix)
            monomer_to_gene[monomer_id] = gene_id
            # Also store base ID (without location suffix) for flexible lookup
            monomer_id_base = monomer_id.split("[")[0]
            monomer_to_gene[monomer_id_base] = gene_id

    print(
        f"Built mapping: {len(cistron_to_gene)} cistrons -> {len(monomer_data)} monomers"
    )
    print(f"Lookup table entries: {len(monomer_to_gene)} (includes base + full IDs)")

    return monomer_to_gene


def main():
    parser = argparse.ArgumentParser(description="Analyze functional genes in vEcoli")
    parser.add_argument(
        "--project", "-p", default="all_media_conditions1", help="Project folder"
    )
    parser.add_argument("--variant", "-v", type=int, default=0, help="Variant ID")
    parser.add_argument(
        "--generations",
        "-g",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Generations",
    )
    parser.add_argument(
        "--downsample", "-d", type=int, default=20, help="Downsample interval"
    )
    parser.add_argument("--threshold", "-t", type=float, default=1, help="Threshold")
    parser.add_argument(
        "--metric",
        "-m",
        default="mean",
        choices=["min", "mean", "median"],
        help="Aggregation metric (default: mean)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="/user/home/il22158/work/vEcoli/reading/results/functional_gene",
        help="Output dir",
    )
    args = parser.parse_args()

    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(
        f"Project: {args.project}, Variant: {args.variant}, Metric: {args.metric}, Threshold: {args.threshold}"
    )
    print("=" * 60)

    # Load data
    sim_data = load_sim_data(args.project, args.variant)
    bulk_data, bulk_ids = load_bulk_data(
        args.project, args.variant, args.generations, args.downsample
    )

    # Calculate functional units with specified metric
    monomer_functional_units, monomer_to_complexes = calculate_functional_units(
        sim_data, bulk_data, bulk_ids, metric=args.metric
    )
    monomer_to_gene = map_monomers_to_genes(sim_data)

    # Build results
    results = []
    for monomer_id, min_units in monomer_functional_units.items():
        gene_id = monomer_to_gene.get(
            monomer_id, monomer_to_gene.get(monomer_id.split("[")[0], "Unknown")
        )
        results.append(
            {
                "gene_id": gene_id,
                "monomer_id": monomer_id,
                f"{args.metric}_functional_units": min_units,
                "num_complexes": len(monomer_to_complexes[monomer_id]),
                "is_functional": min_units >= args.threshold,
            }
        )

    results_df = pd.DataFrame(results).sort_values(f"{args.metric}_functional_units")

    # Summary
    print(f"\n{'=' * 60}")
    print(
        f"Monomers: {len(results_df)} | Functional: {results_df['is_functional'].sum()}"
    )
    print(
        f"Genes: {len(results_df['gene_id'].unique())} | Functional: {len(results_df[results_df['is_functional']]['gene_id'].unique())}"
    )
    print(f"{'=' * 60}")

    # Save CSV with metric in filename
    output_file = f"{args.output}/{args.project}_functional_genes_v{args.variant}_{args.metric}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ {output_file}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    units = results_df[f"{args.metric}_functional_units"].values
    ax.hist(
        units[units < 100], bins=50, edgecolor="black", alpha=0.7, color="steelblue"
    )
    ax.axvline(
        args.threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {args.threshold}",
    )
    ax.set_xlabel(f"{args.metric.capitalize()} Functional Units", fontsize=12)
    ax.set_ylabel("Number of Monomers", fontsize=12)
    ax.set_title(
        f"Functional Units Distribution ({args.metric.capitalize()})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    plot_file = f"{args.output}/{args.project}_functional_genes_v{args.variant}_{args.metric}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"✓ {plot_file}\n")
    plt.close()


if __name__ == "__main__":
    main()
