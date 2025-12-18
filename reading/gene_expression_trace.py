import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os


def save_results(
    results,
    gene_id,
    project_folder,
    variants,
    save_dir="/user/home/il22158/work/vEcoli/reading/results/gene_trace",
    save_pic=True,
    save_csv=True,
):
    """Save results to disk including plots and data."""
    if not results:
        print("No results to save!")
        return None

    # Create directory structure
    variants_str = "_".join([str(v) for v in sorted(variants)])
    output_dir = f"{save_dir}/{project_folder}_v_{variants_str}"
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    if save_pic:
        fig_path = f"{output_dir}/{gene_id}_dynamics.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved to: {fig_path}")

    # Save summary statistics as CSV
    summary_data = []
    for label, data in results.items():
        row = {
            "label": label,
            "variant": data["variant"],
            "generation": data["generation"],
            "duration_s": data["time"][-1] - data["time"][0],
            "mRNA_initial": data["mRNA"][0],
            "mRNA_final": data["mRNA"][-1],
            "mRNA_change_pct": 100
            * ((data["mRNA"][-1] - data["mRNA"][0]) / max(data["mRNA"][0], 1)),
            "protein_initial": data["protein"][0],
            "protein_final": data["protein"][-1],
            "protein_change_pct": 100
            * ((data["protein"][-1] - data["protein"][0]) / max(data["protein"][0], 1)),
        }
        if data["tl_init_events"] is not None:
            row["total_tl_initiations"] = data["tl_init_events"].sum()
        if data["tc_init_events"] is not None:
            row["total_tc_initiations"] = data["tc_init_events"].sum()
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = f"{output_dir}/{gene_id}_summary.csv"
    if save_csv:
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")

    return summary_df


def track_knockout_dynamics(
    gene_id,
    project_folder,
    variants,
    generations,
    figsize=(16, 10),
    plot=True,
    save=True,
    downsample_sec=20,
):
    """Track mRNA and protein dynamics for a knocked-out gene.
    Parameters
    ----------
    gene_id : str
        Gene ID to track.
    project_folder : str
        Project folder name.
    variants : list of int
        Variant(s) to analyze.
    generations : list of int
        Generation(s) to analyze.
    figsize : tuple
        Figure size as (width, height).
    plot : bool
        Whether to plot the results.
    save : bool
        Whether to save the results.
    downsample_sec : int, optional
        Analyze every N seconds (default: 20, i.e., read data every 20 seconds)."""

    # Load sim_data
    variant_sim_data_path = f"/user/home/il22158/work/vEcoli/out/{project_folder}/variant_sim_data/{variants[0]}.cPickle"
    with open(variant_sim_data_path, "rb") as f:
        sim_data = pickle.load(f)

    # Find IDs
    cistron_data = sim_data.process.transcription.cistron_data
    gene_mask = cistron_data["gene_id"] == gene_id
    if not gene_mask.any():
        print(f"Gene {gene_id} not found!")
        return None

    cistron_id = cistron_data["id"][gene_mask][0]
    rna_idxs = sim_data.process.transcription.cistron_id_to_rna_indexes(cistron_id)
    rna_idx = int(rna_idxs[0]) if hasattr(rna_idxs, "__iter__") else int(rna_idxs)
    tu_id = sim_data.process.transcription.rna_data["id"][rna_idx]

    monomer_data = sim_data.process.translation.monomer_data
    monomer_mask = monomer_data["cistron_id"] == cistron_id
    monomer_id = monomer_data["id"][monomer_mask][0] if monomer_mask.any() else None

    print(f"Tracking: Gene={gene_id} | TU={tu_id} | Monomer={monomer_id}\n")

    # Get indices for mRNA and protein lookups
    all_tu_ids = sim_data.process.transcription.rna_data["id"]
    tu_idx_rna = np.where(all_tu_ids == tu_id)[0][0]

    results = {}

    for variant in variants:
        for generation in generations:
            agent_id = "0" * generation
            label = f"V{variant}_G{generation}"

            try:
                # Load config for bulk metadata (protein)
                config_path = f"/user/home/il22158/work/vEcoli/out/{project_folder}/configuration/experiment_id={project_folder}/variant={variant}/lineage_seed=0/generation={generation}/agent_id={agent_id}/config.pq"
                df_config = pd.read_parquet(config_path)
                bulk_ids = df_config["output_metadata__bulk"].iloc[0]
                if isinstance(bulk_ids, np.ndarray):
                    bulk_ids = bulk_ids.tolist()

                if monomer_id and monomer_id in bulk_ids:
                    monomer_idx_bulk = bulk_ids.index(monomer_id)
                else:
                    monomer_idx_bulk = None
                    print(
                        f"{label}: WARNING - monomer_id {monomer_id} not found in bulk_ids. No translation/protein output will be tracked for this gene."
                    )

                # Load mRNA metadata (separate from bulk!)
                mRNA_ids = df_config[
                    "output_metadata__listeners__rna_counts__mRNA_counts"
                ].iloc[0]
                if isinstance(mRNA_ids, np.ndarray):
                    mRNA_ids = mRNA_ids.tolist()
                tu_idx_mrna = mRNA_ids.index(tu_id) if tu_id in mRNA_ids else None

                # Load history
                history_path = f"/user/home/il22158/work/vEcoli/out/{project_folder}/history/experiment_id={project_folder}/variant={variant}/lineage_seed=0/generation={generation}/agent_id={agent_id}"
                pq_files = sorted(glob.glob(f"{history_path}/*.pq"))

                if len(pq_files) == 0:
                    print(f"{label}: No history files found")
                    continue

                df = (
                    pd.concat([pd.read_parquet(f) for f in pq_files], ignore_index=True)
                    .sort_values("time")
                    .reset_index(drop=True)
                )

                # Downsample dataframe by time
                original_len = len(df)
                if downsample_sec > 1:
                    df = df[df["time"] % downsample_sec == 0].reset_index(drop=True)
                    print(
                        f"{label}: Downsampling every {downsample_sec} sec | Original points: {original_len} | After downsampling: {len(df)}"
                    )
                    if len(df) == 0:
                        continue

                # Extract mRNA counts from listener (NOT from bulk!)
                if tu_idx_mrna is not None:
                    mRNA_counts = (
                        df["listeners__rna_counts__mRNA_counts"]
                        .apply(lambda x: x[tu_idx_mrna] if len(x) > tu_idx_mrna else 0)
                        .values
                    )
                else:
                    mRNA_counts = np.zeros(len(df))
                    print(f"{label}: Warning - {tu_id} not found in mRNA_counts!")

                # Extract protein counts from bulk
                if monomer_idx_bulk is not None:
                    protein_counts = (
                        df["bulk"]
                        .apply(
                            lambda x: x[monomer_idx_bulk]
                            if len(x) > monomer_idx_bulk
                            else 0
                        )
                        .values
                    )
                else:
                    protein_counts = np.zeros(len(df))
                    print(
                        f"{label}: Setting protein_counts to zeros (monomer not present in bulk_ids)"
                    )

                # Transcription initiation events
                rna_init_cols = [col for col in df.columns if "rna_init_event" in col]
                if rna_init_cols:
                    tc_init_events = (
                        df[rna_init_cols[0]]
                        .apply(lambda x: x[tu_idx_rna] if len(x) > tu_idx_rna else 0)
                        .values
                    )
                else:
                    tc_init_events = None

                # Translation initiation events (robust, short version)
                ribo_init_cols = [
                    col for col in df.columns if "ribosome_init_event" in col
                ]
                if not ribo_init_cols or monomer_id is None or monomer_idx_bulk is None:
                    if not ribo_init_cols:
                        print(
                            f"{label}: ERROR - No 'ribosome_init_event' columns found in DataFrame."
                        )
                    else:
                        print(
                            f"{label}: WARNING - monomer_id {monomer_id} not present in bulk_ids, setting tl_init_events to zeros."
                        )
                    tl_init_events = np.zeros(len(df))
                else:
                    monomer_idx = int(monomer_idx_bulk)
                    tl_init_events = (
                        df[ribo_init_cols[0]]
                        .apply(lambda x: x[monomer_idx] if len(x) > monomer_idx else 0)
                        .values
                    )

                mRNA_counts = mRNA_counts.astype(np.int32)
                protein_counts = protein_counts.astype(np.int32)
                mrna_change = mRNA_counts[-1] - mRNA_counts[0]
                protein_change = protein_counts[-1] - protein_counts[0]

                results[label] = {
                    "variant": variant,
                    "generation": generation,
                    "time": df["time"].values,
                    "mRNA": mRNA_counts,
                    "protein": protein_counts,
                    "transcription_success_count": mrna_change
                    if mrna_change > 0
                    else 0,
                    "translation_success_count": protein_change
                    if protein_change > 0
                    else 0,
                    "tc_init_events": tc_init_events,
                    "tl_init_events": tl_init_events,
                }

                print(
                    f"{label}: Loaded {len(df)} timepoints, mRNA: {mRNA_counts[0]}→{mRNA_counts[-1]}, Protein: {protein_counts[0]}→{protein_counts[-1]}"
                )

            except Exception as e:
                print(f"{label}: Error - {e}")

    if not results:
        return None

    # Plot
    if plot:
        n_plots = 2
        has_tc_init = any(
            "tc_init_events" in r and r["tc_init_events"] is not None
            for r in results.values()
        )
        has_tl_init = any(
            "tl_init_events" in r and r["tl_init_events"] is not None
            for r in results.values()
        )
        if has_tc_init:
            n_plots += 1
        if has_tl_init:
            n_plots += 1

        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

        if n_plots == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        plot_idx = 0

        # mRNA
        for (label, data), color in zip(results.items(), colors):
            axes[plot_idx].plot(
                data["time"],
                data["mRNA"],
                linewidth=2,
                label=label,
                color=color,
                alpha=0.8,
            )
        axes[plot_idx].set_ylabel("mRNA Count", fontsize=12)
        axes[plot_idx].set_title(
            f"mRNA Dynamics: {gene_id} ({tu_id})", fontsize=14, fontweight="bold"
        )
        axes[plot_idx].legend(loc="best", ncol=min(3, len(results)))
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

        # Protein
        if monomer_id:
            for (label, data), color in zip(results.items(), colors):
                axes[plot_idx].plot(
                    data["time"],
                    data["protein"],
                    linewidth=2,
                    label=label,
                    color=color,
                    alpha=0.8,
                )
            axes[plot_idx].set_ylabel("Protein Count", fontsize=12)
            axes[plot_idx].set_title(
                f"Protein Dynamics: {monomer_id}", fontsize=14, fontweight="bold"
            )
            axes[plot_idx].legend(loc="best", ncol=min(3, len(results)))
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # Transcription initiation events
        if has_tc_init:
            for (label, data), color in zip(results.items(), colors):
                if data.get("tc_init_events") is not None:
                    axes[plot_idx].plot(
                        data["time"],
                        data["tc_init_events"],
                        linewidth=1.5,
                        label=label,
                        color=color,
                        alpha=0.8,
                    )
            axes[plot_idx].set_ylabel("Transcription Initiation", fontsize=12)
            axes[plot_idx].set_title(
                "Transcription Initiation Events", fontsize=14, fontweight="bold"
            )
            axes[plot_idx].legend(loc="best", ncol=min(3, len(results)))
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        # Translation initiation events
        if has_tl_init:
            for (label, data), color in zip(results.items(), colors):
                if data.get("tl_init_events") is not None:
                    axes[plot_idx].plot(
                        data["time"],
                        data["tl_init_events"],
                        linewidth=1.5,
                        label=label,
                        color=color,
                        alpha=0.8,
                    )
            axes[plot_idx].set_ylabel("Translation Initiation", fontsize=12)
            axes[plot_idx].set_title(
                "Translation Initiation Events", fontsize=14, fontweight="bold"
            )
            axes[plot_idx].legend(loc="best", ncol=min(3, len(results)))
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1

        axes[-1].set_xlabel("Time (seconds)", fontsize=12)
        plt.tight_layout()

        if save:
            save_results(results, gene_id, project_folder, variants)

        plt.show()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    for label, data in results.items():
        print(f"\n{label}:")
        print(f"  Duration: {data['time'][-1] - data['time'][0]}s")
        print(f"  mRNA:    {data['mRNA'][0]} → {data['mRNA'][-1]}")
        print(f"  Protein: {data['protein'][0]} → {data['protein'][-1]}")
        if data["tc_init_events"] is not None:
            total_inits = data["tc_init_events"].sum()
            print(
                f"  Total transcription initiations: {total_inits} → {'✓ Gene Inhibited' if total_inits == 0 else '✗ Gene Working'}"
            )

    return results


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Track knockout dynamics for a gene across variants and generations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults (EG10030, gene_knockout_metabolic1, variant 4 (where the selected gene is knocked out), generations 1,2,3)
  python gene_expression_trace.py
  
  # Override specific parameters
  python gene_expression_trace.py --gene-id EG10367 --generations 0 1 2
  
  # Specify all parameters
  python gene_expression_trace.py --gene-id EG10003 --project all_media-conditions --variants 0 --generations 0
   
  # Multiple variants
  python gene_expression_trace.py --variants 0 1 2
  
  # Custom figure size
  python gene_expression_trace.py --figsize 20 12
        """,
    )
    parser.add_argument(
        "--downsample_sec",
        "-d",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--gene-id", "-g", default="EG10030", help="Gene ID to track (default: EG10030)"
    )
    parser.add_argument(
        "--project",
        "-p",
        default="gene_knockout_metabolic1",
        help="Project folder name (default: gene_knockout_metabolic1)",
    )
    parser.add_argument(
        "--variants",
        "-v",
        nargs="+",
        default=["4"],
        help="Variant(s) to analyze (default: 4)",
    )
    parser.add_argument(
        "--generations",
        "-gen",
        nargs="+",
        default=["1", "2", "3"],
        help="Generation(s) to analyze (default: 1 2 3)",
    )
    parser.add_argument(
        "--figsize",
        "-f",
        nargs=2,
        type=float,
        default=[16, 10],
        help="Figure size as width height (default: 16 10)",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip plotting (only print statistics)"
    )

    args = parser.parse_args()

    # Convert to appropriate types
    variants = [int(v) for v in args.variants]
    generations = [int(g) for g in args.generations]
    figsize = tuple(args.figsize)

    print(f"Running analysis for gene: {args.gene_id}")
    print(f"Project: {args.project}")
    print(f"Variants: {variants}")
    print(f"Generations: {generations}")
    print(f"Figure size: {figsize}")
    print("-" * 80)

    # Run the analysis
    if args.no_plot:
        # Temporarily disable plotting
        import matplotlib

        matplotlib.use("Agg")

    results = track_knockout_dynamics(
        gene_id=args.gene_id,
        project_folder=args.project,
        variants=variants,
        generations=generations,
        figsize=figsize,
    )

    if results is None:
        print("\nAnalysis failed or no data found.")
        return 1

    print("\nAnalysis completed successfully!")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
