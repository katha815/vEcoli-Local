import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import argparse


def plot_growth_data_general(
    generation=range(1, 8),
    project_folder="gene_knockout_test",
    variant_key="gene_knockout",
    read_interval_sec=None,
    show_legend=True,
    figsize=(14, 8),
):
    """
    Plot growth rate and dry mass for all variants across single or multiple generations.

    Parameters:
    -----------
    generation : int or list/range
        Single generation number (e.g., 1) or multiple generations (e.g., range(1,8) or [1,3,5])
    project_folder : str
        Path to the project output folder
    variant_key : str
        Key in metadata to look for variant information (e.g., "gene_knockout", "condition")
    read_interval_sec : int or None
        If specified, take only one data point every N seconds to reduce data density
        If None, plot all data points
    show_legend : bool
        Whether to show legend on plots
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    all_variants : dict
        Dictionary containing data for all variants
    """

    # Convert single generation to list
    if isinstance(generation, int):
        gen_list = [generation]
    else:
        gen_list = list(generation)

    # Load condition metadata
    metadata_file = f"/user/home/il22158/work/vEcoli/out/{project_folder}/variant_sim_data/metadata.json"
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Get the variant metadata dictionary
    if variant_key in metadata:
        variant_metadata = metadata[variant_key]
    else:
        print(
            f"Warning: '{variant_key}' not found in metadata. Available keys: {list(metadata.keys())}"
        )
        variant_metadata = {}

    print(f"Loading generations: {gen_list}")
    if read_interval_sec:
        print(f"Downsampling: every {read_interval_sec} seconds\n")

    # Get all variant IDs
    variant_ids = sorted([int(k) for k in variant_metadata.keys()])

    # Initialize storage for all variants
    all_variants = {}

    # Loop through variants
    for variant_id in variant_ids:
        # Get label from metadata
        variant_str = str(variant_id)
        if variant_str in variant_metadata:
            variant_info = variant_metadata[variant_str]

            # Convert to label string
            if isinstance(variant_info, str):
                label = variant_info
            elif isinstance(variant_info, dict):
                if "genes_to_knockout" in variant_info:
                    genes = variant_info["genes_to_knockout"]
                    if genes:
                        label = f"KO: {', '.join(genes)}"
                    else:
                        label = "No knockout"
                elif "condition" in variant_info:
                    label = variant_info["condition"]
                else:
                    label = str(variant_info)
            else:
                label = f"variant_{variant_id}"
        else:
            label = f"variant_{variant_id}"

        print(f"{'=' * 60}")
        print(f"Variant {variant_id}: {label}")
        print(f"{'=' * 60}")

        # Storage for this variant's data across generations
        variant_times = []
        variant_growth_rates = []
        variant_dry_masses = []

        # Loop through generations for this variant
        for gen in gen_list:
            # Agent ID has number of zeros equal to generation number
            agent_id = "0" * gen

            # Construct path
            base_path = f"/user/home/il22158/work/vEcoli/out/{project_folder}/history/experiment_id={project_folder}/variant={variant_id}/lineage_seed=0/generation={gen}/agent_id={agent_id}"

            # Find all parquet files
            pq_files = sorted(glob.glob(f"{base_path}/*.pq"))

            if len(pq_files) == 0:
                print(f"  ⚠ Gen {gen}: No files found")
                continue

            # Read and downsample data during loading
            gen_data = []
            total_rows_before = 0
            total_rows_after = 0

            for pq_file in pq_files:
                df_temp = pd.read_parquet(pq_file)
                total_rows_before += len(df_temp)

                # Downsample during reading
                if read_interval_sec and read_interval_sec > 1:
                    df_temp = df_temp.iloc[::read_interval_sec]

                total_rows_after += len(df_temp)
                gen_data.append(df_temp)

            # Concatenate and sort
            df_gen = pd.concat(gen_data, ignore_index=True)
            df_gen = df_gen.sort_values("time").reset_index(drop=True)

            print(
                f"  ✓ Gen {gen}: {len(pq_files)} files, {total_rows_after} points (from {total_rows_before})"
            )

            # Store data (using absolute time)
            variant_times.append(df_gen["time"].values)
            variant_growth_rates.append(
                df_gen["listeners__mass__instantaneous_growth_rate"].values
            )
            variant_dry_masses.append(df_gen["listeners__mass__dry_mass"].values)

        # Concatenate all generations for this variant
        if len(variant_times) > 0:
            all_variants[variant_id] = {
                "label": label,
                "time": np.concatenate(variant_times),
                "growth_rate": np.concatenate(variant_growth_rates),
                "dry_mass": np.concatenate(variant_dry_masses),
            }
            print(
                f"  → Total: {len(all_variants[variant_id]['time'])} points across {len(gen_list)} generation(s)\n"
            )

    # Plot all variants
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_variants)))

    for idx, (variant_id, data) in enumerate(all_variants.items()):
        label = data["label"]

        # Growth rate
        ax1.plot(
            data["time"],
            data["growth_rate"],
            linewidth=1.5,
            color=colors[idx],
            alpha=0.8,
            label=label,
        )

        # Dry mass
        ax2.plot(
            data["time"],
            data["dry_mass"],
            linewidth=1.5,
            color=colors[idx],
            alpha=0.8,
            label=label,
        )

    # Format plots
    gen_str = (
        f"Generations {min(gen_list)}-{max(gen_list)}"
        if len(gen_list) > 1
        else f"Generation {gen_list[0]}"
    )
    downsample_str = (
        f" (sampled every {read_interval_sec}s)" if read_interval_sec else ""
    )

    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Growth rate (/s)", fontsize=12)
    ax1.set_title(
        f"Instantaneous Growth Rate - {gen_str}{downsample_str}",
        fontsize=14,
        fontweight="bold",
    )
    if show_legend:
        ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Dry mass (fg)", fontsize=12)
    ax2.set_title(
        f"Dry Mass - {gen_str}{downsample_str}", fontsize=14, fontweight="bold"
    )
    if show_legend:
        ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)

    save_path = "/user/home/il22158/work/vEcoli/reading/results/"

    plt.tight_layout()
    plt.savefig(f"{save_path}{project_folder}_{gen_str.replace(' ', '_')}_growth.png")
    plt.close()

    # Print statistics
    print(f"\n{'=' * 60}")
    print(f"Summary Statistics - {gen_str}")
    print(f"{'=' * 60}")

    for variant_id, data in all_variants.items():
        label = data["label"]
        print(f"\n{label} (variant {variant_id}):")
        print(
            f"  Total duration: {data['time'][-1] - data['time'][0]:.0f} s ({(data['time'][-1] - data['time'][0]) / 60:.1f} min)"
        )
        print(f"  Data points: {len(data['time'])}")
        print(f"  Generations: {len(gen_list)}")
        print(
            f"  Growth rate: mean={np.mean(data['growth_rate']):.6f} /s, std={np.std(data['growth_rate']):.6f} /s"
        )
        print(
            f"  Dry mass: {data['dry_mass'][0]:.1f} → {data['dry_mass'][-1]:.1f} fg ({data['dry_mass'][-1] / data['dry_mass'][0]:.2f}x)"
        )

    return all_variants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot vEcoli growth data")
    parser.add_argument("--generation", "-g", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--project_folder", "-p", type=str, default="gene_knockout_test"
    )
    parser.add_argument("--variant_key", "-v", type=str, default="gene_knockout")
    parser.add_argument("--read_interval_sec", "-i", type=int, default=20)
    parser.add_argument("--no_legend", action="store_true")
    parser.add_argument("--figsize", type=int, nargs=2, default=[14, 8])

    args = parser.parse_args()

    generation = (
        args.generation[0]
        if len(args.generation) == 1
        else (
            range(args.generation[0], args.generation[1])
            if len(args.generation) == 2
            else args.generation
        )
    )

    plot_growth_data_general(
        generation=generation,
        project_folder=args.project_folder,
        variant_key=args.variant_key,
        read_interval_sec=args.read_interval_sec,
        show_legend=not args.no_legend,
        figsize=tuple(args.figsize),
    )

# Example usage:

# Single generation
# data = plot_growth_data_general(generation=1, project_folder="gene_knockout_test",
#                                 variant_key="gene_knockout", read_interval_sec=10)

# Multiple generations
# data = plot_growth_data_general(generation=range(1, 9), project_folder="gene_knockout_metabolic",
#                                 variant_key="gene_knockout", read_interval_sec=20)

# Condition variants
# data = plot_growth_data_general(generation=1, project_folder="gene_knockout_test3",
#                                 variant_key="gene_knockout", read_interval_sec=20)
