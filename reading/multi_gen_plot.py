import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import glob


def plot_multi_generation_data(
    generations=range(1, 8),
    read_interval_sec=10,
    show_legend=True,
    project_folder="all_media_conditions1",
    save_path="/user/home/il22158/work/vEcoli/reading/results/",
    figsize=(14, 4),
):
    """
    Plot growth rate and dry mass for all variants across multiple generations with continuous timeline.

    Parameters:
    -----------
    generations : list or range
        Generation numbers to analyze (e.g., range(1,9) or [1,3,5,8])
    align_time : bool align_time=True,
        If True, create seamless continuous timeline (gen1: 0-2000s, gen2: 2000-4000s, etc.)
        If False, use absolute timestamps from experiment start
    read_interval_sec : int
        Read only one data point every N seconds from parquet files (default 10 for memory efficiency)
        Set to 1 to read all data points
    show_legend : bool
        Whether to show legend on plots
    show_generation_markers : bool
        If True, add vertical lines to mark generation boundaries
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    all_data : dict
        Nested dictionary: {variant_id: {'condition': str, 'time': array, 'growth_rate': array,
                                         'dry_mass': array}
    """

    # Load condition metadata
    metadata_file = f"/user/home/il22158/work/vEcoli/out/{project_folder}/variant_sim_data/metadata.json"
    with open(metadata_file, "r") as f:
        condition_metadata = json.load(f)

    # Convert generations to list if it's a range
    gen_list = list(generations)

    print(f"Loading generations: {gen_list}")
    print(f"Downsampling: every {read_interval_sec} seconds")
    # print(f"Timeline mode: {'Continuous' if align_time else 'Absolute'}\n")

    # Initialize storage for all variants
    all_data = {}

    # Loop through variants (outer loop to maintain color consistency)
    for variant_id in range(5):
        # Get condition name from metadata
        variant_str = str(variant_id)
        if variant_str in condition_metadata["condition"]:
            condition_info = condition_metadata["condition"][variant_str]
            if isinstance(condition_info, dict) and "condition" in condition_info:
                condition_name = condition_info["condition"]
            elif isinstance(condition_info, str):
                condition_name = condition_info
            else:
                condition_name = f"variant_{variant_id}"
        else:
            condition_name = f"variant_{variant_id}"

        print(f"{'=' * 60}")
        print(f"Variant {variant_id}: {condition_name}")
        print(f"{'=' * 60}")

        # Storage for this variant's data across generations
        variant_times = []
        variant_growth_rates = []
        variant_dry_masses = []
        # generation_boundaries = []
        # time_offset = 0  # For continuous timeline

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
                if read_interval_sec > 1:
                    df_temp = df_temp.iloc[::read_interval_sec]

                total_rows_after += len(df_temp)
                gen_data.append(df_temp)

            # Concatenate and sort
            df_gen = pd.concat(gen_data, ignore_index=True)
            df_gen = df_gen.sort_values("time").reset_index(drop=True)

            print(
                f"  ✓ Gen {gen}: {len(pq_files)} files, {total_rows_after} points (from {total_rows_before})"
            )

            # Store data
            variant_times.append(df_gen["time"].values)
            variant_growth_rates.append(
                df_gen["listeners__mass__instantaneous_growth_rate"].values
            )
            variant_dry_masses.append(df_gen["listeners__mass__dry_mass"].values)

        # Concatenate all generations for this variant
        if len(variant_times) > 0:
            all_data[variant_id] = {
                "condition": condition_name,
                "time": np.concatenate(variant_times),
                "growth_rate": np.concatenate(variant_growth_rates),
                "dry_mass": np.concatenate(variant_dry_masses),
            }
            print(
                f"  → Total: {len(all_data[variant_id]['time'])} points across {len(gen_list)} generations\n"
            )

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    colors = ["steelblue", "coral", "green", "purple", "orange"]

    # Plot all variants
    for variant_id, data in all_data.items():
        label = data["condition"]

        # Growth rate
        ax1.plot(
            data["time"],
            data["growth_rate"],
            linewidth=1.5,
            color=colors[variant_id],
            alpha=0.8,
            label=label,
        )

        # Dry mass
        ax2.plot(
            data["time"],
            data["dry_mass"],
            linewidth=1.5,
            color=colors[variant_id],
            alpha=0.8,
            label=label,
        )

        # # Add generation boundary markers (only once per variant to avoid legend clutter)
        # if show_generation_markers and variant_id == 0:
        #     for boundary in data['generation_boundaries']:
        #         ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        #         ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    # Format plots
    # time_mode = "Continuous Timeline" if align_time else "Absolute Time"
    gen_str = (
        f"Generations {min(gen_list)}-{max(gen_list)}"
        if len(gen_list) > 1
        else f"Generation {gen_list[0]}"
    )

    ax1.set_xlabel("Time (s), sampled every {read_interval_sec}s", fontsize=12)
    ax1.set_ylabel("Growth rate (/s)", fontsize=12)
    # ax1.set_title(f'Instantaneous Growth Rate - {gen_str}, sampled every {read_interval_sec}s)',
    #               fontsize=14, fontweight='bold')
    if show_legend:
        ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Time (s), sampled every {read_interval_sec}s", fontsize=12)
    ax2.set_ylabel("Dry mass (fg)", fontsize=12)
    # ax2.set_title(f'Dry Mass - {gen_str}, sampled every {read_interval_sec}s)',
    #               fontsize=14, fontweight='bold')
    if show_legend:
        ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.savefig(
        f"{save_path}{project_folder}_multi_gen_plot_{min(gen_list)}_{max(gen_list)}_every{read_interval_sec}s.png",
        dpi=300,
    )

    # Print statistics
    print(f"\n{'=' * 60}")
    print(f"Summary Statistics - {gen_str}")
    print(f"{'=' * 60}")

    for variant_id, data in all_data.items():
        condition = data["condition"]
        print(f"\n{condition}:")
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

    return all_data


# Plot all 8 generations with continuous timeline (default 10s sampling)
data_all = plot_multi_generation_data(generations=range(1, 8))
