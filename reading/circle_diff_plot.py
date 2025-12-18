#!/usr/bin/env python3
"""
Calculate Time Distortion Index (TDI) between consecutive generations of bacterial simulations.
Uses DTW-based shape comparison without normalization to preserve absolute scale information.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

# Set matplotlib font to avoid Arial warning
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "sans-serif"]


def load_generation(
    gen, project_folder="all_media_conditions1", variant=0, downsample_sec=None
):
    """Load and concatenate parquet files for a generation with optional downsampling"""
    agent_id = "0" * gen
    base_path = f"/user/home/il22158/work/vEcoli/out/{project_folder}/history/experiment_id={project_folder}/variant={variant}/lineage_seed=0/generation={gen}/agent_id={agent_id}"
    pq_files = sorted(glob.glob(f"{base_path}/*.pq"))

    if len(pq_files) == 0:
        raise FileNotFoundError(
            f"No parquet files found for generation {gen} in {base_path}"
        )

    print(f"Gen {gen}: Found {len(pq_files)} files", end="")

    # Load and downsample during reading
    dfs = []
    total_before = 0
    for pq_file in pq_files:
        df_temp = pd.read_parquet(pq_file)
        total_before += len(df_temp)
        if downsample_sec:
            df_temp = df_temp.iloc[::downsample_sec]
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)

    print(f" -> {len(df)} rows ({100 * len(df) / total_before:.1f}%)")
    return df


def dtw_distance(x, x_prime, window_ratio=1, q=1):
    """Dynamic Time Warping distance with window constraint"""
    n, m = len(x), len(x_prime)
    window = int(window_ratio * max(n, m))
    window = max(window, abs(n - m))
    R = np.full((n, m), np.inf)
    R[0, 0] = 0
    for i in range(n):
        for j in range(m):
            cost = abs(x[i] - x_prime[j]) ** q
            if abs(i - j) > window:
                R[i, j] = np.inf
                continue
            if i == 0 and j == 0:
                R[i, j] = cost
            else:
                R[i, j] = cost + min(
                    R[i - 1, j] if i > 0 else np.inf,
                    R[i, j - 1] if j > 0 else np.inf,
                    R[i - 1, j - 1] if (i > 0 and j > 0) else np.inf,
                )
    return R[n - 1, m - 1] ** (1.0 / q)


def compute_tdi(df1, df2, feature):
    """Compute composite Time Distortion Index with 3 components"""
    s1 = df1[feature].values
    s2 = df2[feature].values

    # Shape distortion (normalized DTW)
    dtw_dist = dtw_distance(s1, s2) / len(s1)

    # Temporal distortion (duration ratio)
    dur1 = df1["time"].max() - df1["time"].min()
    dur2 = df2["time"].max() - df2["time"].min()
    dur_ratio = abs(dur2 / dur1 - 1)

    # Amplitude distortion (mean ratio)
    amp_ratio = abs(s2.mean() / s1.mean() - 1)

    return {
        "dtw_distance": dtw_dist,
        "duration_ratio": dur_ratio,
        "amplitude_ratio": amp_ratio,
        "composite_tdi": (dtw_dist + dur_ratio + amp_ratio) / 3,
    }


def analyze_time_distortion(
    project_folder,
    variant=0,
    generations=range(1, 9),
    features=None,
    downsample_sec=20,
    window_ratio=1,
    q=1,
):
    """
    Main analysis function: calculate TDI across generations

    Args:
        project_folder: Name of the project folder in vEcoli/out/
        variant: Variant ID to analyze
        generations: Range or list of generation numbers
        features: List of feature column names to analyze
        downsample_sec: Downsample interval (take every Nth row)
        window_ratio: DTW window constraint ratio (default: 1)
        q: DTW distance metric order (default: 1 for Manhattan)

    Returns:
        tdi_df: DataFrame with TDI results
    """
    if features is None:
        features = [
            "listeners__mass__dry_mass",
            "listeners__mass__instantaneous_growth_rate",
        ]

    generations = list(generations)

    # Load all generations
    print(f"\nLoading {len(generations)} generations from {project_folder}...")
    gen_data = {}
    for g in generations:
        gen_data[g] = load_generation(
            g,
            project_folder=project_folder,
            variant=variant,
            downsample_sec=downsample_sec,
        )

    # Calculate TDI for all features between consecutive generations
    print("\nCalculating TDI between consecutive generations...")
    tdi_results = []
    for feat in features:
        feat_name = feat.split("__")[-1]
        print(f"\n{feat_name}:")
        for i in range(len(generations) - 1):
            gen1, gen2 = generations[i], generations[i + 1]

            # Compute TDI with custom DTW parameters
            s1 = gen_data[gen1][feat].values
            s2 = gen_data[gen2][feat].values

            # Shape distortion (normalized DTW)
            dtw_dist = dtw_distance(s1, s2, window_ratio=window_ratio, q=q) / len(s1)

            # Temporal distortion (duration ratio)
            dur1 = gen_data[gen1]["time"].max() - gen_data[gen1]["time"].min()
            dur2 = gen_data[gen2]["time"].max() - gen_data[gen2]["time"].min()
            dur_ratio = abs(dur2 / dur1 - 1)

            # Amplitude distortion (mean ratio)
            amp_ratio = abs(s2.mean() / s1.mean() - 1)

            tdi = {
                "dtw_distance": dtw_dist,
                "duration_ratio": dur_ratio,
                "amplitude_ratio": amp_ratio,
                "composite_tdi": (dtw_dist + dur_ratio + amp_ratio) / 3,
                "generation_pair": f"{gen1}-{gen2}",
                "feature": feat_name,
            }
            tdi_results.append(tdi)
            print(
                f"  {gen1}->{gen2}: TDI={tdi['composite_tdi']:.4f} "
                f"(DTW={tdi['dtw_distance']:.4f}, Dur={tdi['duration_ratio']:.4f}, "
                f"Amp={tdi['amplitude_ratio']:.4f})"
            )

    tdi_df = pd.DataFrame(tdi_results)

    return tdi_df


def plot_tdi_results(tdi_df, project_folder, variant=0):
    """Create visualization of TDI results"""
    output_path = f"/user/home/il22158/work/vEcoli/reading/results/circle_diff/{project_folder}_v{variant}"
    features = tdi_df["feature"].unique()
    n_features = len(features)

    fig, axes = plt.subplots(1, n_features, figsize=(7 * n_features, 5))
    if n_features == 1:
        axes = [axes]

    feature_labels = {
        "dry_mass": "Dry Mass",
        "instantaneous_growth_rate": "Growth Rate",
    }

    for idx, feat_name in enumerate(features):
        ax = axes[idx]
        feat_data = tdi_df[tdi_df["feature"] == feat_name]
        x = np.arange(len(feat_data))

        ax.plot(
            x,
            feat_data["composite_tdi"],
            marker="o",
            label="Composite TDI",
            linewidth=2.5,
            markersize=8,
            color="black",
        )
        ax.plot(
            x,
            feat_data["dtw_distance"],
            marker="s",
            label="DTW (shape)",
            linewidth=1.5,
            markersize=6,
            alpha=0.7,
            color="steelblue",
        )
        ax.plot(
            x,
            feat_data["duration_ratio"],
            marker="^",
            label="Duration ratio",
            linewidth=1.5,
            markersize=6,
            alpha=0.7,
            color="coral",
        )
        ax.plot(
            x,
            feat_data["amplitude_ratio"],
            marker="d",
            label="Amplitude ratio",
            linewidth=1.5,
            markersize=6,
            alpha=0.7,
            color="mediumseagreen",
        )

        ax.set_xlabel("Generation Pair", fontsize=12)
        ax.set_ylabel("TDI Value", fontsize=12)
        feat_label = feature_labels.get(feat_name, feat_name)
        ax.set_title(
            f"Time Distortion Index - {feat_label}", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(feat_data["generation_pair"], rotation=45)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = f"{output_path}/tdi_analysis_{project_folder}_v{variant}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"\n✓ Plot saved: {plot_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Time Distortion Index between generations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python circle_diff_plot.py --project_folder all_media_conditions1
  python circle_diff_plot.py --project_folder gene_knockout_test3 --generations 1 2 3 4 --downsample 10
  python circle_diff_plot.py --project_folder all_media_conditions1 --window_ratio 0.2 --q 2
        """,
    )

    parser.add_argument(
        "--project_folder",
        "-p",
        type=str,
        default="all_media_conditions1",
        help="Project folder name (default: all_media_conditions1)",
    )

    parser.add_argument(
        "--variant", "-v", type=int, default=0, help="Variant ID (default: 0)"
    )

    parser.add_argument(
        "--generations",
        "-g",
        type=int,
        nargs="+",
        default=list(range(1, 9)),
        help="Generation numbers to analyze (default: 1 2 3 4 5 6 7 8)",
    )

    parser.add_argument(
        "--downsample",
        "-d",
        type=int,
        default=20,
        help="Downsample interval - take every Nth row (default: 20)",
    )

    parser.add_argument(
        "--window_ratio",
        "-w",
        type=float,
        default=1,
        help="DTW window constraint ratio (default: 1)",
    )

    parser.add_argument(
        "--q",
        type=int,
        default=1,
        help="DTW distance metric order: 1=Manhattan, 2=Euclidean (default: 1)",
    )

    # parser.add_argument('--output', '-o', type=str,
    #                    default='/user/home/il22158/work/vEcoli/reading/results/circle_diff',
    #                    help='Output directory for results')

    args = parser.parse_args()

    # Run analysis
    tdi_df = analyze_time_distortion(
        project_folder=args.project_folder,
        variant=args.variant,
        generations=args.generations,
        downsample_sec=args.downsample,
        window_ratio=args.window_ratio,
        q=args.q,
    )

    # Save results
    tdi_df.to_csv(
        f"/user/home/il22158/work/vEcoli/reading/results/circle_diff/{args.project_folder}_v{args.variant}/tdi_results_{args.project_folder}_v{args.variant}.csv",
        index=False,
    )
    print(
        f"\n✓ CSV saved: /user/home/il22158/work/vEcoli/reading/results/circle_diff/{args.project_folder}_v{args.variant}/tdi_results_{args.project_folder}_v{args.variant}.csv"
    )

    # Create plots
    plot_tdi_results(tdi_df, args.project_folder, args.variant)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
