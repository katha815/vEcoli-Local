"""Plot TDI metrics across generations from saved parquet time-series files.

The script traces how one variant changes across generations by comparing each
requested generation against the immediately previous generation. It reports
all four TDI outputs for each adjacent generation pair and can save the
resulting plots to disk.

Sample use:
    python /user/home/il22158/work/vEcoli/reading/tdi_plot.py \
    --project-folder gene_ko_40trial_seed100_basal --variants 1:40 --generations 1:16

You can also point the script at a specific saved parquet file with
``--data-path``.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import argparse
import re


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
    }
)


def resolve_timeseries_path(project_folder, data_path=None):
    if data_path:
        return data_path
    project_name = re.sub(r"_seed\d+", "", project_folder)
    return (
        "/user/home/il22158/work/vEcoli/reading/results/growth_rate/"
        f"growth_rate_timeseries_{project_name}_all.parquet"
    )


def infer_lineage_seed(project_folder):
    match = re.search(r"seed(\d+)", project_folder)
    return int(match.group(1)) if match else None


def build_plot_filename(project_folder, variant, lineage_seed=None, save_name=None):
    if save_name:
        return save_name if save_name.endswith(".png") else f"{save_name}.png"

    project_name = re.sub(r"_seed\d+", "", project_folder)
    effective_seed = (
        lineage_seed if lineage_seed is not None else infer_lineage_seed(project_folder)
    )
    if effective_seed is not None:
        return f"tdi_{project_name}_seed{effective_seed}_v{variant}.png"
    return f"tdi_{project_name}_v{variant}.png"


def load_generation_cached(
    cache,
    gen,
    project_folder,
    variant,
    data_path=None,
    downsample_sec=None,
    lineage_seed=None,
):
    resolved_path = resolve_timeseries_path(project_folder, data_path)
    if lineage_seed is None:
        lineage_seed = infer_lineage_seed(project_folder)
    key = (resolved_path, lineage_seed, gen, variant, downsample_sec)
    if key in cache:
        return cache[key]

    if resolved_path not in cache:
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Saved parquet file not found: {resolved_path}")
        cache[resolved_path] = pd.read_parquet(resolved_path)

    df_all = cache[resolved_path]
    required_columns = {
        "variant",
        "generation",
        "time",
        "listeners__mass__instantaneous_growth_rate",
        "listeners__mass__dry_mass",
    }
    missing_columns = required_columns.difference(df_all.columns)
    if missing_columns:
        raise ValueError(
            f"Saved parquet file is missing required columns: {sorted(missing_columns)}"
        )

    df = df_all[(df_all["generation"] == gen) & (df_all["variant"] == variant)]
    if lineage_seed is not None and "lineage_seed" in df.columns:
        df = df[df["lineage_seed"] == lineage_seed]
    if df.empty:
        raise ValueError(
            f"No rows found for gen {gen}, variant {variant} in {resolved_path}"
        )

    df = df.sort_values("time").reset_index(drop=True)
    if downsample_sec:
        df = df.iloc[::downsample_sec].reset_index(drop=True)
    cache[key] = df
    return df


def dtw_distance(x, x_prime, window_ratio=1, q=1):
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
    s1 = df1[feature].values
    s2 = df2[feature].values
    dtw_dist = (
        dtw_distance(s1, s2)
        / min(len(s1), len(s2))
        / np.mean([np.mean(s1), np.mean(s2)])
    )  #
    dur1 = df1["time"].max() - df1["time"].min()
    dur2 = df2["time"].max() - df2["time"].min()
    dur_ratio = abs(dur2 / dur1 - 1)
    amp_ratio = abs(s2.max() / s1.max() - 1)
    return {
        "dtw_distance": dtw_dist,
        "duration_ratio": dur_ratio,
        "amplitude_ratio": amp_ratio,
        "composite_tdi": (dtw_dist + dur_ratio + amp_ratio) / 3,
    }


def compute_tdi_across_generations(
    project_folder="all_media_conditions1",
    features=(
        "listeners__mass__dry_mass",
        "listeners__mass__instantaneous_growth_rate",
    ),
    generations=range(1, 9),
    variant=0,
    data_path=None,
    downsample_sec=None,
    lineage_seed=None,
):
    cache = {}
    tdi_results = []
    generation_list = list(generations)
    if len(generation_list) < 2:
        return pd.DataFrame(tdi_results)

    for idx in range(len(generation_list) - 1):
        gen_prev = generation_list[idx]
        gen_curr = generation_list[idx + 1]
        try:
            df_prev = load_generation_cached(
                cache,
                gen_prev,
                project_folder,
                variant,
                data_path=data_path,
                downsample_sec=downsample_sec,
                lineage_seed=lineage_seed,
            )
            df_curr = load_generation_cached(
                cache,
                gen_curr,
                project_folder,
                variant,
                data_path=data_path,
                downsample_sec=downsample_sec,
                lineage_seed=lineage_seed,
            )
        except Exception as e:
            print(f"Failed to load gen pair {gen_prev}-{gen_curr}: {e}")
            continue
        for feat in features:
            tdi = compute_tdi(df_prev, df_curr, feat)
            tdi["generation_pair"] = f"{gen_prev}-{gen_curr}"
            tdi["feature"] = feat.split("__")[-1]
            tdi_results.append(tdi)
    return pd.DataFrame(tdi_results)


def plot_tdi_metrics(
    project_folder="all_media_conditions1",
    variant=0,
    generations=range(1, 9),
    features=(
        "listeners__mass__dry_mass",
        "listeners__mass__instantaneous_growth_rate",
    ),
    save=True,
    save_dir="/user/home/il22158/work/vEcoli/reading/results/tdi",
    save_name=None,
    data_path=None,
    downsample_sec=None,
    lineage_seed=None,
):
    tdi_df = compute_tdi_across_generations(
        project_folder=project_folder,
        features=features,
        generations=generations,
        variant=variant,
        data_path=data_path,
        downsample_sec=downsample_sec,
        lineage_seed=lineage_seed,
    )
    plot_features = [f.split("__")[-1] for f in features]
    fig, axes = plt.subplots(1, len(plot_features), figsize=(12, 5))
    for idx, feat in enumerate(plot_features):
        ax = axes[idx]
        data = tdi_df[tdi_df["feature"] == feat]
        x = np.arange(len(data))
        ax.plot(
            x,
            data["composite_tdi"],
            marker="o",
            label="Composite TDI",
            color="black",
        )
        ax.plot(
            x,
            data["dtw_distance"],
            marker="s",
            label="DTW (shape)",
            alpha=0.7,
            color="steelblue",
        )
        ax.plot(
            x,
            data["duration_ratio"],
            marker="^",
            label="Duration ratio",
            alpha=0.7,
            color="coral",
        )
        ax.plot(
            x,
            data["amplitude_ratio"],
            marker="d",
            label="Amplitude ratio",
            alpha=0.7,
            color="mediumseagreen",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(data["generation_pair"], rotation=45, ha="right")
        ax.set_title(f"TDI: {feat.replace('_', ' ').title()} (Variant {variant})")
        ax.set_xlabel("Generation")
        ax.set_ylabel("TDI Value")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        os.makedirs(save_dir, exist_ok=True)
        fname = build_plot_filename(
            project_folder,
            variant,
            lineage_seed=lineage_seed,
            save_name=save_name,
        )
        plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
    plt.show()


def parse_generation_range(spec):
    if ":" in spec:
        start_str, end_str = spec.split(":", 1)
        start = int(start_str)
        end = int(end_str)
        step = 1 if end >= start else -1
        return range(start, end + step, step)
    return [int(item) for item in spec.split(",") if item.strip()]


def parse_variant_list(spec):
    variants = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start_str, end_str = item.split(":", 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            variants.extend(range(start, end + step, step))
        else:
            variants.append(int(item))
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-folder",
        default="all_media_conditions1",
        help="Project folder name used to derive the saved parquet filename",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional path to a saved parquet timeseries file",
    )
    parser.add_argument(
        "--variants",
        required=True,
        help="Variant list or range, for example 1,2,3 or 1:40",
    )
    parser.add_argument(
        "--generations",
        default="1:8",
        help="Generation range as start:end or a comma-separated list, for example 1:8 or 1,2,4.",
    )
    parser.add_argument(
        "--downsample-sec",
        type=int,
        default=None,
        help="Row stride used when downsampling each parquet file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional lineage seed override; defaults to the seed embedded in --project-folder",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Display plots without saving them",
    )
    parser.add_argument(
        "--save-dir",
        default="/user/home/il22158/work/vEcoli/reading/results/tdi",
        help="Directory used when saving plots",
    )
    parser.add_argument(
        "--save-name",
        default=None,
        help="Optional output filename override; use this to customize the saved plot name",
    )
    args = parser.parse_args()

    generations = parse_generation_range(args.generations)
    variants = parse_variant_list(args.variants)

    for variant in variants:
        plot_tdi_metrics(
            project_folder=args.project_folder,
            variant=variant,
            generations=generations,
            save=not args.no_save,
            save_dir=args.save_dir,
            save_name=args.save_name,
            data_path=args.data_path,
            downsample_sec=args.downsample_sec,
            lineage_seed=args.seed,
        )


if __name__ == "__main__":
    main()
