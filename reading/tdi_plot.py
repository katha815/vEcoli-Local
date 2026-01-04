import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os


def load_generation_cached(cache, gen, project_folder, variant, downsample_sec=20):
    key = (gen, variant)
    if key in cache:
        return cache[key]
    agent_id = "0" * gen
    base_path = f"/user/home/il22158/work/vEcoli/out/{project_folder}/history/experiment_id={project_folder}/variant={variant}/lineage_seed=0/generation={gen}/agent_id={agent_id}"
    pq_files = sorted(glob.glob(f"{base_path}/*.pq"))
    dfs = []
    for pq_file in pq_files:
        df_temp = pd.read_parquet(pq_file)
        if downsample_sec:
            df_temp = df_temp.iloc[::downsample_sec]
        dfs.append(df_temp)
    if not dfs:
        raise ValueError(f"No parquet files found for gen {gen}, variant {variant}")
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
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
    dtw_dist = dtw_distance(s1, s2) / len(s1)
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
    compared_variant=1,
    downsample_sec=20,
):
    cache = {}
    tdi_results = []
    for gen in generations:
        try:
            df0 = load_generation_cached(cache, gen, project_folder, 0, downsample_sec)
            df1 = load_generation_cached(
                cache, gen, project_folder, compared_variant, downsample_sec
            )
        except Exception as e:
            print(f"Failed to load gen {gen}: {e}")
            continue
        for feat in features:
            tdi = compute_tdi(df0, df1, feat)
            tdi["generation"] = gen
            tdi["feature"] = feat.split("__")[-1]
            tdi_results.append(tdi)
    return pd.DataFrame(tdi_results)


def plot_tdi_metrics(
    project_folder="all_media_conditions1",
    compared_variant=1,
    generations=range(1, 9),
    features=(
        "listeners__mass__dry_mass",
        "listeners__mass__instantaneous_growth_rate",
    ),
    save=True,
    save_dir="/user/home/il22158/work/vEcoli/reading/results/tdi",
    downsample_sec=20,
):
    tdi_df = compute_tdi_across_generations(
        project_folder=project_folder,
        features=features,
        generations=generations,
        compared_variant=compared_variant,
        downsample_sec=downsample_sec,
    )
    plot_features = [f.split("__")[-1] for f in features]
    fig, axes = plt.subplots(1, len(plot_features), figsize=(12, 5))
    for idx, feat in enumerate(plot_features):
        ax = axes[idx]
        data = tdi_df[tdi_df["feature"] == feat]
        ax.plot(
            data["generation"],
            data["composite_tdi"],
            marker="o",
            label="Composite TDI",
            color="black",
        )
        ax.plot(
            data["generation"],
            data["dtw_distance"],
            marker="s",
            label="DTW (shape)",
            alpha=0.7,
            color="steelblue",
        )
        ax.plot(
            data["generation"],
            data["duration_ratio"],
            marker="^",
            label="Duration ratio",
            alpha=0.7,
            color="coral",
        )
        ax.plot(
            data["generation"],
            data["amplitude_ratio"],
            marker="d",
            label="Amplitude ratio",
            alpha=0.7,
            color="mediumseagreen",
        )
        ax.set_title(
            f"TDI: {feat.replace('_', ' ').title()} (Variant {compared_variant} vs 0)"
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel("TDI Value")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"tdi_{project_folder}_v0_vs_v{compared_variant}.png"
        plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
    plt.show()


# Example usage:
for x in range(1, 5):
    plot_tdi_metrics(compared_variant=x)
