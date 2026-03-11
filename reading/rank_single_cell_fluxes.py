"""
Rank single-cell exchange fluxes from a vEcoli simulation output folder.

Usage:
  python reading/rank_single_cell_fluxes.py [OPTIONS]

Options (all optional, defaults shown below):
  --base_out_dir   path to the vEcoli out/<project> folder
  --variants       space-separated variant IDs, e.g. --variants 0 1 2
                   (default: all variants in metadata.json)
  --generations    space-separated generation numbers, e.g. --generations 1 8
  --lineage_seed   lineage seed integer (default: 0)
  --threshold      |flux| threshold to count a variant as active (default: 1e-10)
  --out_dir        directory to save the ranked CSV
"""

import argparse
from pathlib import Path

# =============================================================================
# DEFAULTS
# =============================================================================
_DEFAULTS = dict(
    base_out_dir="/user/home/il22158/work/vEcoli/out/all_media_conditions1",
    variants=None,  # None = all from metadata.json
    generations=[1, 2, 3, 4, 5, 6, 7, 8],
    lineage_seed=0,
    threshold=1e-10,
    out_dir="/user/home/il22158/work/vEcoli/reading/results/single_cell_fluxes",
)
# =============================================================================


def build_parser():
    parser = argparse.ArgumentParser(
        description="Rank single-cell exchange fluxes from a vEcoli simulation output folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base_out_dir",
        default=_DEFAULTS["base_out_dir"],
        help="Path to out/<project> folder",
    )
    parser.add_argument(
        "--variants",
        default=None,
        nargs="+",
        type=int,
        help="Variant IDs to include (omit = all)",
    )
    parser.add_argument(
        "--generations",
        default=_DEFAULTS["generations"],
        nargs="+",
        type=int,
        help="Generation numbers to aggregate",
    )
    parser.add_argument(
        "--lineage_seed",
        default=_DEFAULTS["lineage_seed"],
        type=int,
        help="Lineage seed",
    )
    parser.add_argument(
        "--threshold",
        default=_DEFAULTS["threshold"],
        type=float,
        help="|flux| threshold for active variants",
    )
    parser.add_argument(
        "--out_dir",
        default=_DEFAULTS["out_dir"],
        help="Output directory for ranked CSV",
    )
    return parser


def main():
    args = build_parser().parse_args()

    # Heavy imports AFTER argparse so --help exits immediately
    import glob
    import json
    import os
    import pickle

    import numpy as np
    import pandas as pd

    VARIANTS = args.variants  # None means all
    GENERATIONS = args.generations
    LINEAGE_SEED = args.lineage_seed
    THRESHOLD = args.threshold
    OUT_DIR = args.out_dir

    # Resolve bare project name to full path
    _REPO_ROOT = Path(__file__).resolve().parent.parent  # vEcoli root
    _given = Path(args.base_out_dir)
    if _given.is_absolute() and _given.exists():
        BASE_OUT_DIR = str(_given)
    elif (_REPO_ROOT / "out" / args.base_out_dir).exists():
        BASE_OUT_DIR = str(_REPO_ROOT / "out" / args.base_out_dir)
    elif (_REPO_ROOT / args.base_out_dir).exists():
        BASE_OUT_DIR = str(_REPO_ROOT / args.base_out_dir)
    else:
        BASE_OUT_DIR = str(_given)

    exchange_col = "listeners__fba_results__external_exchange_fluxes"
    project = Path(BASE_OUT_DIR).name
    print(f"Resolved base_out_dir: {BASE_OUT_DIR}")

    metadata_file = f"{BASE_OUT_DIR}/variant_sim_data/metadata.json"
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    all_variant_ids = sorted(int(k) for k in metadata.get("condition", {}).keys())
    variant_ids = VARIANTS if VARIANTS is not None else all_variant_ids
    print(f"Project: {project}")
    print(f"Variants to process: {variant_ids}")
    print(f"Generations included: {GENERATIONS}")
    print()

    sim_data_path = f"{BASE_OUT_DIR}/parca/kb/simData.cPickle"
    with open(sim_data_path, "rb") as f:
        sim_data = pickle.load(f)
    exchange_names = list(sim_data.external_state.all_external_exchange_molecules)
    print(f"Loaded {len(exchange_names)} exchange names from simData.cPickle")

    all_variant_stats = []

    for variant_id in variant_ids:
        for generation in GENERATIONS:
            variant_path = (
                f"{BASE_OUT_DIR}/history/experiment_id={project}"
                f"/variant={variant_id}/lineage_seed={LINEAGE_SEED}"
                f"/generation={generation}/agent_id=0*"
            )
            pq_files = sorted(glob.glob(f"{variant_path}/*.pq"))

            if not pq_files:
                variant_path_exact = (
                    f"{BASE_OUT_DIR}/history/experiment_id={project}"
                    f"/variant={variant_id}/lineage_seed={LINEAGE_SEED}"
                    f"/generation={generation}/agent_id=0"
                )
                pq_files = sorted(glob.glob(f"{variant_path_exact}/*.pq"))

            if not pq_files:
                print(
                    f"  Variant {variant_id} gen {generation}: no pq files found, skipping"
                )
                continue

            df_variant = pd.read_parquet(pq_files[0])
            if exchange_col not in df_variant.columns:
                print(
                    f"  Variant {variant_id} gen {generation}: exchange column missing, skipping"
                )
                continue

            fluxes = np.array(
                [df_variant[exchange_col].iloc[i] for i in range(len(df_variant))]
            )
            means = np.mean(np.abs(fluxes), axis=0)
            maxs = np.max(np.abs(fluxes), axis=0)

            all_variant_stats.append(
                {
                    "variant_id": variant_id,
                    "generation": generation,
                    "condition": metadata["condition"].get(
                        str(variant_id), f"variant_{variant_id}"
                    ),
                    "means": means,
                    "maxs": maxs,
                }
            )
            print(
                f"  Variant {variant_id} gen {generation}: {len(df_variant)} timepoints  [{all_variant_stats[-1]['condition']}]"
            )

    print(f"\nLoaded {len(all_variant_stats)} variant-generation entries")

    all_means = np.array([s["means"] for s in all_variant_stats])
    all_maxs = np.array([s["maxs"] for s in all_variant_stats])

    global_mean = np.mean(all_means, axis=0)
    global_max = np.max(all_maxs, axis=0)
    sorted_idx = np.argsort(global_mean)[::-1]

    variant_ids_arr = np.array([s["variant_id"] for s in all_variant_stats])
    active_variant_ids_by_met = []
    for i in range(all_maxs.shape[1]):
        active_mask = all_maxs[:, i] > THRESHOLD
        active_variant_ids_by_met.append(variant_ids_arr[active_mask].tolist())

    assert len(exchange_names) == len(global_mean), (
        f"Mismatch: exchange_names={len(exchange_names)}, flux vector={len(global_mean)}"
    )

    ranked_flux_df = pd.DataFrame(
        {
            "exchange_name": np.array(exchange_names)[sorted_idx],
            "mean_abs_flux": global_mean[sorted_idx],
            "max_abs_flux": global_max[sorted_idx],
            "active_variant_ids": [active_variant_ids_by_met[i] for i in sorted_idx],
        }
    )

    gens_str = "g" + "_".join(str(g) for g in sorted(GENERATIONS))
    variants_str = (
        "allvariants"
        if VARIANTS is None
        else f"v{'_'.join(str(v) for v in sorted(VARIANTS))}"
    )
    source_suffix = f"{project}__{gens_str}__seed{LINEAGE_SEED}__{variants_str}_thresh{THRESHOLD:.0e}"

    os.makedirs(OUT_DIR, exist_ok=True)
    full_path = f"{OUT_DIR}/exchange_flux_ranked__{source_suffix}.csv"
    ranked_flux_df.to_csv(full_path, index=False)
    print(f"\nSaved: {full_path}")

    sorted_means = global_mean[sorted_idx]
    cumsum = np.cumsum(sorted_means)
    total = cumsum[-1]
    print(
        f"\nTop 6  account for: {100 * cumsum[5] / total:.2f}% of total exchange flux"
    )
    print(f"Top 10 account for: {100 * cumsum[9] / total:.2f}% of total exchange flux")
    print("Top 10 exchange names:")
    for i, name in enumerate(ranked_flux_df["exchange_name"].head(10), 1):
        print(f"  {i:2d}. {name}")


if __name__ == "__main__":
    main()
