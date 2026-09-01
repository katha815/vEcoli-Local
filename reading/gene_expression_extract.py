#!/usr/bin/env python3
"""
Aim:
Extract gene-expression results from raw vEcoli simulation histories.

This script mirrors the output layout used by growth_rate_extract.py while
reusing the gene-to-RNA/protein mapping logic from gene_expression_trace.py.

Default run targets:
    - gene list: reading/imported/Single_KO_RNA_names.txt
    - projects: gene_knockout_462KO_basal_operon_on_p1 ... p5

How to use:
    - First implementation / full run:
        python gene_expression_extract.py --all --projects <project> --lineage-seeds 100 101
    - Quick test run:
        python gene_expression_extract.py --trial --max-genes <n>
    - Save timeseries too:
        python gene_expression_extract.py --save-timeseries

Outputs:
    - summary CSV with per-gene transcription/translation success and mRNA/protein
        level statistics
    - optional downsampled timeseries parquet

Summary columns include:
    project, variant, generation, lineage_seed, gene_id, label,
    transcription_success_count, translation_success_count,
    total_tc_initiations, total_tl_initiations,
    mRNA/protein initial/final/mean/max and duration.
"""

import argparse
import glob
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path("/user/home/il22158/work/vEcoli")
DEFAULT_GENE_LIST = ROOT_DIR / "reading/imported/Single_KO_RNA_names.txt"
DEFAULT_PROJECTS = [
    "gene_knockout_462KO_basal_operon_on_p1",
    "gene_knockout_462KO_basal_operon_on_p2",
    "gene_knockout_462KO_basal_operon_on_p3",
    "gene_knockout_462KO_basal_operon_on_p4",
    "gene_knockout_462KO_basal_operon_on_p5",
]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "reading/results/gene_expression_extract"


class DefaultHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    pass


def load_gene_list(gene_list_path, max_genes=None):
    """Load gene IDs from a newline-delimited text file."""
    genes = []
    with open(gene_list_path, "r") as handle:
        for line in handle:
            gene_id = line.strip()
            if not gene_id or gene_id.startswith("#"):
                continue
            genes.append(gene_id)

    if max_genes is not None and max_genes > 0:
        genes = genes[:max_genes]

    return genes


def discover_variants(project_folder):
    """Discover variant IDs from metadata when available, otherwise from history paths."""
    metadata_file = ROOT_DIR / f"out/{project_folder}/variant_sim_data/metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as handle:
                metadata = json.load(handle)
            variant_keys = sorted(int(key) for key in metadata.keys())
            if variant_keys:
                return variant_keys
        except Exception:
            pass

    history_root = (
        ROOT_DIR / f"out/{project_folder}/history/experiment_id={project_folder}"
    )
    variant_ids = []
    for variant_path in sorted(glob.glob(str(history_root / "variant=*"))):
        try:
            variant_ids.append(int(Path(variant_path).name.split("=", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(variant_ids))


def get_available_generations(project_folder, variant, lineage_seed):
    """Return generations that have at least one parquet file on disk."""
    base_path = (
        ROOT_DIR
        / f"out/{project_folder}/history/experiment_id={project_folder}/variant={variant}/lineage_seed={lineage_seed}"
    )
    generations = []
    for generation_path in sorted(glob.glob(str(base_path / "generation=*"))):
        generation_name = Path(generation_path).name
        try:
            generation = int(generation_name.split("=", 1)[1])
        except (IndexError, ValueError):
            continue

        agent_paths = glob.glob(f"{generation_path}/agent_id=*/*.pq")
        if agent_paths:
            generations.append(generation)

    return sorted(set(generations))


@lru_cache(maxsize=None)
def load_sim_data(project_folder, variant):
    """Load the sim_data pickle for a project/variant."""
    variant_sim_data_path = (
        ROOT_DIR / f"out/{project_folder}/variant_sim_data/{variant}.cPickle"
    )
    with open(variant_sim_data_path, "rb") as handle:
        return pickle.load(handle)


def resolve_gene_info(sim_data, gene_id):
    """Resolve a gene ID to cistron, RNA, and monomer metadata."""
    cistron_data = sim_data.process.transcription.cistron_data
    gene_mask = cistron_data["gene_id"] == gene_id
    if not gene_mask.any():
        return None

    cistron_id = cistron_data["id"][gene_mask][0]
    rna_idxs = sim_data.process.transcription.cistron_id_to_rna_indexes(cistron_id)
    rna_idx = int(rna_idxs[0]) if hasattr(rna_idxs, "__iter__") else int(rna_idxs)
    rna_id = sim_data.process.transcription.rna_data["id"][rna_idx]

    monomer_data = sim_data.process.translation.monomer_data
    monomer_mask = monomer_data["cistron_id"] == cistron_id
    monomer_id = monomer_data["id"][monomer_mask][0] if monomer_mask.any() else None

    all_rna_ids = sim_data.process.transcription.rna_data["id"]
    rna_idx_rna = np.where(all_rna_ids == rna_id)[0][0]

    return {
        "gene_id": gene_id,
        "cistron_id": cistron_id,
        "rna_id": rna_id,
        "monomer_id": monomer_id,
        "rna_idx_rna": int(rna_idx_rna),
    }


def _listify_metadata(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def extract_gene_expression_data(
    project_folder,
    gene_ids,
    generation=None,
    read_interval_sec=1,
    max_variants=None,
    save_timeseries=False,
    lineage_seed=0,
):
    """Extract transcription/translation success and expression levels for genes."""
    if not gene_ids:
        return pd.DataFrame(), pd.DataFrame()

    variants = discover_variants(project_folder)
    if max_variants is not None:
        variants = variants[:max_variants]

    gen_list = (
        None
        if generation is None
        else ([generation] if isinstance(generation, int) else list(generation))
    )

    sim_data = None
    if variants:
        try:
            sim_data = load_sim_data(project_folder, variants[0])
        except FileNotFoundError:
            print(f"⚠ {project_folder}: sim_data not found for variant {variants[0]}")
            return pd.DataFrame(), pd.DataFrame()

    if sim_data is None:
        return pd.DataFrame(), pd.DataFrame()

    gene_info_rows = []
    missing_genes = []
    for gene_id in gene_ids:
        gene_info = resolve_gene_info(sim_data, gene_id)
        if gene_info is None:
            missing_genes.append(gene_id)
            continue
        gene_info_rows.append(gene_info)

    if missing_genes:
        print(
            f"⚠ {project_folder}: {len(missing_genes)} genes were not found in sim_data; they will be skipped."
        )

    if not gene_info_rows:
        return pd.DataFrame(), pd.DataFrame()

    gene_info_df = pd.DataFrame(gene_info_rows)

    summary_rows = []
    timeseries_frames = []

    for variant in variants:
        try:
            sim_data = load_sim_data(project_folder, variant)
        except FileNotFoundError:
            print(f"⚠ {project_folder}: missing sim_data for variant {variant}")
            continue

        if gen_list is None:
            variant_gen_list = get_available_generations(
                project_folder, variant, lineage_seed
            )
        else:
            variant_gen_list = gen_list

        if not variant_gen_list:
            continue

        for generation in variant_gen_list:
            agent_id = "0" * generation
            history_path = (
                ROOT_DIR
                / f"out/{project_folder}/history/experiment_id={project_folder}/variant={variant}/lineage_seed={lineage_seed}/generation={generation}/agent_id={agent_id}"
            )
            pq_files = sorted(glob.glob(f"{history_path}/*.pq"))
            if not pq_files:
                print(
                    f"{project_folder} V{variant} G{generation}: no history files found"
                )
                continue

            df = (
                pd.concat(
                    [pd.read_parquet(file_path) for file_path in pq_files],
                    ignore_index=True,
                )
                .sort_values("time")
                .reset_index(drop=True)
            )

            original_len = len(df)
            if read_interval_sec and read_interval_sec > 1:
                df = df[df["time"] % read_interval_sec == 0].reset_index(drop=True)
                print(
                    f"{project_folder} V{variant} G{generation}: downsample {read_interval_sec}s | {original_len} -> {len(df)} rows"
                )

            if df.empty:
                continue

            config_path = (
                ROOT_DIR
                / f"out/{project_folder}/configuration/experiment_id={project_folder}/variant={variant}/lineage_seed={lineage_seed}/generation={generation}/agent_id={agent_id}/config.pq"
            )
            try:
                df_config = pd.read_parquet(config_path)
            except FileNotFoundError:
                print(f"{project_folder} V{variant} G{generation}: missing config.pq")
                continue

            bulk_ids = _listify_metadata(df_config["output_metadata__bulk"].iloc[0])
            mRNA_ids = _listify_metadata(
                df_config["output_metadata__listeners__rna_counts__mRNA_counts"].iloc[0]
            )

            if not isinstance(bulk_ids, list):
                bulk_ids = list(bulk_ids)
            if not isinstance(mRNA_ids, list):
                mRNA_ids = list(mRNA_ids)

            rna_init_cols = [col for col in df.columns if "rna_init_event" in col]
            ribo_init_cols = [col for col in df.columns if "ribosome_init_event" in col]

            gene_block = gene_info_df.copy()
            gene_block["project"] = project_folder
            gene_block["variant"] = variant
            gene_block["generation"] = generation
            gene_block["lineage_seed"] = lineage_seed
            gene_block["label"] = gene_block["gene_id"]
            gene_block["duration_sec"] = float(df["time"].iloc[-1] - df["time"].iloc[0])
            gene_block["n_timepoints"] = int(len(df))

            for _, gene_row in gene_block.iterrows():
                rna_id = gene_row["rna_id"]
                monomer_id = gene_row["monomer_id"]
                rna_idx_mrna = mRNA_ids.index(rna_id) if rna_id in mRNA_ids else None
                monomer_idx_bulk = (
                    bulk_ids.index(monomer_id)
                    if monomer_id and monomer_id in bulk_ids
                    else None
                )

                if rna_idx_mrna is not None:
                    mRNA_values = (
                        df["listeners__rna_counts__mRNA_counts"]
                        .apply(
                            lambda value: value[rna_idx_mrna]
                            if len(value) > rna_idx_mrna
                            else 0
                        )
                        .astype(np.int32)
                        .to_numpy()
                    )
                else:
                    mRNA_values = np.zeros(len(df), dtype=np.int32)

                if monomer_idx_bulk is not None:
                    protein_values = (
                        df["bulk"]
                        .apply(
                            lambda value: value[monomer_idx_bulk]
                            if len(value) > monomer_idx_bulk
                            else 0
                        )
                        .astype(np.int32)
                        .to_numpy()
                    )
                else:
                    protein_values = np.zeros(len(df), dtype=np.int32)

                if rna_init_cols:
                    tc_values = (
                        df[rna_init_cols[0]]
                        .apply(
                            lambda value: value[gene_row["rna_idx_rna"]]
                            if len(value) > gene_row["rna_idx_rna"]
                            else 0
                        )
                        .astype(np.int32)
                        .to_numpy()
                    )
                else:
                    tc_values = np.zeros(len(df), dtype=np.int32)

                if ribo_init_cols and monomer_idx_bulk is not None:
                    tl_values = (
                        df[ribo_init_cols[0]]
                        .apply(
                            lambda value: value[monomer_idx_bulk]
                            if len(value) > monomer_idx_bulk
                            else 0
                        )
                        .astype(np.int32)
                        .to_numpy()
                    )
                else:
                    tl_values = np.zeros(len(df), dtype=np.int32)

                transcription_success = int(max(mRNA_values[-1] - mRNA_values[0], 0))
                translation_success = int(
                    max(protein_values[-1] - protein_values[0], 0)
                )

                summary_row = {
                    **gene_row.to_dict(),
                    "mRNA_initial": int(mRNA_values[0]),
                    "mRNA_final": int(mRNA_values[-1]),
                    "mRNA_min": int(np.min(mRNA_values)),
                    "mRNA_max": int(np.max(mRNA_values)),
                    "mRNA_mean": float(np.mean(mRNA_values)),
                    "mRNA_change": int(mRNA_values[-1] - mRNA_values[0]),
                    "mRNA_change_pct": float(
                        100
                        * ((mRNA_values[-1] - mRNA_values[0]) / max(mRNA_values[0], 1))
                    ),
                    "protein_initial": int(protein_values[0]),
                    "protein_final": int(protein_values[-1]),
                    "protein_min": int(np.min(protein_values)),
                    "protein_max": int(np.max(protein_values)),
                    "protein_mean": float(np.mean(protein_values)),
                    "protein_change": int(protein_values[-1] - protein_values[0]),
                    "protein_change_pct": float(
                        100
                        * (
                            (protein_values[-1] - protein_values[0])
                            / max(protein_values[0], 1)
                        )
                    ),
                    "transcription_success_count": transcription_success,
                    "translation_success_count": translation_success,
                    "total_tc_initiations": int(tc_values.sum()),
                    "total_tl_initiations": int(tl_values.sum()),
                }

                summary_rows.append(summary_row)

                if save_timeseries:
                    ts_df = pd.DataFrame(
                        {
                            "project": project_folder,
                            "variant": variant,
                            "generation": generation,
                            "lineage_seed": lineage_seed,
                            "gene_id": gene_row["gene_id"],
                            "label": gene_row["label"],
                            "cistron_id": gene_row["cistron_id"],
                            "rna_id": gene_row["rna_id"],
                            "monomer_id": gene_row["monomer_id"],
                            "time": df["time"].to_numpy(),
                            "mRNA": mRNA_values,
                            "protein": protein_values,
                            "tc_init_events": tc_values,
                            "tl_init_events": tl_values,
                        }
                    )
                    timeseries_frames.append(ts_df)

            print(
                f"✓ {project_folder} V{variant} G{generation}: processed {len(gene_block)} genes, {len(df)} timepoints"
            )

    df_summary = pd.DataFrame(summary_rows)
    df_timeseries = (
        pd.concat(timeseries_frames, ignore_index=True)
        if timeseries_frames
        else pd.DataFrame()
    )
    return df_summary, df_timeseries


def main():
    parser = argparse.ArgumentParser(
        description="Extract gene-expression data from raw vEcoli simulations.",
        formatter_class=DefaultHelpFormatter,
        epilog="""
Examples:
  python reading/gene_expression_extract.py
  python reading/gene_expression_extract.py --trial
First Implementation:
  python reading/gene_expression_extract.py --all --lineage-seeds 100 101 --save-timeseries --suffix first_im
        """,
    )
    parser.add_argument(
        "--save-timeseries",
        action="store_true",
        help="Save the full downsampled timeseries parquet in addition to the summary CSV.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all variants and every generation that exists on disk for each variant.",
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Trial run: 2 variants and 2 generations.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        nargs="+",
        default=None,
        help="Specific generations to extract. Use --all to discover generations on disk.",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Maximum number of variants to extract per project.",
    )
    parser.add_argument(
        "--projects",
        type=str,
        nargs="+",
        default=DEFAULT_PROJECTS,
        help="Full project folder names to extract from.",
    )
    parser.add_argument(
        "--gene-list",
        type=str,
        default=str(DEFAULT_GENE_LIST),
        help="Path to a newline-delimited list of gene IDs to extract.",
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=None,
        help="Limit the number of genes loaded from the gene list for quick tests.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="default",
        help="Suffix for output filenames only.",
    )
    parser.add_argument(
        "--lineage-seeds",
        type=int,
        nargs="+",
        default=[0],
        help="One or more lineage seeds to extract.",
    )
    parser.add_argument(
        "--read-interval-sec",
        type=int,
        default=1,
        help="Downsample by time, taking one point every N seconds.",
    )

    args = parser.parse_args()

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
        generations = None
        max_variants = args.max_variants
        run_type = "CUSTOM"

    gene_ids = load_gene_list(args.gene_list, max_genes=args.max_genes)

    print("=" * 80)
    print(f"GENE EXPRESSION EXTRACTION - {run_type}")
    print(f"Projects: {args.projects}")
    print(f"Gene list: {args.gene_list}")
    print(f"Genes loaded: {len(gene_ids)}")
    print(f"Generations: {'all available' if generations is None else generations}")
    print(f"Max variants per project: {max_variants if max_variants else 'all'}")
    print(f"Lineage seeds: {args.lineage_seeds}")
    print(f"Read interval sec: {args.read_interval_sec}")
    print(f"Output suffix: {args.suffix}")
    print("=" * 80)
    print()

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    project_frames = []
    timeseries_frames = []
    for project_folder in args.projects:
        for lineage_seed in args.lineage_seeds:
            df_summary, df_timeseries = extract_gene_expression_data(
                project_folder=project_folder,
                gene_ids=gene_ids,
                generation=generations,
                read_interval_sec=args.read_interval_sec,
                max_variants=max_variants,
                save_timeseries=args.save_timeseries,
                lineage_seed=lineage_seed,
            )
            if len(df_summary) > 0:
                project_frames.append(df_summary)
            if args.save_timeseries and len(df_timeseries) > 0:
                timeseries_frames.append(df_timeseries)

    if not project_frames:
        print(
            "\nERROR: No data was extracted. Check project names, gene list, and data paths."
        )
        return 1

    df_all = pd.concat(project_frames, ignore_index=True, sort=False)

    if args.save_timeseries:
        if not timeseries_frames:
            print("\nERROR: No timeseries rows were produced.")
            return 1

        df_timeseries = pd.concat(timeseries_frames, ignore_index=True, sort=False)

        timeseries_file = (
            DEFAULT_OUTPUT_DIR
            / f"gene_expression_timeseries_{args.suffix}_{run_type.lower()}.parquet"
        )
        df_timeseries.to_parquet(timeseries_file, index=False)
        print(f"✓ Saved timeseries: {timeseries_file} ({len(df_timeseries)} rows)")
        df_summary = df_all.copy()

    print()
    print("=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total rows: {len(df_summary)}")
    print()
    print("Rows per project:")
    print(df_summary.groupby("project")["gene_id"].count())
    print()

    summary_file = (
        DEFAULT_OUTPUT_DIR
        / f"gene_expression_summary_{args.suffix}_{run_type.lower()}.csv"
    )
    df_summary.to_csv(summary_file, index=False)
    print(f"✓ Saved summary: {summary_file}")

    print()
    print("=" * 80)
    print("STATISTICS BY PROJECT")
    print("=" * 80)
    project_stats = (
        df_summary.groupby("project")
        .agg(
            {
                "gene_id": "count",
                "transcription_success_count": ["mean", "std", "min", "max"],
                "translation_success_count": ["mean", "std", "min", "max"],
                "mRNA_max": ["mean", "max"],
                "protein_max": ["mean", "max"],
            }
        )
        .round(6)
    )
    print(project_stats)
    print()
    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
