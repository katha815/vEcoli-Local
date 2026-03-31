#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GENE_LIST = Path(
    "/user/home/il22158/work/vEcoli/reading/imported/Single_KO_RNA_names.txt"
)
DEFAULT_TIMESERIES_FILES = [
    Path(
        "/user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_timeseries_seed100_all.parquet"
    ),
    Path(
        "/user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_timeseries_seed101_all.parquet"
    ),
]
DEFAULT_OUT_ROOT = Path("/user/home/il22158/work/vEcoli/out")
DEFAULT_OUTPUT_DIR = Path("/user/home/il22158/work/vEcoli/surrogate/results")


def parse_args() -> argparse.Namespace:
    help_text = (
        "Instructions:\n"
        "1) KO variants are selected from labels starting with KO: and filtered by the gene list.\n"
        "2) Baseline runs are selected as label=baseline under projects starting with all_media_conditions and variant=0.\n"
        "3) Generation boundaries are rebuilt from raw file names under out/.../generation=*/agent_id=*/<step>.pq.\n"
        "4) step_scale controls conversion of filename step index to time.\n"
        "   - Use step_scale=1 for early files: growth_rate_timeseries_seed100_all.parquet and growth_rate_timeseries_seed101_all.parquet\n"
        "   - Use step_scale=20 for files that require *20 conversion (for example growth_rate_timeseries_441_KOs_seed100_all.parquet and growth_rate_timeseries_441_KOs_seed101_all.parquet).\n"
        "\n"
        "Examples:\n"
        "- Early files (no *20):\n"
        "  python surrogate/preprocess_timeseries_surrogate.py --timeseries-files <seed100_file> <seed101_file> --step-scale 1 --output-prefix surrogate_preprocessed_early\n"
        "- 441 KO files (*20 style):\n"
        "  python surrogate/preprocess_timeseries_surrogate.py --timeseries-files <ko_seed100_file> <ko_seed101_file> --step-scale 20 --output-prefix surrogate_preprocessed_ko\n"
        "- Mixed files in one run:\n"
        "  python surrogate/preprocess_timeseries_surrogate.py --timeseries-files <all_4_files> --step-scale 20 --output-prefix surrogate_preprocessed_all\n"
        "\n"
        "Batch outputs (written to --output-dir):\n"
        "- 1. <output-prefix>_generation_long.csv:\n"
        "  One row per reconstructed generation per run with start/end time, points, growth_mean, dry-mass fold/log-fold metrics, and run identifiers.\n"
        "- 2. <output-prefix>_run_summary.csv:\n"
        "  One row per run. mean_growth_over_generations is the arithmetic mean of per-generation growth_mean values.\n"
        "- 3. <output-prefix>_skipped_runs.csv:\n"
        "  Runs excluded from long/summary with a reason (for example empty_run, missing_raw_hints, no_assigned_rows).\n"
        "Single-run mode (prototype for surrogate input):\n"
        "  python surrogate/preprocess_timeseries_surrogate.py --mode single-run --project <project> --variant <variant> --lineage-seed <lineage_seed> --step-scale 20 --output-prefix surrogate_preprocessed_single_run\n"
    )

    p = argparse.ArgumentParser(
        description="Minimal preprocessing for surrogate inputs.",
        epilog=help_text,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--gene-list", type=Path, default=DEFAULT_GENE_LIST)
    p.add_argument(
        "--timeseries-files",
        type=Path,
        nargs="+",
        default=DEFAULT_TIMESERIES_FILES,
        help="Input saved timeseries parquet file(s).",
    )
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--output-prefix", type=str, default="surrogate_preprocessed")
    p.add_argument(
        "--mode",
        choices=["batch", "single-run"],
        default="batch",
        help="Run full dataset preprocessing (batch) or reproduce prototype for one run (single-run).",
    )
    p.add_argument(
        "--project", type=str, default=None, help="Required in --mode single-run."
    )
    p.add_argument(
        "--variant", type=int, default=None, help="Required in --mode single-run."
    )
    p.add_argument(
        "--lineage-seed", type=int, default=None, help="Required in --mode single-run."
    )
    p.add_argument(
        "--step-scale",
        type=float,
        default=1.0,
        help="Scalar for filename step index -> time (common values: 1 for early files, 20 for *20 conversion).",
    )
    p.add_argument("--tolerance", type=float, default=50.0)
    p.add_argument("--max-runs", type=int, default=None)
    return p.parse_args()


def load_gene_set(path: Path) -> set[str]:
    g = pd.read_csv(path, header=None, names=["gene_id"])
    g["gene_id"] = g["gene_id"].astype(str).str.strip()
    g = g[g["gene_id"] != ""].drop_duplicates()
    return set(g["gene_id"])


def infer_seed(project: pd.Series, source_file: pd.Series) -> pd.Series:
    s = project.astype(str).str.extract(r"seed(?:=|_)?(\d+)")[0]
    s = s.fillna(project.astype(str).str.extract(r"lineage_seed(?:=|_)?(\d+)")[0])
    s = s.fillna(source_file.astype(str).str.extract(r"seed(?:=|_)?(\d+)")[0])
    default_mask = project.astype(str).str.contains(
        "default", case=False, na=False
    ) | source_file.astype(str).str.contains("default", case=False, na=False)
    s.loc[s.isna() & default_mask] = "0"
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def raw_generation_hints(
    out_root: Path, project: str, variant: int, seed: int
) -> pd.DataFrame | None:
    base = (
        out_root
        / project
        / "history"
        / f"experiment_id={project}"
        / f"variant={int(variant)}"
        / f"lineage_seed={int(seed)}"
    )
    files = list(base.glob("generation=*/agent_id=*/**/*.pq"))
    if not files:
        return None

    rows = []
    for p in files:
        gen_token = [x for x in p.parts if x.startswith("generation=")]
        if not gen_token:
            continue
        generation = int(gen_token[0].split("=")[1])
        step_idx = pd.to_numeric(p.stem, errors="coerce")
        if pd.isna(step_idx):
            continue
        rows.append({"generation": generation, "step_idx": float(step_idx)})

    if not rows:
        return None

    return (
        pd.DataFrame(rows)
        .groupby("generation", as_index=False)["step_idx"]
        .max()
        .sort_values("generation")
        .reset_index(drop=True)
    )


def build_raw_base(out_root: Path, project: str, variant: int, seed: int) -> Path:
    return (
        out_root
        / project
        / "history"
        / f"experiment_id={project}"
        / f"variant={int(variant)}"
        / f"lineage_seed={int(seed)}"
    )


def estimate_dt_from_raw(raw_base: Path, fallback_dt: float = 20.0) -> float:
    first_raw = next(raw_base.glob("generation=*/agent_id=*/**/*.pq"), None)
    if first_raw is None:
        return fallback_dt

    first_df = pd.read_parquet(first_raw)
    if "time" not in first_df.columns or len(first_df) < 2:
        return fallback_dt

    t = pd.to_numeric(first_df["time"], errors="coerce").dropna().to_numpy()
    if len(t) < 2:
        return fallback_dt

    dt = float(np.nanmedian(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return fallback_dt
    return dt


def prototype_raw_meta(
    out_root: Path, project: str, variant: int, seed: int, fallback_dt: float = 20.0
) -> tuple[pd.DataFrame, float]:
    hints = raw_generation_hints(out_root, project, variant, seed)
    if hints is None or hints.empty:
        raw_base = build_raw_base(out_root, project, variant, seed)
        raise RuntimeError(f"No raw generation pq files found under: {raw_base}")

    raw_base = build_raw_base(out_root, project, variant, seed)
    dt = estimate_dt_from_raw(raw_base, fallback_dt=fallback_dt)

    raw_meta = hints.rename(columns={"step_idx": "step_idx_hint"}).copy()
    # Match notebook prototype: convert step-index endpoint to within-generation end-time.
    raw_meta["gen_end_time_hint"] = (raw_meta["step_idx_hint"] - 1.0) * dt
    raw_meta["cum_end_time_hint"] = raw_meta["gen_end_time_hint"].cumsum()
    raw_meta["cum_start_time_hint"] = raw_meta["cum_end_time_hint"].shift(
        fill_value=0.0
    )
    return raw_meta, dt


def select_single_run(
    ts: pd.DataFrame, project: str, variant: int, lineage_seed: int
) -> pd.DataFrame:
    if "project" not in ts.columns:
        raise KeyError("Column 'project' not found in timeseries parquet files.")
    if "variant" not in ts.columns:
        raise KeyError("Column 'variant' not found in timeseries parquet files.")

    seed_token = f"seed{int(lineage_seed)}"
    saved_run = ts[
        (ts["project"].astype(str) == str(project))
        & (pd.to_numeric(ts["variant"], errors="coerce") == int(variant))
        & (ts["source_file"].astype(str).str.contains(seed_token, case=False, na=False))
    ].copy()
    return saved_run


def build_gen_table(saved_run: pd.DataFrame) -> pd.DataFrame:
    gen_table = (
        saved_run.dropna(subset=["generation_reconstructed"])
        .groupby("generation_reconstructed", as_index=False)
        .agg(
            start_time=("time", "min"),
            end_time=("time", "max"),
            n_points=("time", "size"),
            growth_mean=("growth_rate", "mean"),
            dry_mass_start=("dry_mass", "first"),
            dry_mass_end=("dry_mass", "last"),
        )
        .sort_values("generation_reconstructed")
        .reset_index(drop=True)
    )
    gen_table["fold_change_mass"] = gen_table["dry_mass_end"] / gen_table[
        "dry_mass_start"
    ].replace(0, np.nan)
    gen_table["log_fold_change_mass"] = np.log(gen_table["fold_change_mass"])
    return gen_table


def run_single_prototype(args: argparse.Namespace) -> None:
    if args.project is None or args.variant is None or args.lineage_seed is None:
        raise ValueError(
            "--project, --variant, and --lineage-seed are required when --mode single-run is used."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for p in args.timeseries_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        x = pd.read_parquet(p).copy()
        x["source_file"] = p.name
        parts.append(x)
    saved = pd.concat(parts, ignore_index=True)

    raw_meta, dt = prototype_raw_meta(
        args.out_root, args.project, args.variant, args.lineage_seed
    )

    print("Raw generation endpoint hints for selected run:")
    print(raw_meta.to_string(index=False))
    print(f"Estimated dt from raw pq = {dt:.3f}")

    saved_run = select_single_run(saved, args.project, args.variant, args.lineage_seed)
    if saved_run.empty:
        raise RuntimeError(
            "No matching rows found in saved pq for selected project/variant/seed token."
        )

    required_columns = [
        "time",
        "listeners__mass__instantaneous_growth_rate",
        "listeners__mass__dry_mass",
    ]
    missing_required = [c for c in required_columns if c not in saved_run.columns]
    if missing_required:
        raise KeyError(
            f"Required column(s) missing in selected run: {missing_required}"
        )

    saved_run["time"] = pd.to_numeric(saved_run["time"], errors="coerce")
    saved_run["growth_rate"] = pd.to_numeric(
        saved_run["listeners__mass__instantaneous_growth_rate"], errors="coerce"
    )
    saved_run["dry_mass"] = pd.to_numeric(
        saved_run["listeners__mass__dry_mass"], errors="coerce"
    )
    saved_run = (
        saved_run.dropna(subset=["time", "growth_rate", "dry_mass"])
        .sort_values("time")
        .reset_index(drop=True)
    )

    print(f"Saved rows selected: {len(saved_run)}")
    print(
        f"Saved time range: {saved_run['time'].min():.3f} to {saved_run['time'].max():.3f}"
    )

    bin_edges = np.r_[0.0, raw_meta["cum_end_time_hint"].to_numpy()]
    gen_labels = np.arange(1, len(bin_edges))
    tol = max(2.5 * dt, args.tolerance)
    bin_edges_tol = bin_edges.copy()
    bin_edges_tol[-1] = bin_edges_tol[-1] + tol

    saved_run["generation_reconstructed"] = pd.cut(
        saved_run["time"],
        bins=bin_edges_tol,
        labels=gen_labels,
        include_lowest=True,
        right=True,
    ).astype("Int64")

    gen_table = build_gen_table(saved_run)

    print("Reconstructed per-generation table (prototype for surrogate input):")
    print(gen_table.to_string(index=False))

    wide = gen_table[
        [
            "generation_reconstructed",
            "growth_mean",
            "fold_change_mass",
            "log_fold_change_mass",
        ]
    ].copy()
    wide = wide.set_index("generation_reconstructed")
    wide_out = wide.T
    print("Wide preview (single-run feature/target style):")
    print(wide_out.to_string())

    out_stem = f"{args.output_prefix}_single_run_project-{args.project}_variant-{int(args.variant)}_seed-{int(args.lineage_seed)}"
    raw_meta_csv = args.output_dir / f"{out_stem}_raw_meta.csv"
    gen_table_csv = args.output_dir / f"{out_stem}_gen_table.csv"
    wide_csv = args.output_dir / f"{out_stem}_wide_preview.csv"

    raw_meta.to_csv(raw_meta_csv, index=False)
    gen_table.to_csv(gen_table_csv, index=False)
    wide_out.to_csv(wide_csv)

    print("Done")
    print(f"Saved: {raw_meta_csv}")
    print(f"Saved: {gen_table_csv}")
    print(f"Saved: {wide_csv}")


def main() -> None:
    args = parse_args()

    if args.mode == "single-run":
        run_single_prototype(args)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gene_set = load_gene_set(args.gene_list)
    print(f"Loaded KO genes: {len(gene_set)}")

    # Load and concatenate saved timeseries files.
    parts = []
    for p in args.timeseries_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        x = pd.read_parquet(p)
        x["source_file"] = p.name
        parts.append(x)
    ts = pd.concat(parts, ignore_index=True)

    # Standard columns expected from your notebook data.
    project_col = "project" if "project" in ts.columns else "experiment_id"
    variant_col = "variant"
    label_col = "label"
    time_col = "time"
    growth_col = (
        "listeners__mass__instantaneous_growth_rate"
        if "listeners__mass__instantaneous_growth_rate" in ts.columns
        else "growth_rate"
    )
    dry_mass_col = (
        "listeners__mass__dry_mass"
        if "listeners__mass__dry_mass" in ts.columns
        else "dry_mass"
    )

    if "lineage_seed" in ts.columns:
        seed_col = "lineage_seed"
        ts[seed_col] = pd.to_numeric(ts[seed_col], errors="coerce").astype("Int64")
        missing_seed = ts[seed_col].isna()
        if missing_seed.any():
            ts.loc[missing_seed, seed_col] = infer_seed(
                ts.loc[missing_seed, project_col], ts.loc[missing_seed, "source_file"]
            )
    elif "seed" in ts.columns:
        seed_col = "seed"
        ts[seed_col] = pd.to_numeric(ts[seed_col], errors="coerce").astype("Int64")
    else:
        seed_col = "lineage_seed_inferred"
        ts[seed_col] = infer_seed(ts[project_col], ts["source_file"])

    ts[variant_col] = pd.to_numeric(ts[variant_col], errors="coerce").astype("Int64")
    ts[time_col] = pd.to_numeric(ts[time_col], errors="coerce")
    ts[growth_col] = pd.to_numeric(ts[growth_col], errors="coerce")
    ts[dry_mass_col] = pd.to_numeric(ts[dry_mass_col], errors="coerce")

    ts["gene_id"] = ts[label_col].astype(str).str.extract(r"KO:\s*([^,\s]+)")[0]
    is_ko = ts[label_col].astype(str).str.startswith("KO:") & ts["gene_id"].isin(
        gene_set
    )
    is_baseline = (
        ts[label_col].astype(str).str.strip().str.lower().eq("baseline")
        & ts[project_col].astype(str).str.startswith("all_media_conditions")
        & ts[variant_col].eq(0)
    )

    keep = is_ko | is_baseline
    df = ts.loc[keep].copy()
    df["run_type"] = np.where(is_ko.loc[df.index], "KO", "baseline")
    df = df.dropna(
        subset=[
            project_col,
            variant_col,
            seed_col,
            label_col,
            time_col,
            growth_col,
            dry_mass_col,
        ]
    ).copy()

    run_keys = (
        df[[project_col, variant_col, seed_col, label_col, "gene_id", "run_type"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if args.max_runs is not None:
        run_keys = run_keys.head(args.max_runs)
    print(f"Candidate runs: {len(run_keys)}")

    long_rows = []
    skipped_rows = []

    for _, rk in run_keys.iterrows():
        project = str(rk[project_col])
        variant = int(rk[variant_col])
        seed = int(rk[seed_col])
        label = str(rk[label_col])

        run_ts = df[
            (df[project_col].astype(str) == project)
            & (df[variant_col].astype(int) == variant)
            & (df[seed_col].astype(int) == seed)
            & (df[label_col].astype(str) == label)
        ][[time_col, growth_col, dry_mass_col]].copy()
        run_ts = run_ts.sort_values(time_col).reset_index(drop=True)
        if run_ts.empty:
            skipped_rows.append(
                {
                    "project": project,
                    "variant": variant,
                    "lineage_seed": seed,
                    "label": label,
                    "reason": "empty_run",
                }
            )
            continue

        hints = raw_generation_hints(args.out_root, project, variant, seed)
        if hints is None:
            skipped_rows.append(
                {
                    "project": project,
                    "variant": variant,
                    "lineage_seed": seed,
                    "label": label,
                    "reason": "missing_raw_hints",
                }
            )
            continue

        # Minimal conversion: cumulative endpoint = cumsum(step_idx * step_scale).
        cumulative_end = (hints["step_idx"] * args.step_scale).cumsum().to_numpy()
        edges = np.r_[0.0, cumulative_end]
        edges[-1] = edges[-1] + args.tolerance

        run_ts["generation_reconstructed"] = pd.cut(
            run_ts[time_col],
            bins=edges,
            labels=np.arange(1, len(edges)),
            include_lowest=True,
            right=True,
        ).astype("Int64")

        assigned = run_ts.dropna(subset=["generation_reconstructed"]).copy()
        if assigned.empty:
            skipped_rows.append(
                {
                    "project": project,
                    "variant": variant,
                    "lineage_seed": seed,
                    "label": label,
                    "reason": "no_assigned_rows",
                }
            )
            continue

        gen_table = (
            assigned.groupby("generation_reconstructed", as_index=False)
            .agg(
                start_time=(time_col, "min"),
                end_time=(time_col, "max"),
                n_points=(time_col, "size"),
                growth_mean=(growth_col, "mean"),
                dry_mass_start=(dry_mass_col, "first"),
                dry_mass_end=(dry_mass_col, "last"),
            )
            .sort_values("generation_reconstructed")
            .reset_index(drop=True)
        )
        gen_table["fold_change_mass"] = gen_table["dry_mass_end"] / gen_table[
            "dry_mass_start"
        ].replace(0, np.nan)
        gen_table["log_fold_change_mass"] = np.log(gen_table["fold_change_mass"])

        gen_table["project"] = project
        gen_table["variant"] = variant
        gen_table["lineage_seed"] = seed
        gen_table["label"] = label
        gen_table["gene_id"] = rk["gene_id"] if pd.notna(rk["gene_id"]) else None
        gen_table["run_type"] = str(rk["run_type"])
        gen_table["step_scale"] = args.step_scale

        long_rows.append(gen_table)

    if not long_rows:
        raise RuntimeError("No runs were preprocessed. Check inputs and step_scale.")

    long_df = pd.concat(long_rows, ignore_index=True)
    skipped_df = pd.DataFrame(skipped_rows)

    run_summary = (
        long_df.groupby(
            ["project", "variant", "lineage_seed", "label", "gene_id", "run_type"],
            as_index=False,
            dropna=False,
        )
        .agg(
            n_generations=("generation_reconstructed", "max"),
            mean_growth_over_generations=("growth_mean", "mean"),
            total_fold_change_mass=("fold_change_mass", "prod"),
            mean_log_fold_change_mass=("log_fold_change_mass", "mean"),
            step_scale=("step_scale", "first"),
        )
        .reset_index(drop=True)
    )

    prefix = args.output_prefix
    long_csv = args.output_dir / f"{prefix}_generation_long.csv"
    summary_csv = args.output_dir / f"{prefix}_run_summary.csv"
    skipped_csv = args.output_dir / f"{prefix}_skipped_runs.csv"

    long_df.to_csv(long_csv, index=False)
    run_summary.to_csv(summary_csv, index=False)
    skipped_df.to_csv(skipped_csv, index=False)

    print("Done")
    print(f"Saved: {long_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {skipped_csv}")
    print(f"Processed runs: {len(run_summary)}")
    print(f"Skipped runs: {len(skipped_df)}")


if __name__ == "__main__":
    main()
