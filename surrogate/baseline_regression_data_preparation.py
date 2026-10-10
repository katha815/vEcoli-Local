from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import re
import argparse

DEFAULT_BASE_DIR = Path("/user/home/il22158/work/vEcoli")
DEFAULT_TRAIN_DIR = DEFAULT_BASE_DIR / "surrogate" / "train_data"

BASE = DEFAULT_BASE_DIR
TRAIN_DIR = DEFAULT_TRAIN_DIR

MAX_GENERATION_FILE = (
    BASE / "surrogate/results/default_KO/gene_knockout_generation_summary.csv"
)
LOG_FILE = TRAIN_DIR / "gene_TU_combined_training_dataset.log"
SUMMARY_FILES = [
    BASE
    / "reading/results/growth_rate/growth_rate_summary_gene_knockout_102trails_seed100_all.csv",
    BASE
    / "reading/results/growth_rate/growth_rate_summary_gene_knockout_102trails_seed101_all.csv",
    BASE
    / "reading/results/growth_rate/growth_rate_summary_gene_knockout_leftover_seed101_all.csv",
    BASE
    / "reading/results/growth_rate/growth_rate_summary_gene_knockout_leftover_seed100_all.csv",
    BASE
    / "reading/results/growth_rate/growth_rate_summary_gene_knockout_TU_ID_rest_list_rerun12_seed10_all.csv",
    BASE
    / "reading/results/growth_rate/growth_rate_summary_gene_knockout_TU_ID_rest_list_rerun12_seed11_all.csv",
]
SUMMARY_FILE_SUFFIXES = [
    "reading/results/growth_rate/growth_rate_summary_gene_knockout_102trails_seed100_all.csv",
    "reading/results/growth_rate/growth_rate_summary_gene_knockout_102trails_seed101_all.csv",
    "reading/results/growth_rate/growth_rate_summary_gene_knockout_leftover_seed101_all.csv",
    "reading/results/growth_rate/growth_rate_summary_gene_knockout_leftover_seed100_all.csv",
    "reading/results/growth_rate/growth_rate_summary_gene_knockout_TU_ID_rest_list_rerun12_seed10_all.csv",
    "reading/results/growth_rate/growth_rate_summary_gene_knockout_TU_ID_rest_list_rerun12_seed11_all.csv",
]
GENE_REF = BASE / "reconstruction/ecoli/flat/rnas.tsv"
TU_REF = BASE / "reconstruction/ecoli/flat/transcription_units.tsv"


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    pass


def parse_args() -> argparse.Namespace:
    default_summary_files_text = "\n".join(
        f"  {DEFAULT_BASE_DIR / relative_path}"
        for relative_path in SUMMARY_FILE_SUFFIXES
    )
    parser = argparse.ArgumentParser(
        description="Rebuild the combined gene/TU surrogate training dataset and feature manifest.",
        formatter_class=HelpFormatter,
        epilog=(
            "Example use:\n"
            "  python surrogate/baseline_regression_gene_TU_combined_rebuild.py\n"
            "  python surrogate/baseline_regression_gene_TU_combined_rebuild.py --train-dir /tmp/vEcoli-train-data\n"
            "  python surrogate/baseline_regression_gene_TU_combined_rebuild.py --generation-window-start 4 --generation-window-end 12 --max-generation-criterion 8\n"
            "\nOutput files (written to --train-dir, default: surrogate/train_data):\n"
            "  gene_TU_combined_training_dataset.parquet  (Parquet; example output dataset)\n"
            "  gene_TU_combined_training_dataset.csv      (CSV; same dataset in text form)\n"
            "  gene_TU_combined_feature_list.csv          (CSV; feature manifest)\n"
            "  raw_summary_combined.csv                    (CSV; filtered summary rows)\n"
            "\nDefault summary files (override by passing your own paths after --summary-file):\n"
            f"{default_summary_files_text}\n"
            "\nTo include your own files, pass multiple paths after one --summary-file flag:\n"
            "  python surrogate/baseline_regression_gene_TU_combined_rebuild.py --summary-file /user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_summary_gene_knockout_462KO_basal_operon_on_p1_all.csv /user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_summary_gene_knockout_462KO_basal_operon_on_p2_all.csv /user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_summary_gene_knockout_462KO_basal_operon_on_p3_all.csv /user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_summary_gene_knockout_462KO_basal_operon_on_p4_all.csv /user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_summary_gene_knockout_462KO_basal_operon_on_p5_all.csv\n"
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Root of the vEcoli checkout used to resolve input files.",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Directory where the rebuilt dataset, feature list, and log file are written.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        nargs="+",
        dest="summary_files",
        help=(
            "Summary CSV to include in the rebuild. Pass one or more files after a single option; "
            "if omitted, the built-in set is used."
        ),
    )
    parser.add_argument(
        "--generation-window-start",
        type=float,
        default=4,
        help="Lower bound for max_generation when selecting non-baseline rows; omit to keep all lower values.",
    )
    parser.add_argument(
        "--generation-window-end",
        type=float,
        default=12,
        help="Upper bound for max_generation when selecting non-baseline rows; omit to keep all higher values.",
    )
    parser.add_argument(
        "--max-generation-criterion",
        "--failed-generation-threshold",
        dest="max_generation_criterion",
        type=float,
        default=8,
        help="Rows with max_generation below this cutoff, or missing, are marked as failed_generation.",
    )
    return parser.parse_args()


def configure_paths(base_dir: Path, train_dir: Path) -> None:
    global \
        BASE, \
        TRAIN_DIR, \
        MAX_GENERATION_FILE, \
        LOG_FILE, \
        SUMMARY_FILES, \
        GENE_REF, \
        TU_REF

    BASE = base_dir
    TRAIN_DIR = train_dir
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    MAX_GENERATION_FILE = (
        BASE / "surrogate/results/default_KO/gene_knockout_generation_summary.csv"
    )
    LOG_FILE = TRAIN_DIR / "gene_TU_combined_training_dataset.log"
    SUMMARY_FILES = [
        BASE
        / "reading/results/growth_rate/growth_rate_summary_gene_knockout_102trails_seed100_all.csv",
        BASE
        / "reading/results/growth_rate/growth_rate_summary_gene_knockout_102trails_seed101_all.csv",
        BASE
        / "reading/results/growth_rate/growth_rate_summary_gene_knockout_leftover_seed101_all.csv",
        BASE
        / "reading/results/growth_rate/growth_rate_summary_gene_knockout_leftover_seed100_all.csv",
        BASE
        / "reading/results/growth_rate/growth_rate_summary_gene_knockout_TU_ID_rest_list_rerun12_seed10_all.csv",
        BASE
        / "reading/results/growth_rate/growth_rate_summary_gene_knockout_TU_ID_rest_list_rerun12_seed11_all.csv",
    ]

    GENE_REF = BASE / "reconstruction/ecoli/flat/rnas.tsv"
    TU_REF = BASE / "reconstruction/ecoli/flat/transcription_units.tsv"


def setup_logging() -> object:
    log_handle = LOG_FILE.open("w", encoding="utf-8")
    print(f"Logging to: {LOG_FILE}")
    return log_handle


def main() -> None:
    args = parse_args()
    if (
        args.generation_window_start is not None
        and args.generation_window_end is not None
        and args.generation_window_start > args.generation_window_end
    ):
        raise ValueError(
            "generation-window-start must be less than or equal to generation-window-end"
        )
    configure_paths(args.base_dir, args.train_dir)
    summary_files = args.summary_files or SUMMARY_FILES
    log_handle = setup_logging()
    try:
        with redirect_stdout(log_handle), redirect_stderr(log_handle):
            build_dataset(
                summary_files=summary_files,
                max_generation_criterion=args.max_generation_criterion,
                generation_window_start=args.generation_window_start,
                generation_window_end=args.generation_window_end,
            )
    finally:
        log_handle.close()


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#").copy()
    df["source_file"] = path.name
    df["label_raw"] = df["label"]
    df["label"] = df["label"].astype("string").str.replace(r"^KO:\s*", "", regex=True)
    return df


def infer_source_group(source_file: str) -> str:
    key = source_file.lower()
    if "tu_id_rest_list_rerun12" in key:
        return "TU_KO"
    if "gene_knockout" in key or "102trails" in key or "leftover" in key:
        return "gene_KO"
    raise ValueError(f"Cannot infer source group from {source_file}")


def extract_seed(source_file: str):
    match = re.search(r"seed(\d+)", source_file)
    return int(match.group(1)) if match else pd.NA


def parse_ko_ids(label: object) -> list[str]:
    if pd.isna(label):
        return []
    text = str(label).strip()
    if not text or text == "baseline":
        return []
    if text.startswith("KO:"):
        text = text.split(":", 1)[1].strip()
    return [part.strip() for part in re.split(r"[,+;|]", text) if part.strip()]


def build_dataset(
    summary_files: list[Path],
    max_generation_criterion: float,
    generation_window_start: float | None,
    generation_window_end: float | None,
) -> None:
    global np, pd

    import numpy as np
    import pandas as pd

    summary_df = pd.concat(
        [load_summary(path) for path in summary_files], ignore_index=True
    )
    max_generation_df = pd.read_csv(MAX_GENERATION_FILE)
    max_generation_df["gene_knockout"] = max_generation_df["gene_knockout"].astype(
        "string"
    )
    max_generation_df["project_name"] = max_generation_df["project_name"].astype(
        "string"
    )
    max_generation_df["seed"] = pd.to_numeric(
        max_generation_df["seed"], errors="coerce"
    ).astype("Int64")
    max_generation_df["max_generation"] = pd.to_numeric(
        max_generation_df["max_generation"], errors="coerce"
    )

    gene_ref = pd.read_csv(GENE_REF, sep="\t", comment="#")
    tu_ref = pd.read_csv(TU_REF, sep="\t", comment="#")

    gene_ids = sorted(gene_ref["gene_id"].dropna().astype(str).unique().tolist())
    tu_ids = sorted(tu_ref["id"].dropna().astype(str).unique().tolist())
    gene_feature_cols = [f"gene_available__{gene_id}" for gene_id in gene_ids]
    tu_feature_cols = [f"tu_available__{tu_id}" for tu_id in tu_ids]

    summary_df["source_group"] = summary_df["source_file"].map(infer_source_group)
    summary_df["lineage_seed"] = (
        summary_df["source_file"].map(extract_seed).astype("Int64")
    )
    summary_df["run_type"] = np.where(
        summary_df["label"].astype(str).eq("baseline"), "baseline", "KO"
    )
    summary_df["ko_target_id"] = summary_df["label"].map(
        lambda value: "" if pd.isna(value) else str(value).strip()
    )
    summary_df["ko_merge_id"] = summary_df["label"].map(
        lambda value: "baseline" if pd.isna(value) else str(value).strip()
    )
    summary_df["ko_gene_ids"] = summary_df.apply(
        lambda row: parse_ko_ids(row["label"])
        if row["source_group"] == "gene_KO"
        else [],
        axis=1,
    )
    summary_df["ko_tu_ids"] = summary_df.apply(
        lambda row: parse_ko_ids(row["label"])
        if row["source_group"] == "TU_KO"
        else [],
        axis=1,
    )
    summary_df["ko_gene_id"] = (
        summary_df["ko_gene_ids"]
        .map(lambda ids: ids[0] if ids else pd.NA)
        .astype("string")
    )
    summary_df["ko_tu_id"] = (
        summary_df["ko_tu_ids"]
        .map(lambda ids: ids[0] if ids else pd.NA)
        .astype("string")
    )
    summary_df["sample_id"] = (
        summary_df["project"].astype(str)
        + "|v"
        + summary_df["variant"].astype(str)
        + "|s"
        + summary_df["lineage_seed"].astype(str)
        + "|"
        + summary_df["label"].astype(str)
    )
    summary_df["growth_next_h"] = summary_df["mean_growth_rate"].astype(float) * 3600.0
    if "max_generation" in summary_df.columns:
        summary_df["max_generation"] = pd.to_numeric(
            summary_df["max_generation"], errors="coerce"
        )

    max_lookup = (
        max_generation_df[["project_name", "seed", "gene_knockout", "max_generation"]]
        .drop_duplicates(subset=["project_name", "seed", "gene_knockout"])
        .rename(
            columns={
                "project_name": "project",
                "seed": "lineage_seed",
                "gene_knockout": "ko_merge_id",
                "max_generation": "max_generation_lookup",
            }
        )
    )
    summary_df = summary_df.merge(
        max_lookup,
        on=["project", "lineage_seed", "ko_merge_id"],
        how="left",
    )
    if "max_generation_lookup" in summary_df.columns:
        if "max_generation" in summary_df.columns:
            summary_df["max_generation"] = summary_df["max_generation"].fillna(
                summary_df["max_generation_lookup"]
            )
        else:
            summary_df["max_generation"] = summary_df["max_generation_lookup"]
        summary_df["max_generation"] = pd.to_numeric(
            summary_df["max_generation"], errors="coerce"
        )
        summary_df = summary_df.drop(columns=["max_generation_lookup"])
    summary_df = summary_df.drop(columns=["ko_merge_id"])
    summary_df["keep_row"] = True
    summary_df.loc[
        summary_df["run_type"].eq("baseline")
        & ~summary_df["project"].eq("gene_knockout_p_list"),
        "keep_row",
    ] = False
    summary_df = summary_df.loc[summary_df["keep_row"]].copy()
    if generation_window_start is not None or generation_window_end is not None:
        lower_bound = (
            generation_window_start if generation_window_start is not None else -np.inf
        )
        upper_bound = (
            generation_window_end if generation_window_end is not None else np.inf
        )
        summary_df = summary_df.loc[
            summary_df["run_type"].eq("baseline")
            | summary_df["max_generation"].between(
                lower_bound, upper_bound, inclusive="both"
            )
        ].copy()
    summary_df["max_generation_record_present"] = summary_df["max_generation"].notna()
    summary_df["max_generation_record_missing"] = ~summary_df[
        "max_generation_record_present"
    ]
    summary_df["failed_generation"] = summary_df["max_generation"].isna() | summary_df[
        "max_generation"
    ].lt(max_generation_criterion)
    summary_df.loc[
        summary_df["source_group"].isin(["gene_KO", "TU_KO"])
        & summary_df["failed_generation"],
        "growth_next_h",
    ] = 0.0
    summary_df["current_growth_h"] = summary_df["growth_next_h"]

    print("Summary shape:", summary_df.shape)
    print(
        "Source-group counts:\n",
        summary_df["source_group"].value_counts(dropna=False).to_string(),
    )
    print("Baseline rows kept:", int(summary_df["run_type"].eq("baseline").sum()))
    print(
        "Rows with max_generation record present:",
        int(summary_df["max_generation_record_present"].sum()),
    )
    print(
        "Rows with max_generation record missing:",
        int(summary_df["max_generation_record_missing"].sum()),
    )
    print(
        f"Rows with max_generation < {max_generation_criterion:g} or missing:",
        int(summary_df["failed_generation"].sum()),
    )
    print(
        f"Rows with max_generation < {max_generation_criterion:g} or missing among present records:",
        int(
            (
                summary_df["max_generation_record_present"]
                & summary_df["failed_generation"]
            ).sum()
        ),
    )
    if generation_window_start is not None or generation_window_end is not None:
        print(
            "Generation window:",
            generation_window_start if generation_window_start is not None else "-inf",
            "to",
            generation_window_end if generation_window_end is not None else "inf",
        )

    gene_mask = summary_df["source_group"].eq("gene_KO")
    tu_mask = summary_df["source_group"].eq("TU_KO")

    gene_features = pd.DataFrame(
        np.nan, index=summary_df.index, columns=gene_feature_cols
    )
    tu_features = pd.DataFrame(np.nan, index=summary_df.index, columns=tu_feature_cols)

    gene_features.loc[gene_mask, :] = 1
    tu_features.loc[tu_mask, :] = 1

    for idx, ko_ids in summary_df.loc[gene_mask, "ko_gene_ids"].items():
        for ko_id in ko_ids:
            col = f"gene_available__{ko_id}"
            if col in gene_features.columns:
                gene_features.at[idx, col] = 0

    for idx, ko_ids in summary_df.loc[tu_mask, "ko_tu_ids"].items():
        for ko_id in ko_ids:
            col = f"tu_available__{ko_id}"
            if col in tu_features.columns:
                tu_features.at[idx, col] = 0

    combined_df = pd.concat(
        [
            summary_df.reset_index(drop=True),
            gene_features.reset_index(drop=True),
            tu_features.reset_index(drop=True),
        ],
        axis=1,
    )

    gene_mask = combined_df["source_group"].eq("gene_KO")
    tu_mask = combined_df["source_group"].eq("TU_KO")

    feature_cols = gene_feature_cols + tu_feature_cols
    feature_manifest = pd.DataFrame(
        {
            "feature_name": feature_cols,
            "feature_group": [
                *("gene_ko" for _ in gene_feature_cols),
                *("tu_ko" for _ in tu_feature_cols),
            ],
        }
    )
    out_parquet = TRAIN_DIR / "gene_TU_combined_training_dataset.parquet"
    out_csv = TRAIN_DIR / "gene_TU_combined_training_dataset.csv"
    out_features = TRAIN_DIR / "gene_TU_combined_feature_list.csv"

    combined_df.to_parquet(out_parquet, index=False)
    combined_df.to_csv(out_csv, index=False)
    feature_manifest.to_csv(out_features, index=False)
    summary_df.to_csv(TRAIN_DIR / "raw_summary_combined.csv", index=False)

    print("Combined dataset shape:", combined_df.shape)
    print(
        "Gene_KO TU block blank:",
        combined_df.loc[gene_mask, tu_feature_cols].isna().all().all(),
    )
    print(
        "TU_KO gene block blank:",
        combined_df.loc[tu_mask, gene_feature_cols].isna().all().all(),
    )
    print(
        "Zero growth rows among failed simulations:",
        int(
            (
                combined_df["failed_generation"] & combined_df["growth_next_h"].eq(0)
            ).sum()
        ),
    )
    print(
        "Rows with max_generation record missing:",
        int(combined_df["max_generation_record_missing"].sum()),
    )
    print("Saved parquet:", out_parquet)
    print("Saved csv:", out_csv)
    print("Saved feature list:", out_features)

    # simple checks
    # assert combined_df["source_group"].value_counts().to_dict() == {
    #     "gene_KO": int(gene_mask.sum()),
    #     "TU_KO": int(tu_mask.sum()),
    # }
    # assert combined_df.loc[gene_mask, tu_feature_cols].isna().all().all()
    # assert combined_df.loc[tu_mask, gene_feature_cols].isna().all().all()
    # assert (
    #     combined_df.loc[combined_df["failed_generation"], "growth_next_h"].eq(0).all()
    # )


if __name__ == "__main__":
    main()
