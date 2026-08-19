from pathlib import Path
import re
import atexit
import sys

import numpy as np
import pandas as pd

BASE = Path("/user/home/il22158/work/vEcoli")
TRAIN_DIR = BASE / "surrogate" / "train_data"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

MAX_GENERATION_FILE = (
    BASE / "surrogate/results/default_KO/gene_knockout_generation_summary.csv"
)
LOG_FILE = TRAIN_DIR / "gene_TU_combined_training_dataset.log"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


log_handle = LOG_FILE.open("w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_handle)
sys.stderr = Tee(sys.__stderr__, log_handle)
print(f"Logging to: {LOG_FILE}")
atexit.register(log_handle.close)

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

TIMESERIES_FILES = [
    BASE
    / "reading/results/growth_rate/growth_rate_timeseries_gene_knockout_102trails_seed100_all.parquet",
    BASE
    / "reading/results/growth_rate/growth_rate_timeseries_gene_knockout_102trails_seed101_all.parquet",
    BASE
    / "reading/results/growth_rate/growth_rate_timeseries_gene_knockout_leftover_seed101_all.parquet",
    BASE
    / "reading/results/growth_rate/growth_rate_timeseries_gene_knockout_leftover_seed100_all.parquet",
    BASE
    / "reading/results/growth_rate/growth_rate_timeseries_gene_knockout_TU_ID_rest_list_rerun12_seed10_all.parquet",
    BASE
    / "reading/results/growth_rate/growth_rate_timeseries_gene_knockout_TU_ID_rest_list_rerun12_seed11_all.parquet",
]

GENE_REF = BASE / "reconstruction/ecoli/flat/rnas.tsv"
TU_REF = BASE / "reconstruction/ecoli/flat/transcription_units.tsv"


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#").copy()
    df["source_file"] = path.name
    df["label_raw"] = df["label"]
    df["label"] = df["label"].astype("string").str.replace(r"^KO:\s*", "", regex=True)
    return df


def load_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df["source_file"] = path.name
    return df


def infer_source_group(source_file: str) -> str:
    key = source_file.lower()
    if "tu_id_rest_list_rerun12" in key:
        return "TU_KO"
    if "102trails" in key or "leftover" in key:
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


summary_df = pd.concat(
    [load_summary(path) for path in SUMMARY_FILES], ignore_index=True
)
timeseries_df = pd.concat(
    [load_timeseries(path) for path in TIMESERIES_FILES], ignore_index=True
)
max_generation_df = pd.read_csv(MAX_GENERATION_FILE)
max_generation_df["gene_knockout"] = max_generation_df["gene_knockout"].astype("string")
max_generation_df["project_name"] = max_generation_df["project_name"].astype("string")
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
summary_df["lineage_seed"] = summary_df["source_file"].map(extract_seed).astype("Int64")
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
    lambda row: parse_ko_ids(row["label"]) if row["source_group"] == "gene_KO" else [],
    axis=1,
)
summary_df["ko_tu_ids"] = summary_df.apply(
    lambda row: parse_ko_ids(row["label"]) if row["source_group"] == "TU_KO" else [],
    axis=1,
)
summary_df["ko_gene_id"] = (
    summary_df["ko_gene_ids"].map(lambda ids: ids[0] if ids else pd.NA).astype("string")
)
summary_df["ko_tu_id"] = (
    summary_df["ko_tu_ids"].map(lambda ids: ids[0] if ids else pd.NA).astype("string")
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

max_lookup = (
    max_generation_df[["project_name", "seed", "gene_knockout", "max_generation"]]
    .drop_duplicates(subset=["project_name", "seed", "gene_knockout"])
    .rename(
        columns={
            "project_name": "project",
            "seed": "lineage_seed",
            "gene_knockout": "ko_merge_id",
        }
    )
)
summary_df = summary_df.merge(
    max_lookup,
    on=["project", "lineage_seed", "ko_merge_id"],
    how="left",
)
summary_df["max_generation"] = pd.to_numeric(
    summary_df["max_generation"], errors="coerce"
)
summary_df = summary_df.drop(columns=["ko_merge_id"])
summary_df["keep_row"] = True
summary_df.loc[
    summary_df["run_type"].eq("baseline")
    & ~summary_df["project"].eq("gene_knockout_p_list"),
    "keep_row",
] = False
summary_df = summary_df.loc[summary_df["keep_row"]].copy()
summary_df["max_generation_record_present"] = summary_df["max_generation"].notna()
summary_df["max_generation_record_missing"] = ~summary_df[
    "max_generation_record_present"
]
summary_df["failed_generation"] = summary_df["max_generation"].isna() | summary_df[
    "max_generation"
].lt(8)
summary_df.loc[
    summary_df["source_group"].isin(["gene_KO", "TU_KO"])
    & summary_df["failed_generation"],
    "growth_next_h",
] = 0.0
summary_df["current_growth_h"] = summary_df["growth_next_h"]

print("Summary shape:", summary_df.shape)
print("Timeseries shape:", timeseries_df.shape)
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
    "Rows with max_generation < 8 or missing:",
    int(summary_df["failed_generation"].sum()),
)
print(
    "Rows with max_generation < 8 or missing among present records:",
    int(
        (
            summary_df["max_generation_record_present"]
            & summary_df["failed_generation"]
        ).sum()
    ),
)


gene_mask = summary_df["source_group"].eq("gene_KO")
tu_mask = summary_df["source_group"].eq("TU_KO")

gene_features = pd.DataFrame(np.nan, index=summary_df.index, columns=gene_feature_cols)
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
timeseries_df.to_parquet(TRAIN_DIR / "raw_timeseries_combined.parquet", index=False)

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
    int((combined_df["failed_generation"] & combined_df["growth_next_h"].eq(0)).sum()),
)
print(
    "Rows with max_generation record missing:",
    int(combined_df["max_generation_record_missing"].sum()),
)
print("Saved parquet:", out_parquet)
print("Saved csv:", out_csv)
print("Saved feature list:", out_features)

# simple checks
assert combined_df["source_group"].value_counts().to_dict() == {
    "gene_KO": int(gene_mask.sum()),
    "TU_KO": int(tu_mask.sum()),
}
assert combined_df.loc[gene_mask, tu_feature_cols].isna().all().all()
assert combined_df.loc[tu_mask, gene_feature_cols].isna().all().all()
assert combined_df.loc[combined_df["failed_generation"], "growth_next_h"].eq(0).all()

log_handle.close()
