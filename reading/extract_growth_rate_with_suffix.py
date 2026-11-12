# This code calls growth_rate_extract on the simulation output's simdata for projects with specific suffixes,
# and saves all extract data into one collective csv
# time-series data is saved in an collective parquet as well

from pathlib import Path

import pandas as pd

import growth_rate_extract

project_suffix = [
    "s1",
    "errored",
    "errored_no_overlap",
    "errored_overlap_s2",
]  # ["_EG10006_with_aa_s1"]
search_dir = Path("/user/home/il22158/work/vEcoli/out")
output_dir = Path("/user/home/il22158/work/vEcoli/reading/results/growth_rate")

all_extract_data = []
all_time_series_data = []

for project in sorted(search_dir.iterdir()):
    if not project.is_dir() or not project.name.endswith(tuple(project_suffix)):
        continue
    print(f"Processing project: {project.name}")
    if project.name.lower().startswith("wildtype"):
        variant_key = "null"
    elif "media" in project.name.lower() or "condition" in project.name.lower():
        variant_key = "condition"
    else:
        variant_key = "gene_knockout"

    history = project / "history" / f"experiment_id={project.name}"
    seeds = set()
    for seed_dir in history.glob("variant=*/lineage_seed=*"):
        if seed_dir.is_dir():
            seeds.add(int(seed_dir.name.split("=", 1)[1]))

    if not seeds:
        print(f"No lineage seeds found for {project.name}")
        continue

    for seed in sorted(seeds):
        summary = growth_rate_extract.extract_growth_rate_data(
            project.name,
            variant_key,
            generation=None,  # type: ignore
            lineage_seed=seed,
        )
        timeseries = growth_rate_extract.extract_growth_rate_data(
            project.name,
            variant_key,
            generation=None,  # type: ignore
            save_timeseries=True,
            lineage_seed=seed,
        )

        if summary.empty or timeseries.empty:
            print(f"No growth data for {project.name}, seed {seed}")
            continue

        all_extract_data.append(summary)
        all_time_series_data.append(timeseries)

if not all_extract_data:
    raise SystemExit("No growth-rate data was extracted.")

output_dir.mkdir(parents=True, exist_ok=True)
summary_path = output_dir / "growth_rate_summary_errored_round1_all.csv"
timeseries_path = output_dir / "growth_rate_timeseries_errored_round1_all.parquet"

pd.concat(all_extract_data, ignore_index=True).to_csv(summary_path, index=False)
pd.concat(all_time_series_data, ignore_index=True).to_parquet(
    timeseries_path, index=False
)

print(f"Saved summary: {summary_path}")
print(f"Saved time series: {timeseries_path}")
