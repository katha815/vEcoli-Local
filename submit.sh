#!/bin/bash
#SBATCH --job-name=colony_4_gen
#SBATCH --partition=compute
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/user/home/il22158
#SBATCH --account=emat024603
#SBATCH --output=slurm_logs/colony_4_gen.%j.out
#SBATCH --mem=100G
#SBATCH --cpus-per-task=24

# === ENVIRONMENT SETUP ===
cd /user/work/il22158/vEcoli
source .venv/bin/activate

# === VERSION CONTROL ===
SNAPSHOT_BRANCH="snapshots/job-${SLURM_JOB_ID}-$(date +%Y%m%d_%H%M%S)"
if command -v git >/dev/null 2>&1; then
	git checkout -b "$SNAPSHOT_BRANCH"
	git add -A
	git commit -m "Snapshot for job ${SLURM_JOB_ID}" || true
else
	echo "git not found on this node; skipping snapshot branch/commit"
fi

# Metadata fallback for environments without git (required by simulation metadata)
mkdir -p source-info
if command -v git >/dev/null 2>&1; then
	export IMAGE_GIT_HASH="$(git -C /user/work/il22158/vEcoli rev-parse HEAD 2>/dev/null || echo unknown)"
	git -C /user/work/il22158/vEcoli diff HEAD > source-info/git_diff.txt 2>/dev/null || echo "" > source-info/git_diff.txt
else
	export IMAGE_GIT_HASH="nogit-job-${SLURM_JOB_ID}"
	echo "git unavailable on node; no diff captured" > source-info/git_diff.txt
fi

# === JOB EXECUTION ===

echo "Snapshot for job ${SLURM_JOB_ID}"

# echo "Starting downsampling of history parquet files again to make it end in 1/20 size..."
# python reading/downsample_history.py --dir /user/home/il22158/work/vEcoli/out/gene_ko_non_metabolic_seed100/history --n 20 
# python reading/downsample_history.py --dir /user/home/il22158/work/vEcoli/out/gene_ko_metabolic_seed100/history --n 20 
# # change total number of samples to 1/n

# echo "Double daughter simulation inherited from all_media_conditions1..."
# python runscripts/workflow.py --config configs/N_double_daugther_4gen.json
# python runscripts/workflow.py --config configs/N_double_daugther_4gen_d1.json

# echo "Extracting growth rates from 441 single knockouts..."
# python reading/growth_rate_extract.py --all --projects gene_ko_441imported_2seeds --suffix 441_KOs_seed100 --lineage-seed 100 --save-timeseries
# python reading/growth_rate_extract.py --all --projects gene_ko_441imported_2seeds --suffix 441_KOs_seed101 --lineage-seed 101 --save-timeseries

# echo "Performing functional gene analysis..."
# python reading/functional_gene_analysis.py  

echo "Running colony simulation from 3rd to 4th generation..."
python ecoli/experiments/ecoli_engine_process.py --config configs/colony_baseline_4_gen.json