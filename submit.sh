#!/bin/bash
#SBATCH --job-name=colony_6th_gen
#SBATCH --partition=compute
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/user/home/il22158
#SBATCH --account=emat024603
#SBATCH --output=slurm_logs/colony_6th_gen.%j.out
#SBATCH --mem=200G
#SBATCH --cpus-per-task=24

set -euo pipefail

# === ENVIRONMENT SETUP ===
REPO_ROOT=/user/work/il22158/vEcoli
cd "$REPO_ROOT"
source .venv/bin/activate

if command -v module >/dev/null 2>&1; then
	module restore || true
	if ! command -v java >/dev/null 2>&1; then
		module load languages/java-sdk/22.0.2 || true
	fi
fi

if ! command -v java >/dev/null 2>&1; then
	echo "ERROR: java not found in batch environment PATH" >&2
	exit 1
fi

if ! command -v nextflow >/dev/null 2>&1; then
	echo "ERROR: nextflow not found in batch environment PATH" >&2
	exit 1
fi

java -version
nextflow -version | head -n 2

# === VERSION CONTROL ===
SNAPSHOT_BRANCH="snapshots/job-${SLURM_JOB_ID}-$(date +%Y%m%d_%H%M%S)"
if command -v git >/dev/null 2>&1; then
	# Keep snapshot feature, but avoid staging bulky runtime outputs.
	git checkout -b "$SNAPSHOT_BRANCH"
	git add -u
	git add submit.sh configs runscripts 2>/dev/null || true
	git reset -q -- out slurm_logs nextflow_temp source-info 2>/dev/null || true
	if ! git diff --cached --quiet; then
		git commit -m "Snapshot for job ${SLURM_JOB_ID}"
	else
		echo "No source/config changes to commit for snapshot"
	fi
else
	echo "git not found on this node; skipping snapshot branch/commit"
fi

# Metadata fallback for environments without git (required by simulation metadata)
 
	export IMAGE_GIT_HASH="nogit-job-${SLURM_JOB_ID}"
	echo "git unavailable on node; no diff captured" > source-info/git_diff.txt
	echo "git unavailable on node; no status captured" > source-info/git_status.txt
fi

# === JOB EXECUTION ===

echo "Snapshot for job ${SLURM_JOB_ID}"

# echo "Starting downsampling of history parquet files again to make it end in 1/20 size..."
# python reading/downsample_history.py --dir /user/home/il22158/work/vEcoli/out/gene_ko_non_metabolic_seed100/history --n 20 
# python reading/downsample_history.py --dir /user/home/il22158/work/vEcoli/out/gene_ko_metabolic_seed100/history --n 20 
# # change total number of samples to 1/n

# echo "Workflow testing..."
# python runscripts/workflow.py --config configs/N_gene_knockout_test.json

# echo "Extracting growth rates from 441 single knockouts..."
# python reading/growth_rate_extract.py --all --projects gene_ko_441imported_2seeds --suffix 441_KOs_seed100 --lineage-seed 100 --save-timeseries
# python reading/growth_rate_extract.py --all --projects gene_ko_441imported_2seeds --suffix 441_KOs_seed101 --lineage-seed 101 --save-timeseries

# echo "Performing functional gene analysis..."
# python reading/functional_gene_analysis.py  

echo "Running colony simulation from 5th to 6th generation..."
python ecoli/experiments/ecoli_engine_process.py --config configs/colony_baseline_6_gen.json