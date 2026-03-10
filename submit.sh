#!/bin/bash
#SBATCH --job-name=gowth_r_441
#SBATCH --partition=compute
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/user/home/il22158
#SBATCH --account=emat024603
#SBATCH --output=slurm_logs/growth_rate_441.%j.out
#SBATCH --mem=50G
#SBATCH --cpus-per-task=10

# === ENVIRONMENT SETUP ===
cd /user/work/il22158/vEcoli
source .venv/bin/activate

# === VERSION CONTROL ===
SNAPSHOT_BRANCH="snapshots/job-${SLURM_JOB_ID}-$(date +%Y%m%d_%H%M%S)"
git checkout -b "$SNAPSHOT_BRANCH"
git add -A
git commit -m "Snapshot for job ${SLURM_JOB_ID}" || true
git checkout -  # Go back to previous branch

# === JOB EXECUTION ===

echo "Snapshot for job ${SLURM_JOB_ID}"

# echo "Starting downsampling of history parquet files again to make it end in 1/20 size..."
# python reading/downsample_history.py --dir /user/home/il22158/work/vEcoli/out/gene_ko_non_metabolic_seed100/history --n 20 
# python reading/downsample_history.py --dir /user/home/il22158/work/vEcoli/out/gene_ko_metabolic_seed100/history --n 20 
# # change total number of samples to 1/n

# echo "Rerun 441 gene knockouts with downsampled time step..."
# python runscripts/workflow.py --config configs/N_gene_ko_441imported_2seeds.json

echo "Extracting growth rates from 441 single knockouts..."
python reading/growth_rate_extract.py --all --projects gene_ko_441imported_2seeds --suffix 441_KOs_seed100 --lineage-seed 100 --save-timeseries
python reading/growth_rate_extract.py --all --projects gene_ko_441imported_2seeds --suffix 441_KOs_seed101 --lineage-seed 101 --save-timeseries

# echo "Performing functional gene analysis..."
# python reading/functional_gene_analysis.py  

# echo "Running colony simulation starting from the saved 2-gen results..."
# python ecoli/experiments/ecoli_engine_process.py --config configs/colony_baseline_test2.json