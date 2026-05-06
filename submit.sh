#!/bin/bash
#SBATCH --job-name=gene_KO_p_list
#SBATCH --partition=compute
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/user/home/il22158
#SBATCH --account=emat024603
#SBATCH --output=slurm_logs/gene_KO_p_list.%j.out
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16

# == Work directory setup ==
WORK_DIR="/user/home/il22158/work/vEcoli"
cd "$WORK_DIR" || exit

module restore        # Restore the saved default modules

# == Python environment setup ==
source "$WORK_DIR/.venv/bin/activate"

# === VERSION CONTROL ===
# git add. #stages changes under the current directory 
# SNAPSHOT_BRANCH="snapshots/job-${SLURM_JOB_ID}-$(date +%Y%m%d_%H%M%S)"
# git checkout -b "$SNAPSHOT_BRANCH"
# git commit -m "Snapshot for job ${SLURM_JOB_ID}" || true
# git checkout -  # Go back to previous branch
# echo "Snapshot for job ${SLURM_JOB_ID}"

# === JOB EXECUTION ===

# echo "Starting downsampling of history parquet files again to make it end in 1/20 size..."
# python reading/downsample_history.py --dir /user/home/il22158/work/vEcoli/out/gene_ko_non_metabolic_seed100/history --n 20 
# python reading/downsample_history.py --dir /user/home/il22158/work/vEcoli/out/gene_ko_metabolic_seed100/history --n 20 
# # change total number of samples to 1/n

echo "Run gene knockout with p-list..."
python runscripts/workflow.py --config configs/N_gene_knockout_p_list.json

# echo "Extracting growth rates from 441 single knockouts..."
# python reading/growth_rate_extract.py --all --projects gene_ko_441imported_2seeds --suffix 441_KOs_seed100 --lineage-seed 100 --save-timeseries
# python reading/growth_rate_extract.py --all --projects gene_ko_441imported_2seeds --suffix 441_KOs_seed101 --lineage-seed 101 --save-timeseries

# echo "Performing functional gene analysis..."
# python reading/functional_gene_analysis.py  

# echo "Running colony simulation from 5th to 6th generation..."
# python ecoli/experiments/ecoli_engine_process.py --config configs/colony_baseline_6_gen.json

# echo "Preprocessing growth rate to fold change data (all parquet files)..."
# GENE_LIST=/user/home/il22158/work/vEcoli/reading/imported/Single_KO_RNA_names.txt
# KO100=/user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_timeseries_441_KOs_seed100_all.parquet
# KO101=/user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_timeseries_441_KOs_seed101_all.parquet
# S100=/user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_timeseries_seed100_all.parquet
# S101=/user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_timeseries_seed101_all.parquet
# DEF=/user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_timeseries_default_all.parquet

# # 1. 441 KO files use *20 step conversion
# python surrogate/preprocess.py \
# 	--mode batch \
# 	--gene-list "${GENE_LIST}" \
# 	--timeseries-files "${KO100}" "${KO101}" \
# 	--step-scale 20 \
# 	--output-prefix surrogate_preprocessed_ko441

# 2. seed/default files include baseline and use step conversion 1
# python surrogate/preprocess.py \
# 	--mode batch \
# 	--gene-list "${GENE_LIST}" \
# 	--timeseries-files "${S100}" "${S101}" "${DEF}" \
# 	--step-scale 1 \
# 	--output-prefix surrogate_preprocessed_seed_default

# echo "Gene screen for outliers..."
# set -e

# GENE_SCREEN="/user/home/il22158/work/vEcoli/reading/gene_screen.py"
# LIST_DIR="/user/home/il22158/work/vEcoli/surrogate/results/failure/outlier_gene_lists_by_project"
# COMMON_ARGS="--generations 1 2 3 4 5 6 7 8 --subset 0"

# # Format: project<TAB>lineage_seed<TAB>variants
# while IFS=$'\t' read -r project seed variants; do
# 	[[ -z "$project" ]] && continue
# 	gene_list="$LIST_DIR/outlier_genes_unique__${project}.txt"
# 	output_project="outlier_translation_${project}_seed${seed}"

# 	python3 "$GENE_SCREEN" \
# 		--project "$project" \
# 		--variants $variants \
# 		--lineage-seed "$seed" \
# 		$COMMON_ARGS \
# 		--gene-list "$gene_list" \
# 		--output-project "$output_project"
# done <<'EOF'
# gene_ko_441imported_2seeds	100	7 11 24 31 41 48 57 59 65 84 95 98 101 117 122 125 133 135 145 157 189 191 195 200 212 213 218 226 230 239 240 241 244 246 267 284 286 288 289 297 317 326 329 356 357 368 370 400 419 433 436
# gene_ko_441imported_2seeds	101	7 11 24 31 57 59 65 95 98 101 117 124 125 133 135 145 157 191 195 200 213 218 226 230 241 244 246 267 284 286 288 297 329 357 368 370 400 433 436
# gene_ko_metabolic_seed100	100	7
# gene_ko_non_metabolic_seed100	100	7
# gene_ko_trial40_seed101	101	7
# EOF