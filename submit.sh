#!/bin/bash
#SBATCH --job-name=func_gene_analysis
#SBATCH --partition=compute
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/user/home/il22158
#SBATCH --account=emat024603
#SBATCH --output=slurm_logs/func_gene_analysis.%j.out
#SBATCH --mem=120G
#SBATCH --cpus-per-task=24

cd /user/work/il22158/vEcoli
source .venv/bin/activate
# echo "Running simulation for all media conditions with seed 100..."
# # cd reading1
# python runscripts/workflow.py --config configs/N_gene_ko_non_metabolic20_seed100.json

# echo "Extracting growth rates from all three simulations..."
# python reading/growth_rate_extract.py --all --projects all_media_conditions1_seed100 gene_ko_metabolic_seed100 gene_ko_non_metabolic_seed100 --suffix seed100 --lineage-seed 100
# # Output: growth_rate_summary_seed100_all.csv

echo "Performing functional gene analysis..."
python reading/functional_gene_analysis.py  