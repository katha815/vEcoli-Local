#!/bin/bash
#SBATCH --job-name=multi_gen_plot
#SBATCH --partition=compute
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/user/home/il22158
#SBATCH --account=emat024603
#SBATCH --output=slurm_logs/multi_gen_plot.%j.out
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16

cd /user/work/il22158/vEcoli
source .venv/bin/activate
echo "Running mutli-generation plot..."
# python runscripts/workflow.py --config configs/N_gene_knockout.json
python /user/home/il22158/work/vEcoli/reading/multi_gen_plot.py