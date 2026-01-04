#!/bin/bash
#SBATCH --job-name=gene_screen3
#SBATCH --partition=compute
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/user/home/il22158
#SBATCH --account=emat024603
#SBATCH --output=slurm_logs/gene_screen3.%j.out
#SBATCH --mem=120G
#SBATCH --cpus-per-task=24

cd /user/work/il22158/vEcoli
source .venv/bin/activate
echo "Running TDI plot betweenmedia conditions..."
cd reading
python tdi_plot.py
