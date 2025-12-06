#!/bin/bash
#SBATCH --job-name=all_conditions
#SBATCH --partition=compute
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/user/home/il22158
#SBATCH --account=emat024603
#SBATCH --output=slurm_logs/all_conditions.%j.out
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16

cd /user/work/il22158/vEcoli
source .venv/bin/activate
echo "Running all media conditions..."
python runscripts/workflow.py --config configs/N_all_media_conditions.json