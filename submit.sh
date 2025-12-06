#!/bin/sh
#SBATCH --job-name=geno_altered
#SBATCH --partition=compute # mlcnu is the partition with gpu
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/user/home/il22158
#SBATCH --account=emat024603
#SBATCH --output=slurm_logs/geno_altered.%j.out
#SBATCH --mem=50G
#SBATCH --cpus-per-task=3

cd /user/work/il22158/vEcoli
source .venv/bin/activate
echo "Running test script..."
python runscripts/workflow.py --config configs/two_generations.json