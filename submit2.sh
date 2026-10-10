#!/bin/bash
#SBATCH --job-name=min_test
#SBATCH --partition=compute
#SBATCH --time=00:01:00
#SBATCH --output=/user/home/il22158/work/slurm_logs/min_test.%j.out
#SBATCH --account=emat024603

echo hello