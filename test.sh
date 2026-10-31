#!/bin/bash
#SBATCH --job-name=array_test
#SBATCH --partition=test
#SBATCH --time=00:05:00
#SBATCH --account=emat024603
#SBATCH --chdir=/user/home/il22158
#SBATCH --output=/user/home/il22158/work/slurm_logs/array_test.%A_%a.out
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-4

TARGETS=(baseline acetate no_ox succinate with_aa)
target="${TARGETS[$SLURM_ARRAY_TASK_ID]}"

echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT}"
echo "Target: ${target}"
echo "Node: $(hostname)  Started: $(date)"
sleep 10
echo "Finished: $(date)"