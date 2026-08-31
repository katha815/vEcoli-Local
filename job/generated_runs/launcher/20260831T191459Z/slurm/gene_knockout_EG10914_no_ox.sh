#!/bin/bash
#SBATCH --job-name=gene_knockout_EG10914_no_ox
#SBATCH --partition=compute
#SBATCH --time=14-00:00:00
#SBATCH --chdir=/user/home/il22158
#SBATCH --account=emat024603
#SBATCH --output=/user/home/il22158/work/slurm_logs/gene_knockout_EG10914_no_ox.%j.out
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

WORK_DIR="/user/home/il22158/work/vEcoli"
cd "$WORK_DIR" || exit 1

source "$WORK_DIR/.venv/bin/activate"

module load languages/java-sdk/22.0.2 openssh/9.7p1-uyheegq git
module list

nextflow -version

    python runscripts/workflow.py --config /user/home/il22158/work/vEcoli/job/generated_runs/launcher/20260831T191459Z/configs/gene_knockout_EG10914_no_ox.json

python /user/home/il22158/work/vEcoli/reading/growth_rate_extract.py     --all     --projects gene_knockout_EG10914_no_ox     --save-timeseries     --suffix gene_knockout_EG10914_no_ox     --lineage-seeds 100 101     --generations 6 7 8 9 10 11 12 13 14
