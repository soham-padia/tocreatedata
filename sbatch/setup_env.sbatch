#!/usr/bin/env bash
#SBATCH --job-name=setup-env
#SBATCH --output=sbatch/logs/%x-%j.out
#SBATCH --error=sbatch/logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

mkdir -p sbatch/logs
module purge
module load python/3.13.5

echo "Using existing checkout at $REPO_ROOT"
echo "If you need new code, run bash update.sh on the login node before sbatch."

bash ./setup_hpc.sh
