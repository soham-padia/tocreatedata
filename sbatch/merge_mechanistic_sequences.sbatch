#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=mech-merge
#SBATCH --output=sbatch/logs/%x-%j.out
#SBATCH --error=sbatch/logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

mkdir -p sbatch/logs
module purge
module load explorer anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME:-humanity-qwen25}"

RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to an output root directory}"

python3 scripts/merge_mechanistic_sequences.py \
  --input-root "$RUN_ROOT" \
  --final-top-k "${FINAL_TOP_K:-1000}"
