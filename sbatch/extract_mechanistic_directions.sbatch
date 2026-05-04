#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=extract-directions
#SBATCH --output=sbatch/logs/%x-%j.out
#SBATCH --error=sbatch/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100-sxm2:1

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

mkdir -p sbatch/logs outputs/mechanistic_directions
echo "Using existing checkout at $REPO_ROOT"
echo "If you need new code, run bash update.sh on the login node before sbatch."

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  module purge
  module load explorer anaconda3/2024.06 cuda/12.1.1
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_NAME"
elif [[ -f ".venv/bin/activate" ]]; then
  module purge
  module load python/3.13.5
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "Missing runtime environment. Provide CONDA_ENV_NAME or create .venv first." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
LAYER_INDEX="${LAYER_INDEX:--1}"
AXIS="${AXIS:-}"
AXES="${AXES:-}"
RUN_SLUG="${RUN_SLUG:-all_axes_layer_${LAYER_INDEX}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/mechanistic_directions/${RUN_SLUG}_${SLURM_JOB_ID}}"

EXTRA_ARGS=()
if [[ -n "$AXIS" ]]; then
  EXTRA_ARGS+=(--axis "$AXIS")
fi
if [[ -n "$AXES" ]]; then
  EXTRA_ARGS+=(--axes "$AXES")
fi

nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY

python3 scripts/extract_mechanistic_directions.py \
  --model-name "$MODEL_NAME" \
  --pairs-path data/axes \
  --output-dir "$OUTPUT_DIR" \
  --layer-index "$LAYER_INDEX" \
  "${EXTRA_ARGS[@]}"
