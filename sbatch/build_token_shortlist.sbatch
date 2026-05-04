#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=build-shortlist
#SBATCH --output=sbatch/logs/%x-%j.out
#SBATCH --error=sbatch/logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:v100-sxm2:1

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

mkdir -p sbatch/logs data/token_shortlists
echo "Using existing checkout at $REPO_ROOT"
echo "If you need new code, run bash update.sh on the login node before sbatch."

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  module purge
  module load explorer anaconda3/2024.06 cuda/12.1.1
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_NAME"
else
  echo "Missing runtime environment. Provide CONDA_ENV_NAME." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_FILE="${OUTPUT_FILE:-data/token_shortlists/qwen25_diverse_5000.jsonl}"
SHORTLIST_SIZE="${SHORTLIST_SIZE:-5000}"
WEIRD_RESERVE="${WEIRD_RESERVE:-500}"
CLEAN_POOL_SIZE="${CLEAN_POOL_SIZE:-20000}"
WEIRD_POOL_SIZE="${WEIRD_POOL_SIZE:-5000}"
PROJECT_DIM="${PROJECT_DIM:-32}"
SEED="${SEED:-0}"

nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY

python3 scripts/build_token_shortlist.py \
  --model-name "$MODEL_NAME" \
  --output-file "$OUTPUT_FILE" \
  --shortlist-size "$SHORTLIST_SIZE" \
  --weird-reserve "$WEIRD_RESERVE" \
  --clean-pool-size "$CLEAN_POOL_SIZE" \
  --weird-pool-size "$WEIRD_POOL_SIZE" \
  --project-dim "$PROJECT_DIM" \
  --seed "$SEED"
