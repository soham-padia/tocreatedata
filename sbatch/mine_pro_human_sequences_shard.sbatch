#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=mech-shard
#SBATCH --output=sbatch/logs/%x-%A_%a.out
#SBATCH --error=sbatch/logs/%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

mkdir -p sbatch/logs
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
DIRECTION_TENSORS="${DIRECTION_TENSORS:?set DIRECTION_TENSORS to a directions.pt path}"
DIRECTION_NAME="${DIRECTION_NAME:-global}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to an output root directory}"
SHARD_INDEX="${SHARD_INDEX:?set SHARD_INDEX}"
NUM_SHARDS="${NUM_SHARDS:?set NUM_SHARDS}"

nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY

python3 scripts/mine_pro_human_sequences.py \
  --model-name "$MODEL_NAME" \
  --direction-tensors "$DIRECTION_TENSORS" \
  --direction-name "$DIRECTION_NAME" \
  --lexicon-file data/lexicon.txt \
  --output-dir "$RUN_ROOT/shards/shard_${SHARD_INDEX}" \
  --min-phrase-len "${MIN_PHRASE_LEN:-1}" \
  --max-phrase-len "${MAX_PHRASE_LEN:-5}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --retain-top-k "${RETAIN_TOP_K:-5000}" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS"
