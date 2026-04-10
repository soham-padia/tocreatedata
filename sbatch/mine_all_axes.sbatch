#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=mine-all-axes
#SBATCH --output=sbatch/logs/%x-%j.out
#SBATCH --error=sbatch/logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

mkdir -p sbatch/logs outputs
module purge
module load python/3.13.5

echo "Using existing checkout at $REPO_ROOT"
echo "If you need new code, run bash update.sh on the login node before sbatch."

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing .venv. Run sbatch sbatch/setup_env.sbatch first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONUNBUFFERED=1
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
RUN_DIR="${RUN_DIR:-outputs/all_axes_${SLURM_JOB_ID}}"

mkdir -p "$RUN_DIR"

nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY

python3 scripts/mine_candidates.py \
  --model-name "$MODEL_NAME" \
  --direction-file data/direction_spec.sample.json \
  --pairs-path data/axes \
  --lexicon-file data/lexicon.txt \
  --output-file "$RUN_DIR/candidates.jsonl" \
  --dataset-file "$RUN_DIR/dataset.jsonl"
