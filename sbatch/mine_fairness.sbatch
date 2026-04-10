#!/usr/bin/env bash
#SBATCH --job-name=mine-fairness
#SBATCH --output=sbatch/logs/%x-%j.out
#SBATCH --error=sbatch/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p sbatch/logs outputs

# Uncomment and adapt if your cluster uses environment modules.
# module purge
# module load cuda/12.1
# module load python/3.11

bash ./update.sh

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-9B}"
AXIS="${AXIS:-fairness}"
RUN_DIR="${RUN_DIR:-outputs/${AXIS}_${SLURM_JOB_ID}}"

mkdir -p "$RUN_DIR"

python3 scripts/mine_candidates.py \
  --model-name "$MODEL_NAME" \
  --direction-file data/direction_spec.sample.json \
  --pairs-path data/axes \
  --axis "$AXIS" \
  --lexicon-file data/lexicon.txt \
  --output-file "$RUN_DIR/candidates.jsonl" \
  --dataset-file "$RUN_DIR/dataset.jsonl"
