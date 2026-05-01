#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-humanity-qwen25}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
AXIS="${AXIS:-fairness}"
PARTITION="${PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MEMORY="${MEMORY:-48G}"
CPUS="${CPUS:-8}"
RUN_UPDATE="${RUN_UPDATE:-0}"
BEAM_WIDTH="${BEAM_WIDTH:-6}"
TOP_K="${TOP_K:-12}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"

cd "$REPO_ROOT"

mkdir -p sbatch/logs outputs

module purge
module load miniconda3/25.9.1

if [[ "$RUN_UPDATE" == "1" ]]; then
  bash ./update.sh
fi

bash ./setup_hpc_conda.sh

JOB_ID="$(
  CONDA_ENV_NAME="$CONDA_ENV_NAME" \
  sbatch \
    --parsable \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --job-name="mine-fairness" \
    --output="$REPO_ROOT/sbatch/logs/mine-fairness-%j.out" \
    --error="$REPO_ROOT/sbatch/logs/mine-fairness-%j.err" \
    --wrap='/bin/bash -c "
      set -e
      module purge
      module load miniconda3/25.9.1
      eval \"\$(conda shell.bash hook)\"
      conda activate '"$CONDA_ENV_NAME"'
      cd \"$SLURM_SUBMIT_DIR\"
      export PYTHONUNBUFFERED=1
      echo STEP: python \$(python3 --version)
      python - <<'"'"'PY'"'"'
import torch
print(\"torch:\", torch.__version__)
print(\"torch cuda build:\", torch.version.cuda)
print(\"cuda available:\", torch.cuda.is_available())
print(\"device count:\", torch.cuda.device_count())
PY
      mkdir -p outputs/'"$AXIS"'_\$SLURM_JOB_ID
      python3 scripts/mine_candidates.py \
        --model-name '"$MODEL_NAME"' \
        --direction-file data/direction_spec.sample.json \
        --pairs-path data/axes \
        --axis '"$AXIS"' \
        --lexicon-file data/lexicon.txt \
        --beam-width '"$BEAM_WIDTH"' \
        --top-k '"$TOP_K"' \
        --max-new-tokens '"$MAX_NEW_TOKENS"' \
        --output-file outputs/'"$AXIS"'_\$SLURM_JOB_ID/candidates.jsonl \
        --dataset-file outputs/'"$AXIS"'_\$SLURM_JOB_ID/dataset.jsonl
    "'
)"

echo "Submitted job $JOB_ID"
echo "Conda env: $CONDA_ENV_NAME"
echo "Model: $MODEL_NAME"
echo "Axis: $AXIS"
echo "Watch with:"
echo "  squeue -j $JOB_ID"
echo "  tail -f sbatch/logs/mine-fairness-$JOB_ID.out"
echo "  tail -f sbatch/logs/mine-fairness-$JOB_ID.err"
