#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
AXIS="${AXIS:-fairness}"
PARTITION="${PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MEMORY="${MEMORY:-48G}"
CPUS="${CPUS:-8}"
RESET_VENV="${RESET_VENV:-1}"
RUN_UPDATE="${RUN_UPDATE:-0}"
BEAM_WIDTH="${BEAM_WIDTH:-6}"
TOP_K="${TOP_K:-12}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"

cd "$REPO_ROOT"

mkdir -p sbatch/logs outputs

module purge
module load python/3.13.5

if [[ "$RUN_UPDATE" == "1" ]]; then
  bash ./update.sh
fi

if [[ "$RESET_VENV" == "1" ]]; then
  rm -rf .venv
fi

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -e . --no-deps
python -m pip install -U "transformers>=4.44.0" "accelerate>=0.33.0" "datasets>=2.20.0" "safetensors>=0.4.5"
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install --no-cache-dir --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch
import transformers
import datasets
import accelerate

print("python ok")
print("torch:", torch.__version__, "cuda_build:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
PY

JOB_OUTPUT="$REPO_ROOT/sbatch/logs/mine-fairness-%j.out"
JOB_ERROR="$REPO_ROOT/sbatch/logs/mine-fairness-%j.err"

JOB_ID="$(
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
    --output="$JOB_OUTPUT" \
    --error="$JOB_ERROR" \
    --wrap='/bin/bash -c "
      set -e
      echo STEP: start
      module purge
      module load python/3.13.5
      cd \"$SLURM_SUBMIT_DIR\"
      source .venv/bin/activate
      export PYTHONUNBUFFERED=1
      export HF_HUB_DISABLE_PROGRESS_BARS=1
      echo STEP: python $(python3 --version)
      python - <<'"'"'PY'"'"'
import torch
print(\"torch:\", torch.__version__)
print(\"torch cuda build:\", torch.version.cuda)
print(\"cuda available:\", torch.cuda.is_available())
print(\"device count:\", torch.cuda.device_count())
PY
      mkdir -p outputs/'"$AXIS"'_\$SLURM_JOB_ID
      echo STEP: launching mining
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
echo "Model: $MODEL_NAME"
echo "Axis: $AXIS"
echo "Watch with:"
echo "  squeue -j $JOB_ID"
echo "  tail -f sbatch/logs/mine-fairness-$JOB_ID.out"
echo "  tail -f sbatch/logs/mine-fairness-$JOB_ID.err"
