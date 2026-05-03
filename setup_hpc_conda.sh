#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-humanity-qwen25}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.4}"

cd "$REPO_ROOT"

module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

eval "$(conda shell.bash hook)"

conda remove -y -n "$CONDA_ENV_NAME" --all >/dev/null 2>&1 || true
conda create -y -n "$CONDA_ENV_NAME" -c conda-forge "python=$PYTHON_VERSION"
conda activate "$CONDA_ENV_NAME"

python -m pip install --no-cache-dir -U pip setuptools wheel
python -m pip install --no-cache-dir -e . --no-deps
python -m pip install --no-cache-dir "transformers>=4.44.0" "safetensors>=0.4.5"
python -m pip install --no-cache-dir --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch
import transformers

print("python ok")
print("torch:", torch.__version__, "cuda_build:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
PY

echo "Conda environment ready: $CONDA_ENV_NAME"
