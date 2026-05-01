#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$REPO_ROOT"

mkdir -p sbatch/logs outputs

module purge
module load python/3.13.5

if [[ "${RUN_UPDATE:-0}" == "1" ]]; then
  bash ./update.sh
else
  echo "Skipping git update inside setup_hpc.sh; run bash update.sh on the login node first."
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

python3 --version
python -m pip --version
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY
echo "HPC environment setup complete"
