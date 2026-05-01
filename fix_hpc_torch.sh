#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$REPO_ROOT"

module purge
module load python/3.13.5

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -e . --no-deps
python -m pip install -U "transformers>=4.44.0" "safetensors>=0.4.5"
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip install --no-cache-dir --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import os
import shutil
import subprocess
import torch

print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

has_nvidia_smi = shutil.which("nvidia-smi") is not None
inside_slurm = "SLURM_JOB_ID" in os.environ

if has_nvidia_smi:
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except Exception as exc:
        print(f"nvidia-smi check failed: {exc}")

if torch.cuda.is_available():
    print("GPU-ready torch install confirmed.")
elif inside_slurm or has_nvidia_smi:
    raise SystemExit(
        "Torch is still not seeing CUDA on a GPU-capable node. "
        "The install is not fixed yet."
    )
else:
    print(
        "Torch wheel install completed. CUDA cannot be verified on this login node. "
        "Run an srun GPU shell and repeat the torch.cuda.is_available() check."
    )
PY

echo "HPC torch repair complete"
