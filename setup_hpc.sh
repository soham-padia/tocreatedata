#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$REPO_ROOT"

mkdir -p sbatch/logs outputs

module purge
module load python/3.13.5

bash ./update.sh

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -e .
python -m pip install -U "transformers" "torch" "accelerate" "datasets"

python3 --version
python -m pip --version
echo "HPC environment setup complete"
