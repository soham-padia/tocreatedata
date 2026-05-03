#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-humanity-qwen25}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MODE="${MODE:-parallel}"
PARTITION="${PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
MEMORY="${MEMORY:-48G}"
CPUS="${CPUS:-8}"
RUN_UPDATE="${RUN_UPDATE:-0}"
BEAM_WIDTH="${BEAM_WIDTH:-3}"
TOP_K="${TOP_K:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_PHRASE_LEN="${MAX_PHRASE_LEN:-2}"
SKIP_AXES="${SKIP_AXES:-}"

cd "$REPO_ROOT"

mkdir -p sbatch/logs outputs/all_axes

module purge
module load explorer anaconda3/2024.06

if [[ "$RUN_UPDATE" == "1" ]]; then
  bash ./update.sh
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  cat <<EOF
Missing conda env: $CONDA_ENV_NAME

Build it first on a GPU interactive node:
  srun --partition=gpu-interactive --nodes=1 --gres=gpu:v100-sxm2:1 --cpus-per-task=2 --mem=10GB --time=02:00:00 --pty /bin/bash
  cd $REPO_ROOT
  bash setup_hpc_conda.sh
EOF
  exit 1
fi

if [[ "$MODE" != "parallel" && "$MODE" != "series" ]]; then
  echo "MODE must be 'parallel' or 'series'." >&2
  exit 1
fi

GROUP_LABELS=("social" "governance" "interpersonal")
GROUP_AXES=(
  "fairness,respect,inclusion,feedback,conflict_resolution"
  "accountability,integrity,ownership,trust,leadership,privacy,safety"
  "boundaries,empathy,learning"
)

filter_axes() {
  GROUP_AXES_VALUE="$1" SKIP_AXES_VALUE="$SKIP_AXES" python3 - <<'PY'
import os

group = [item.strip() for item in os.environ["GROUP_AXES_VALUE"].split(",") if item.strip()]
skip = {item.strip() for item in os.environ["SKIP_AXES_VALUE"].split(",") if item.strip()}
kept = [item for item in group if item not in skip]
print(",".join(kept))
PY
}

submit_group() {
  local label="$1"
  local axes="$2"
  local dependency="${3:-}"

  if [[ -z "$axes" ]]; then
    echo "Skipping shard '$label' because no axes remain after SKIP_AXES filtering."
    return 0
  fi

  local -a sbatch_args=(
    --parsable
    --partition="$PARTITION"
    --nodes=1
    --ntasks=1
    --gres=gpu:1
    --cpus-per-task="$CPUS"
    --mem="$MEMORY"
    --time="$TIME_LIMIT"
    --job-name="mine-${label}"
    sbatch/mine_all_axes.sbatch
  )

  if [[ -n "$dependency" ]]; then
    sbatch_args=(--dependency="afterok:$dependency" "${sbatch_args[@]}")
  fi

  local job_id
  job_id="$(
    CONDA_ENV_NAME="$CONDA_ENV_NAME" \
    MODEL_NAME="$MODEL_NAME" \
    AXES="$axes" \
    RUN_SLUG="all_axes/${label}" \
    BEAM_WIDTH="$BEAM_WIDTH" \
    TOP_K="$TOP_K" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    MAX_PHRASE_LEN="$MAX_PHRASE_LEN" \
    sbatch "${sbatch_args[@]}"
  )"

  echo "$job_id"
}

declare -a submitted=()
prev_job=""

for i in "${!GROUP_LABELS[@]}"; do
  label="${GROUP_LABELS[$i]}"
  axes="$(filter_axes "${GROUP_AXES[$i]}")"
  if [[ "$MODE" == "series" ]]; then
    job_id="$(submit_group "$label" "$axes" "$prev_job")"
    [[ -n "$job_id" ]] && prev_job="$job_id"
  else
    job_id="$(submit_group "$label" "$axes")"
  fi
  if [[ -n "${job_id:-}" ]]; then
    submitted+=("${label}:${job_id}:${axes}")
  fi
done

echo "Submission mode: $MODE"
echo "Conda env: $CONDA_ENV_NAME"
echo "Model: $MODEL_NAME"
echo "Search params: beam=$BEAM_WIDTH phrase_len=$MAX_PHRASE_LEN top_k=$TOP_K max_new_tokens=$MAX_NEW_TOKENS"
if [[ -n "$SKIP_AXES" ]]; then
  echo "Skipped axes: $SKIP_AXES"
fi
echo "Submitted jobs:"
for item in "${submitted[@]}"; do
  IFS=":" read -r label job_id axes <<<"$item"
  echo "  $label -> $job_id ($axes)"
done
echo "Watch with:"
echo "  squeue -u \$USER"
echo "  ls outputs/all_axes"
