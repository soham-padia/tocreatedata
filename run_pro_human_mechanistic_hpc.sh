#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-humanity-qwen25}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
RUN_UPDATE="${RUN_UPDATE:-0}"
RUN_SLUG="${RUN_SLUG:-pro_human_global}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
NUM_SHARDS="${NUM_SHARDS:-5}"
MIN_PHRASE_LEN="${MIN_PHRASE_LEN:-1}"
MAX_PHRASE_LEN="${MAX_PHRASE_LEN:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
RETAIN_TOP_K="${RETAIN_TOP_K:-5000}"
FINAL_TOP_K="${FINAL_TOP_K:-1000}"
DIRECTION_NAME="${DIRECTION_NAME:-global}"
LAYER_INDEX="${LAYER_INDEX:--1}"
DIRECTION_TENSORS="${DIRECTION_TENSORS:-}"
EXTRACT_TIME="${EXTRACT_TIME:-01:00:00}"
SHARD_TIME="${SHARD_TIME:-01:00:00}"
MERGE_TIME="${MERGE_TIME:-00:20:00}"

cd "$REPO_ROOT"

mkdir -p sbatch/logs outputs/mechanistic_dataset

module purge
module load explorer anaconda3/2024.06

if [[ "$RUN_UPDATE" == "1" ]]; then
  bash ./update.sh
fi

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  echo "Missing conda env: $CONDA_ENV_NAME" >&2
  exit 1
fi

RUN_ROOT="outputs/mechanistic_dataset/${RUN_SLUG}_${RUN_TAG}"
mkdir -p "$RUN_ROOT"

extract_job=""
if [[ -z "$DIRECTION_TENSORS" ]]; then
  DIRECTION_OUTPUT_DIR="$RUN_ROOT/direction"
  DIRECTION_TENSORS="$DIRECTION_OUTPUT_DIR/directions.pt"
  extract_job="$(
    CONDA_ENV_NAME="$CONDA_ENV_NAME" \
    MODEL_NAME="$MODEL_NAME" \
    LAYER_INDEX="$LAYER_INDEX" \
    OUTPUT_DIR="$DIRECTION_OUTPUT_DIR" \
    sbatch --parsable --job-name=extract-pro-human --time="$EXTRACT_TIME" sbatch/extract_mechanistic_directions.sbatch
  )"
fi

declare -a shard_jobs=()
for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  submit_args=(--parsable "--job-name=mech-shard-${shard}" "--time=${SHARD_TIME}")
  if [[ -n "$extract_job" ]]; then
    submit_args+=(--dependency="afterok:${extract_job}")
  fi
  job_id="$(
    CONDA_ENV_NAME="$CONDA_ENV_NAME" \
    MODEL_NAME="$MODEL_NAME" \
    DIRECTION_TENSORS="$DIRECTION_TENSORS" \
    DIRECTION_NAME="$DIRECTION_NAME" \
    RUN_ROOT="$RUN_ROOT" \
    SHARD_INDEX="$shard" \
    NUM_SHARDS="$NUM_SHARDS" \
    MIN_PHRASE_LEN="$MIN_PHRASE_LEN" \
    MAX_PHRASE_LEN="$MAX_PHRASE_LEN" \
    BATCH_SIZE="$BATCH_SIZE" \
    RETAIN_TOP_K="$RETAIN_TOP_K" \
    sbatch "${submit_args[@]}" sbatch/mine_pro_human_sequences_shard.sbatch
  )"
  shard_jobs+=("$job_id")
done

dependency="$(IFS=:; echo "${shard_jobs[*]}")"
merge_job="$(
  CONDA_ENV_NAME="$CONDA_ENV_NAME" \
  RUN_ROOT="$RUN_ROOT" \
  FINAL_TOP_K="$FINAL_TOP_K" \
  sbatch --parsable --job-name=mech-merge --time="$MERGE_TIME" --dependency="afterok:${dependency}" sbatch/merge_mechanistic_sequences.sbatch
)"

echo "Submitted pro-human mechanistic run"
echo "Run root: $RUN_ROOT"
echo "Direction tensors: $DIRECTION_TENSORS"
echo "Direction name: $DIRECTION_NAME"
echo "Times: extract=$EXTRACT_TIME shards=$SHARD_TIME merge=$MERGE_TIME"
if [[ -n "$extract_job" ]]; then
  echo "Direction extraction job:"
  echo "  $extract_job"
fi
echo "Shards:"
for job_id in "${shard_jobs[@]}"; do
  echo "  $job_id"
done
echo "Merge job:"
echo "  $merge_job"
echo "Watch with:"
echo "  squeue -u \$USER"
echo "  ls $RUN_ROOT"
