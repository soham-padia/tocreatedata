# Slurm Jobs

This folder contains the Slurm entrypoints for the Explorer HPC workflow.

The repo currently has two categories of jobs:

1. **Mechanistic jobs**
   These are the current primary path.
2. **Older behavioral / rubric jobs**
   These are still useful as baselines, but they are not the main research objective anymore.

## Explorer Workflow

Recommended Explorer flow:

1. Clone the repo once on Explorer.
2. Build the conda environment on a `gpu-interactive` node.
3. Run `bash update.sh` on the login node before submission.
4. Submit direction extraction and mining jobs from the login node.

The current Explorer environment path uses:

```bash
module load explorer anaconda3/2024.06 cuda/12.1.1
conda activate humanity-qwen25
```

## Current Mechanistic Jobs

### `extract_mechanistic_directions.sbatch`

Extracts activation-space direction vectors from `chosen` vs `rejected` pairs.

Important environment variables:

- `CONDA_ENV_NAME`
- `MODEL_NAME` default: `Qwen/Qwen2.5-7B-Instruct`
- `LAYER_INDEX` default: `-1`
- `AXIS` optional
- `AXES` optional comma-separated subset
- `OUTPUT_DIR` optional
- `RUN_SLUG` optional

Writes:

- `directions.pt`
- `summary.json`

under `outputs/mechanistic_directions/` unless `OUTPUT_DIR` is provided.

Example:

```bash
CONDA_ENV_NAME=humanity-qwen25 \
RUN_SLUG=all_axes_layer_minus1 \
LAYER_INDEX=-1 \
sbatch sbatch/extract_mechanistic_directions.sbatch
```

### `mine_pro_human_sequences_shard.sbatch`

Scores lexicon sequences by mechanistic projection and retains the top candidates for one shard.

Important environment variables:

- `CONDA_ENV_NAME`
- `MODEL_NAME`
- `DIRECTION_TENSORS`
- `DIRECTION_NAME` default: `global`
- `DIRECTION_SIGN` default: `1`
- `RUN_ROOT`
- `SHARD_INDEX`
- `NUM_SHARDS`
- `SEARCH_MODE` default: `exhaustive`
- `MIN_PHRASE_LEN`
- `MAX_PHRASE_LEN`
- `BATCH_SIZE`
- `RETAIN_TOP_K`
- `BEAM_WIDTH` for `SEARCH_MODE=beam`

`SEARCH_MODE` choices:

- `exhaustive`
  Short exact search over the controlled lexicon.
- `beam`
  Heuristic expansion for longer phrase-level searches, such as length 15.

Important:

- phrase length here is in **lexicon units**, not raw tokenizer tokens
- `DIRECTION_SIGN=-1` means opposite projection against the saved direction vector

### `merge_mechanistic_sequences.sbatch`

Merges shard outputs into a final `top_sentences.jsonl`.

Important environment variables:

- `CONDA_ENV_NAME`
- `RUN_ROOT`
- `FINAL_TOP_K`

Note:

- On Explorer, this merge job still requests `--gres=gpu:1` because the public `gpu` partition rejects jobs that do not match its access policy.
- The merge itself is CPU-light; the GPU request is a cluster-policy workaround, not a model requirement.

### `run_pro_human_mechanistic_hpc.sh`

This is the main "submit and forget" launcher.

What it does:

1. optionally extracts the direction if `DIRECTION_TENSORS` is not supplied
2. submits one or more shard jobs
3. submits a merge job with `afterok` dependency

Important environment variables:

- `CONDA_ENV_NAME`
- `RUN_UPDATE=1` optional
- `RUN_SLUG`
- `DIRECTION_TENSORS` optional
- `DIRECTION_NAME`
- `DIRECTION_SIGN`
- `LAYER_INDEX`
- `NUM_SHARDS`
- `SEARCH_MODE`
- `MIN_PHRASE_LEN`
- `MAX_PHRASE_LEN`
- `BATCH_SIZE`
- `RETAIN_TOP_K`
- `FINAL_TOP_K`
- `BEAM_WIDTH`
- `EXTRACT_TIME`
- `SHARD_TIME`
- `MERGE_TIME`

Smoke-test example:

```bash
CONDA_ENV_NAME=humanity-qwen25 \
RUN_UPDATE=1 \
RUN_SLUG=pro_human_smoke \
NUM_SHARDS=1 \
MIN_PHRASE_LEN=1 \
MAX_PHRASE_LEN=3 \
BATCH_SIZE=8 \
RETAIN_TOP_K=4000 \
FINAL_TOP_K=1000 \
SHARD_TIME=01:00:00 \
MERGE_TIME=00:20:00 \
bash run_pro_human_mechanistic_hpc.sh
```

Heuristic long-sequence example:

```bash
CONDA_ENV_NAME=humanity-qwen25 \
DIRECTION_TENSORS=outputs/mechanistic_dataset/pro_human_smoke_.../direction/directions.pt \
RUN_SLUG=pro_human_beam15 \
DIRECTION_SIGN=1 \
SEARCH_MODE=beam \
NUM_SHARDS=1 \
MIN_PHRASE_LEN=1 \
MAX_PHRASE_LEN=15 \
BEAM_WIDTH=64 \
BATCH_SIZE=32 \
RETAIN_TOP_K=5000 \
FINAL_TOP_K=1000 \
SHARD_TIME=01:00:00 \
MERGE_TIME=00:20:00 \
bash run_pro_human_mechanistic_hpc.sh
```

Opposite-projection example:

```bash
CONDA_ENV_NAME=humanity-qwen25 \
DIRECTION_TENSORS=outputs/mechanistic_dataset/pro_human_smoke_.../direction/directions.pt \
RUN_SLUG=negative_projection_beam15 \
DIRECTION_SIGN=-1 \
SEARCH_MODE=beam \
NUM_SHARDS=1 \
MIN_PHRASE_LEN=1 \
MAX_PHRASE_LEN=15 \
BEAM_WIDTH=64 \
BATCH_SIZE=32 \
RETAIN_TOP_K=5000 \
FINAL_TOP_K=1000 \
SHARD_TIME=01:00:00 \
MERGE_TIME=00:20:00 \
bash run_pro_human_mechanistic_hpc.sh
```

## Older Baseline Jobs

These still exist and can be used for comparison:

- `mine_fairness.sbatch`
- `mine_all_axes.sbatch`
- `mine_mechanistic_dataset.sbatch`
- `score_pairs.sbatch`
- `setup_env.sbatch`

The older convenience launchers are:

- `bash run_overnight_hpc_fairness.sh`
- `bash run_overnight_hpc_fairness_conda.sh`
- `bash run_all_axes_hpc_conda.sh`

Those are useful if you want the previous behavioral/rubric baselines, but they are not the main mechanistic workflow.

## Notes

- Logs are written to `sbatch/logs/`.
- The public Explorer `gpu` partition requires `--gres=gpu:1` for these jobs.
- Batch nodes may not have outbound GitHub access; update on the login node first.
- The launcher is usually safer with `NUM_SHARDS=1` or `2` on Explorer because of per-user submit limits.
- If a long search is timing out, reduce:
  - `NUM_SHARDS`
  - `MAX_PHRASE_LEN`
  - `BEAM_WIDTH`
  - `BATCH_SIZE`
- If you only want a smoke test, do not jump straight to `MAX_PHRASE_LEN=15`. Use the short exhaustive path first.
