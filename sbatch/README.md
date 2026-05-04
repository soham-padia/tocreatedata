# Slurm Jobs

This folder contains starter `sbatch` files for the HPC environment.

## Typical flow on the cluster

1. Clone the repo once on the HPC.
2. Create and populate the Python environment with `sbatch sbatch/setup_env.sbatch`.
3. Run `bash update.sh` on the login node before submitting jobs.
4. Submit with `sbatch sbatch/score_pairs.sbatch`, `sbatch sbatch/mine_fairness.sbatch`, or `sbatch sbatch/mine_all_axes.sbatch`.

## Notes

- Update `#SBATCH` resource lines for your cluster.
- If your cluster uses modules, uncomment and edit the `module load` lines.
- Logs are written to `sbatch/logs/`.
- The sample GPU jobs now default to the text-only `Qwen/Qwen2.5-7B-Instruct` model for better compatibility.
- `setup_env.sbatch` wraps `setup_hpc.sh`, which creates `.venv` and installs dependencies.
- Explorer's public GPU partition allows one GPU per job; the GPU jobs here request `--partition=gpu` and `--gres=gpu:1`.
- The current scripts load `python/3.13.5` explicitly because the cluster `.venv` is built against that module.
- The batch nodes may not have outbound GitHub access, so the jobs do not run `git fetch`; update on the login node first.
- The HPC setup installs the CUDA 12.1 PyTorch wheel explicitly because Explorer's NVIDIA driver is not compatible with CUDA 13 wheels.
- If the cluster `.venv` already has the wrong torch build, run `bash fix_hpc_torch.sh` on the login node to replace it with the CUDA 12.1 wheel set.
- For the simplest overnight path from the login node, run `bash run_overnight_hpc_fairness.sh`.
- Northeastern RC recommends building the conda environment on a `gpu-interactive` node with `module load explorer anaconda3/2024.06 cuda/12.1.1`, then submitting the batch mining job from the login node with `bash run_overnight_hpc_fairness_conda.sh`.
- For all-axis mining with one command, use `bash run_all_axes_hpc_conda.sh`. It submits three predefined axis shards either in `MODE=parallel` or `MODE=series` and stores outputs under `outputs/all_axes/`.
- For mechanistic direction extraction, submit `sbatch sbatch/extract_mechanistic_directions.sbatch` with `CONDA_ENV_NAME=humanity-qwen25`. Use `LAYER_INDEX`, `AXIS`, or `AXES` to control the extraction target, and outputs will be written under `outputs/mechanistic_directions/`.
