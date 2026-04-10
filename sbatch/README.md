# Slurm Jobs

This folder contains starter `sbatch` files for the HPC environment.

## Typical flow on the cluster

1. Clone the repo once on the HPC.
2. Create and populate the Python environment with `sbatch sbatch/setup_env.sbatch`.
3. Run `bash update.sh` before submitting jobs, or let the job do it.
4. Submit with `sbatch sbatch/score_pairs.sbatch`, `sbatch sbatch/mine_fairness.sbatch`, or `sbatch sbatch/mine_all_axes.sbatch`.

## Notes

- Update `#SBATCH` resource lines for your cluster.
- If your cluster uses modules, uncomment and edit the `module load` lines.
- Logs are written to `sbatch/logs/`.
- The sample job defaults to the `fairness` axis and `Qwen/Qwen3.5-9B`.
- `setup_env.sbatch` wraps `setup_hpc.sh`, which creates `.venv` and installs dependencies.
- Explorer's public GPU partition allows one GPU per job; the GPU jobs here request `--partition=gpu` and `--gres=gpu:1`.
