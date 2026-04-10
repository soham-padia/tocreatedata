# Slurm Jobs

This folder contains starter `sbatch` files for the HPC environment.

## Typical flow on the cluster

1. Clone the repo once on the HPC.
2. Create and populate the Python environment there.
3. Run `bash update.sh` before submitting jobs, or let the job do it.
4. Submit with `sbatch sbatch/mine_fairness.sbatch`.

## Notes

- Update `#SBATCH` resource lines for your cluster.
- If your cluster uses modules, uncomment and edit the `module load` lines.
- Logs are written to `sbatch/logs/`.
- The sample job defaults to the `fairness` axis and `Qwen/Qwen3.5-9B`.
