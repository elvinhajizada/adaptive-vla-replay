#!/bin/bash
# Smoke test on a CPU node (validates install, offline cache, LIBERO/EGL
# and the lerobot-eval path while the GPU queue is blocked). Body: scripts/smoke.sh.
# EGL without a GPU relies on Mesa's software device; if the env reset fails here,
# that is a CPU-node rendering limit, not necessarily a pipeline bug.
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-smoke-cpu
#SBATCH --partition=<your-partition>
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
export OMP_NUM_THREADS=16
bash "${AVLA_REPO:-$SLURM_SUBMIT_DIR}/scripts/smoke.sh" cpu
