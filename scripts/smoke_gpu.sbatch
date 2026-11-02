#!/bin/bash
# Smoke test on a GPU node. Body: scripts/smoke.sh.
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-smoke
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
bash "${AVLA_REPO:-$SLURM_SUBMIT_DIR}/scripts/smoke.sh" cuda
