#!/bin/bash
# Reference eval: θ0 on LIBERO-Spatial, 10 tasks x 10 episodes.
# Target: within about 10 pts of the 80% community figure. Body: scripts/eval.sh.
# batch_size=10 runs 10 envs in parallel (async subprocesses), hence 12 CPUs.
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-eval-spatial
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
bash "${AVLA_REPO:-$SLURM_SUBMIT_DIR}/scripts/eval.sh" libero_spatial all 10
