#!/bin/bash
# M1 reference-eval follow-up: theta0 on LIBERO-Spatial rendered at 256x256 (theta0's training-frame
# size) instead of the env default 360x360. Run because the 360 eval (job 113009) gave 64%,
# more than 10 pts below the 80% target (pre-agreed check, 2026-10-19).
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-eval-sp256
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
export AVLA_RENDER=256
bash "${AVLA_REPO:-$SLURM_SUBMIT_DIR}/scripts/eval.sh" libero_spatial all 10
