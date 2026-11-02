#!/bin/bash
# M1 task choice: theta0 zero-shot on the 11 single-scene kitchen candidates (LIBERO-90 env ids,
# E11), 10 episodes each, protocol settings (render 256, seed 1000, init states).
# Keep a task only if zero-shot < 30% (task-choice rule). Tasks run sequentially so the
# run stays deterministic. About 110 episodes.
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-zeroshot
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=slurm-%j.out
bash "${AVLA_REPO:-$SLURM_SUBMIT_DIR}/scripts/eval.sh" libero_90 "[18,19,22,23,24,26,27,33,34,38,39]" 10
