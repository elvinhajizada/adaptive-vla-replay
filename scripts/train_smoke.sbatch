#!/bin/bash
# Training smoke test: short fine-tune of θ0 on one LIBERO-90 task to check (a) the training
# pipeline end-to-end, (b) that saved checkpoints keep θ0's normalizer stats, and
# (c) training throughput at batch 64. Plumbing only: task 0 is arbitrary (not one of the
# 5 chosen tasks) and the checkpoint is not evaluated.
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-train-smoke
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
bash "${AVLA_REPO:-$SLURM_SUBMIT_DIR}/scripts/train.sh" "[0]" 300 150 64
