#!/bin/bash
# Single-task fine-tune and check: fine-tune θ0 on one LIBERO-90 task, then evaluate it (50 episodes).
# Usage: sbatch scripts/finetune_eval.sbatch ENV_ID STEPS [N_EPISODES]. Body: scripts/finetune_eval.sh.
# 24 CPUs: dataloader workers for training, then 50 async envs for the batch-50 eval; three of
# these fit next to one other job on the GPU node's 96 CPUs.
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-ft-eval
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G   # about 2.4 GB RSS per simulator process; 50-episode batch ~122 GB (memguard, 2026-10-26)
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
bash "${AVLA_REPO:-$SLURM_SUBMIT_DIR}/scripts/finetune_eval.sh" "$@"
