#!/bin/bash
# Probe baselines (2026-10-22): θ0 at the protocol settings (render 256,
# seed 1000, init states) on the candidate probes, before the probe design is fixed.
#   near-scene: LIBERO-10 kitchen tasks 2, 3, 8, 9 (KITCHEN_SCENE3/4/8/6), 50 episodes each
#   near-skill: LIBERO-Goal, all 10 tasks, 20 episodes each
# batch = N_EPISODES (protocol), so batch 50 for LIBERO-10: 48 CPUs (throughput sweep, job 113028).
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-probe-base
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
EVAL="${AVLA_REPO:-$SLURM_SUBMIT_DIR}/scripts/eval.sh"
# eval.sh names its output dir after SLURM_JOB_ID; give each suite its own id suffix.
SLURM_JOB_ID="$SLURM_JOB_ID-l10kitchen" bash "$EVAL" libero_10 "[2,3,8,9]" 50 || echo "libero_10 eval FAILED"
SLURM_JOB_ID="$SLURM_JOB_ID-goal" bash "$EVAL" libero_goal all 20 || echo "libero_goal eval FAILED"
