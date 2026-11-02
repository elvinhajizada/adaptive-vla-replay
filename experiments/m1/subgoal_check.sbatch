#!/bin/bash
# One-time check of the subgoal-logging eval path (AVLA_SUBGOALS=1, adaptive_vla/subgoal_env.py)
# before it is used for results: θ0 on LIBERO-10 task 3 (scene-4 near probe), 50 episodes at the
# protocol settings. Must reproduce job 113038 (42/50, same seed and init states) up to GPU
# nondeterminism, and gives the first per-goal breakdown of the probe.
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-subgoal-chk
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
AVLA_SUBGOALS=1 bash "${AVLA_REPO:-$SLURM_SUBMIT_DIR}/scripts/eval.sh" libero_10 "[3]" 50
