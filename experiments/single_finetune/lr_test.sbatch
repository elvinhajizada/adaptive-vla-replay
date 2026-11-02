#!/bin/bash
# Does a 10x smaller update learn task 23 with less forgetting? (2026-10-22.)
# Task 23 at the protocol length (8 passes) with peak LR 1e-5 instead of θ0's 1e-4 (final LR
# 2.5e-7, same schedule shape; train.sh AVLA_LR). Evaluated on task 23 (50 eps, as the 1e-4 run:
# 27/50 at 1k steps), the scene-4 near probe (LIBERO-10 task 3, 50 eps; θ0 42/50, 1e-4: 0/50)
# and LIBERO-Spatial 10 x 10 (θ0 76/100, 1e-4: 0/100), with subgoal logging.
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-lr1e-5
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export AVLA_LR=1e-5 AVLA_SUBGOALS=1
bash scripts/finetune_eval.sh 23 8p 50 || exit 1        # trains, evals task 23, slims
SLIM="$AVLA_RUNS/train/$SLURM_JOB_ID/checkpoints/last/pretrained_model"
FULL="$AVLA_RUNS/materialized/$SLURM_JOB_ID-lr1e-5"
PYTHONPATH="$PWD" python scripts/slim_checkpoint.py --materialize "$SLIM" "$FULL" || exit 1
SLURM_JOB_ID="$SLURM_JOB_ID-lr1e-5" bash scripts/eval.sh libero_10 "[3]" 50 "$FULL" || echo "== probe eval FAILED"
SLURM_JOB_ID="$SLURM_JOB_ID-lr1e-5" bash scripts/eval.sh libero_spatial all 10 "$FULL" || echo "== spatial eval FAILED"
rm -rf "$FULL"
echo "LR TEST DONE $(date -Is)"
