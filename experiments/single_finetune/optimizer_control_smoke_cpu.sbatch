#!/bin/bash
# E35 plumbing check (CPU, not a result): task 23 at batch 4, 5-step schedule, stopped after the
# step-2 checkpoint, once with Adam warmed by 2 zero-LR updates on θ0's data (scripts/optim_train.py)
# and once with plain SGD through AVLA_TRAIN_ARGS. Checkpoints are deleted afterwards.
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-optim-smoke
#SBATCH --partition=<your-partition>
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export PYTHONPATH=$PWD
SGD="--use_policy_training_preset=false --optimizer.type=sgd --optimizer.lr=0.1 --optimizer.grad_clip_norm=10"
    SGD+=" --scheduler.type=cosine_decay_with_warmup --scheduler.num_warmup_steps=1000"
    SGD+=" --scheduler.num_decay_steps=30000 --scheduler.peak_lr=0.1 --scheduler.decay_lr=0.0025"
AVLA_WARM_ADAM=2 AVLA_STOP_STEP=2 AVLA_TRAIN_ARGS="--policy.device=cpu" SLURM_JOB_ID="$SLURM_JOB_ID-warm" \
  bash scripts/train.sh "[13]" 5 2 4
AVLA_STOP_STEP=2 AVLA_TRAIN_ARGS="--policy.device=cpu $SGD" SLURM_JOB_ID="$SLURM_JOB_ID-sgd" \
  bash scripts/train.sh "[13]" 5 2 4
for r in warm sgd; do
  ls "$AVLA_RUNS/train/$SLURM_JOB_ID-$r/checkpoints"
  grep -o '"type": "[a-z_]*"' "$AVLA_RUNS/train/$SLURM_JOB_ID-$r/checkpoints/last/pretrained_model/train_config.json" | head -3
  rm -rf "$AVLA_RUNS/train/$SLURM_JOB_ID-$r/checkpoints"
done
echo "SMOKE DONE"
