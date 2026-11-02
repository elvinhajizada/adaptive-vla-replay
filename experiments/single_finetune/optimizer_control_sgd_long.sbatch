#!/bin/bash
# E35 follow-up: no SGD LR matched fresh Adam's step-100 training loss (0.539) in job 115619 (0.03:
# 0.647; 0.1 unstable; 0.3 diverged), so compare at matched loss instead of matched step. Plain SGD
# at peak LR 0.03 (the best of the sweep) and LR 0.01, same 1k-step schedule shape, checkpoints at
# steps 100, 200, 300 (stop after 300). Slimmed for offline drift.
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-optim-sgd
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
for lr in 0.03 0.01; do
    decay=$(python -c "print($lr / 40)")
    SGD="--use_policy_training_preset=false --optimizer.type=sgd --optimizer.lr=$lr --optimizer.grad_clip_norm=10"
    SGD+=" --scheduler.type=cosine_decay_with_warmup --scheduler.num_warmup_steps=1000"
    SGD+=" --scheduler.num_decay_steps=30000 --scheduler.peak_lr=$lr --scheduler.decay_lr=$decay"
    RUN="$SLURM_JOB_ID-sgd$lr"
    AVLA_STOP_STEP=300 AVLA_TRAIN_ARGS="$SGD" SLURM_JOB_ID="$RUN" bash scripts/train.sh "[13]" 1000 100 \
        || { echo "== sgd $lr FAILED"; continue; }
    for s in 000100 000200 000300; do
        PYTHONPATH="$PWD" python scripts/slim_checkpoint.py "$AVLA_RUNS/train/$RUN/checkpoints/$s/pretrained_model" \
            || echo "== slim FAILED $RUN $s"
    done
done
grep -E "TRAIN DONE|step:(50|100|150|200|250|300) " "$AVLA_RUNS/slurm-$SLURM_JOB_ID.out" | sed 's/ data_s.*//'
echo "SGD FOLLOW-UP DONE $(date -Is)"
