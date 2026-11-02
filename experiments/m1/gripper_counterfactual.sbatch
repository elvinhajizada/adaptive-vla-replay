#!/bin/bash
# E26 counterfactuals: retrain the 1k-step task-23 fine-tune (job 113795: same data, seed, recipe)
# twice with part of the action loss masked: AVLA_LOSS_DIMS=arm (gripper unsupervised) and
# AVLA_LOSS_DIMS=grip (only the gripper supervised). No closed-loop eval; the checkpoints are
# slimmed and measured offline by experiments/m1/gripper_forensics.py.
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-grip-cf
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
for dims in arm grip; do
    AVLA_LOSS_DIMS=$dims SLURM_JOB_ID="$SLURM_JOB_ID-$dims" bash scripts/train.sh "[13]" 1000 || { echo "== $dims FAILED"; continue; }
    PYTHONPATH="$PWD" python scripts/slim_checkpoint.py "$AVLA_RUNS/train/$SLURM_JOB_ID-$dims/checkpoints/last/pretrained_model" \
        || echo "== slim FAILED $dims"
done
echo "COUNTERFACTUAL DONE $(date -Is)"
