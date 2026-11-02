#!/bin/bash
# E35: is the early drift of a fine-tune an artefact of a fresh Adam? Task 23 (dataset index 13), θ0's data, seed
# and 1k-step schedule shape (auto-scaled warmup 33, cosine to step 1000), each run stopped after its
# step-100 checkpoint (scripts/optim_train.py, AVLA_STOP_STEP):
#   sgd<LR>  plain SGD (no momentum), peak LR swept 0.03 to 3, same schedule shape and clip norm
#   warm     AdamW (θ0's recipe) with moments warmed by 200 zero-LR updates on θ0's four suites
# Fresh-Adam reference: the onset run 113822 at step 100 (same data, seed and schedule).
# Checkpoints are slimmed; drift is measured on CPU by experiments/m1/gripper_forensics.py.
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-optim-ctrl
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
slim() { PYTHONPATH="$PWD" python scripts/slim_checkpoint.py "$AVLA_RUNS/train/$1/checkpoints/000100/pretrained_model" \
    || echo "== slim FAILED $1"; }
AVLA_WARM_ADAM=200 AVLA_STOP_STEP=100 SLURM_JOB_ID="$SLURM_JOB_ID-warm" bash scripts/train.sh "[13]" 1000 100 \
    && slim "$SLURM_JOB_ID-warm" || echo "== warm FAILED"
for lr in 0.03 0.1 0.3 1 3; do
    decay=$(python -c "print($lr / 40)")
    SGD="--use_policy_training_preset=false --optimizer.type=sgd --optimizer.lr=$lr --optimizer.grad_clip_norm=10"
    SGD+=" --scheduler.type=cosine_decay_with_warmup --scheduler.num_warmup_steps=1000"
    SGD+=" --scheduler.num_decay_steps=30000 --scheduler.peak_lr=$lr --scheduler.decay_lr=$decay"
    AVLA_STOP_STEP=100 AVLA_TRAIN_ARGS="$SGD" SLURM_JOB_ID="$SLURM_JOB_ID-sgd$lr" bash scripts/train.sh "[13]" 1000 100 \
        && slim "$SLURM_JOB_ID-sgd$lr" || echo "== sgd $lr FAILED"
done
grep -E "TRAIN DONE|step:(50|100) " "$AVLA_RUNS/slurm-$SLURM_JOB_ID.out" | sed 's/ data_s.*//'
echo "OPTIMIZER CONTROL DONE $(date -Is)"
