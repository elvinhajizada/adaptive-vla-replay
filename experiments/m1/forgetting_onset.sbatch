#!/bin/bash
# How fast does the task-23 fine-tune erase θ0's skills? (2026-10-22.)
# Retrains the 1k-step task-23 run of job 113795 (same data, seed, recipe; 7.9 passes) with a
# checkpoint every 100 steps. Checkpoints 100/200/400/700 are evaluated on the scene-4 near probe
# (LIBERO-10 task 3; θ0 42/50, 1k 0/50) and on task 23 itself (θ0 0/10, 1k 27/50), 20 episodes
# each with subgoal logging. Every checkpoint is then slimmed; the offline action-error check
# (scripts/offline_action_error.py) covers all of them for a graded Spatial/Goal curve.
# Note: these checkpoints sit inside one 1k schedule (LR still high at step 100 to 700).
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-onset
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
bash scripts/train.sh "[13]" 1000 100 || exit 1
RUN="$AVLA_RUNS/train/$SLURM_JOB_ID/checkpoints"
export AVLA_SUBGOALS=1
for s in 000100 000200 000400 000700; do
    P="$RUN/$s/pretrained_model"
    SLURM_JOB_ID="$SLURM_JOB_ID-s$s" bash scripts/eval.sh libero_10 "[3]" 20 "$P" || echo "== $s probe eval FAILED"
    SLURM_JOB_ID="$SLURM_JOB_ID-s$s" bash scripts/eval.sh libero_90 "[23]" 20 "$P" || echo "== $s task-23 eval FAILED"
done
for d in "$RUN"/0*/pretrained_model; do PYTHONPATH="$PWD" python scripts/slim_checkpoint.py "$d" || echo "== slim FAILED $d"; done
echo "ONSET DONE $(date -Is)"
