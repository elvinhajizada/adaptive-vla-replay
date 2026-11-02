#!/bin/bash
# Is the task-23 forgetting scene-local or global? (2026-10-22.)
# The 1k task-23 checkpoint (job 113795) drove the scene-4 near probe from 42/50 to 0/50
# (job 113800). Here the same checkpoint on the far probe (LIBERO-Spatial, different scene) and
# the near-skill probe (LIBERO-Goal, shared objects in another scene), 10 tasks x 10 episodes each
# at the protocol settings, with subgoal logging. θ0 references: Spatial 76/100 (job 113024, same
# 10 x 10), Goal first 10 episodes per task of job 113038 (same seeds and init states).
# The checkpoint is slim (delta vs θ0), so it is materialized first and the copy removed after.
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-far-probe
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
SLIM="$AVLA_RUNS/train/113795/checkpoints/last/pretrained_model"
FULL="$AVLA_RUNS/materialized/$SLURM_JOB_ID-ft23-1k"
PYTHONPATH="$PWD" python scripts/slim_checkpoint.py --materialize "$SLIM" "$FULL" || exit 1
export AVLA_SUBGOALS=1
SLURM_JOB_ID="$SLURM_JOB_ID-ft23-1k" bash scripts/eval.sh libero_spatial all 10 "$FULL" || echo "== spatial eval FAILED"
SLURM_JOB_ID="$SLURM_JOB_ID-ft23-1k" bash scripts/eval.sh libero_goal all 10 "$FULL" || echo "== goal eval FAILED"
rm -rf "$FULL"
echo "FAR-PROBE CHECK DONE $(date -Is)"
