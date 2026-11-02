#!/bin/bash
# M2 plumbing check (CPU, not a result): build a k = 14 replay spec for task 26 at batch 4, train 3
# steps through scripts/replay_train.py with checkpoints at steps 2 and 3, then check normalizer
# and the saved checkpoint list. Checkpoints are deleted afterwards (plumbing only).
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-replay-smoke
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
SPEC="$AVLA_RUNS/replay_specs/smoke-$SLURM_JOB_ID.json"
mkdir -p "$(dirname "$SPEC")"
python scripts/replay_spec.py --current 26 --k 14 --seed 0 --batch 4 --out "$SPEC"
AVLA_REPLAY_SPEC="$SPEC" AVLA_SAVE_STEPS=2 AVLA_TRAIN_ARGS="--policy.device=cpu" \
  bash scripts/train.sh - 3 3 4
ls "$AVLA_RUNS/train/$SLURM_JOB_ID/checkpoints"
rm -rf "$AVLA_RUNS/train/$SLURM_JOB_ID/checkpoints"
echo "SMOKE DONE"
