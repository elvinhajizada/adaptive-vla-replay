#!/bin/bash
# E44 (CPU): is a small replay buffer memorized? θ0 and one replay run's checkpoint on
# its own buffered vs unbuffered θ0 frames and its current task (experiments/m2/buffer_memorization.py,
# then scripts/offline_action_error.py). Output: $AVLA_RUNS/memorization/<LABEL>/table.csv.
# Usage: sbatch experiments/m2/buffer_memorization.sbatch LABEL SPEC POLICY_DIR
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-memo
#SBATCH --partition=<your-partition>
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
LABEL="${1:?usage: buffer_memorization.sbatch LABEL SPEC POLICY_DIR}"; SPEC="${2:?SPEC}"; POLICY="${3:?POLICY_DIR}"
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export PYTHONPATH="$PWD" OMP_NUM_THREADS=32
echo "== host $(hostname)  job $SLURM_JOB_ID  $(date -Is)"
OUT="$AVLA_RUNS/memorization/$LABEL"
python experiments/m2/buffer_memorization.py "$SPEC" "$OUT"
python scripts/offline_action_error.py --only-frames --csv="$OUT/table.csv" \
    --frames=buffered="$OUT/buffered.json" --frames=unbuffered="$OUT/unbuffered.json" \
    --frames=current="$OUT/current.json" "$LABEL=$POLICY"
