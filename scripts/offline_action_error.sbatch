#!/bin/bash
# Graded forgetting on CPU: predicted-action change of fine-tuned checkpoints on old-task
# demo frames, vs θ0 and vs the demos. Body: scripts/offline_action_error.py.
# Usage: sbatch scripts/offline_action_error.sbatch LABEL=POLICY_DIR [...]
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-offline-err
#SBATCH --partition=<your-partition>
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export PYTHONPATH="$PWD" OMP_NUM_THREADS=32
echo "== host $(hostname)  job $SLURM_JOB_ID  $(date -Is)"
python scripts/offline_action_error.py "$@"
