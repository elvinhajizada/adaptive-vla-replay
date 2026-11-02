#!/bin/bash
# Convention checks on LIBERO-90, on a CPU node: image orientation of the
# libero_90 env vs the converted demos, and the fps label / action-chunk timing (θ0 vs our data).
# Open-loop only, no rollouts, not a result. Body: experiments/setup/libero90_parity.py (5 new tasks, 3 inits).
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-l90-parity
#SBATCH --partition=<your-partition>
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export OMP_NUM_THREADS=16
echo "== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is)"
PYTHONPATH=. python experiments/setup/libero90_parity.py "$@"
