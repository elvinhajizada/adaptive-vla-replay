#!/bin/bash
# Eval diagnostic on a CPU node: do the eval's observations and
# θ0's predictions on them match the training demos? Open-loop only, no rollouts, not a
# result. Body: experiments/setup/eval_parity.py (all 10 Spatial tasks, 3 init states each).
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-parity-cpu
#SBATCH --partition=<your-partition>
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export OMP_NUM_THREADS=16
echo "== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is)"
PYTHONPATH=. python experiments/setup/eval_parity.py "$@"
