#!/bin/bash
# M1 eval diagnostic on a CPU node: replay demo actions in the eval
# env and compare the EEF path with the demo's (control-side parity). No policy, not a result.
# Body: experiments/m1/replay_check.py (all 10 Spatial tasks, 3 init states each).
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-replay-cpu
#SBATCH --partition=<your-partition>
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export OMP_NUM_THREADS=8
echo "== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is)"
PYTHONPATH=. python experiments/m1/replay_check.py "$@"
