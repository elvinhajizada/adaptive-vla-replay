#!/bin/bash
# Gripper forensics on CPU, E26. Body: experiments/single_finetune/gripper_forensics.py.
# Usage: sbatch experiments/single_finetune/gripper_forensics.sbatch SUBCOMMAND [ARGS...]   (measure | patch | grad | report)
# Several commands in one job: separate them with ' ;; ' as one quoted argument, e.g.
#   sbatch experiments/single_finetune/gripper_forensics.sbatch "measure --hooks ft1k=DIR ;; report"
# AVLA_FORENSICS_SMOKE=1 runs on 2 tasks per suite (outputs go to forensics-smoke/).
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-grip-forensics
#SBATCH --partition=<your-partition>
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export PYTHONPATH="$PWD" OMP_NUM_THREADS=32
echo "== host $(hostname)  job $SLURM_JOB_ID  $(date -Is)  smoke=${AVLA_FORENSICS_SMOKE:-0}"
IFS=$'\n' read -r -d '' -a STEPS < <(printf '%s' "$*" | sed 's/ ;; /\n/g'; printf '\0') || true
for s in "${STEPS[@]}"; do
    echo "== $s  $(date -Is)"
    # shellcheck disable=SC2086
    python experiments/single_finetune/gripper_forensics.py $s
done
echo "FORENSICS JOB DONE $(date -Is)"
