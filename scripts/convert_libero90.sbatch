#!/bin/bash
# Data conversion: convert nvidia LIBERO-90 to θ0 conventions + verify. CPU only.
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-convert
#SBATCH --partition=<your-partition>
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
python scripts/convert_libero90.py "$@"
