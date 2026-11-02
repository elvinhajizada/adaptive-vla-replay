#!/bin/bash
# CPU-side checks, 2026-10-22: the dataloader worker-count benchmark (experiments/setup/dataloader_bench.py).
# The 24-worker case runs first and again last, to separate worker count from file-cache warmth;
# then three 8-worker loaders run at once, like three fine-tunes sharing a node.
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-cpu-checks
#SBATCH --partition=<your-partition>
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export PYTHONPATH=.
THETA0="$HF_HOME/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/$(python -c 'from adaptive_vla.config import Config; print(Config().pretrained_revision)')"
echo "== host $(hostname)  job $SLURM_JOB_ID  $(date -Is)"

echo "== dataloader benchmark (single loader)"
python experiments/setup/dataloader_bench.py 4 10 2>&1 | grep -q "^workers" || { echo "benchmark smoke FAILED:"; python experiments/setup/dataloader_bench.py 4 10 2>&1 | grep -v torchcodec | tail -5; exit 1; }
for w in 24 8 4 12 24; do python experiments/setup/dataloader_bench.py $w 150 2>&1 | grep "^workers"; done
echo "== three 8-worker loaders at once"
for i in 1 2 3; do python experiments/setup/dataloader_bench.py 8 150 2>&1 | grep "^workers" | sed "s/^/[$i] /" & done; wait
echo "CPU CHECKS DONE $(date -Is)"
