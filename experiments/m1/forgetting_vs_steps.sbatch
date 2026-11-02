#!/bin/bash
# Step-count follow-up (2026-10-22): does a longer fine-tune on task 23 cost more old-skill
# retention? Evaluates the 1k/2k/3k task-23 checkpoints (jobs 113795-113797) on the scene-4
# near-scene probe, LIBERO-10 task 3 "put the black bowl in the bottom drawer and close it",
# 50 episodes each at the protocol settings; θ0 scored 42/50 there with the same seed and init
# states (job 113038), so each comparison is paired. Then weight drift vs θ0 for all three, and
# finally slims the checkpoints (delta vs θ0 only, training_state dropped; lossless, verified).
# Submit with --dependency=afterany:113797.
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-forget-steps
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export PYTHONPATH=.
THETA0="$HF_HOME/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/$(python -c 'from adaptive_vla.config import Config; print(Config().pretrained_revision)')"
declare -A JOB=([1k]=113795 [2k]=113796 [3k]=113797)
CKPTS=()
for s in 1k 2k 3k; do
    p="$AVLA_RUNS/train/${JOB[$s]}/checkpoints/last/pretrained_model"
    if [ ! -f "$p/model.safetensors" ]; then echo "== $s: no full checkpoint at $p, skipped"; continue; fi
    CKPTS+=("$s=$p")
    SLURM_JOB_ID="$SLURM_JOB_ID-ft23-$s" bash scripts/eval.sh libero_10 "[3]" 50 "$p" || echo "== $s eval FAILED"
done
echo "== weight drift"
python scripts/weight_drift.py "$THETA0" "${CKPTS[@]}" | grep -v '^{'
echo "== slim checkpoints"
for c in "${CKPTS[@]}"; do python scripts/slim_checkpoint.py "${c#*=}" || echo "== slim FAILED for ${c%%=*} (checkpoint kept)"; done
echo "FORGET-VS-STEPS DONE $(date -Is)"
