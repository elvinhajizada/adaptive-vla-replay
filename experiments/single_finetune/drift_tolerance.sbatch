#!/bin/bash
# Drift-tolerance curve: θ0 interpolated with the task-23
# 1k checkpoint (113795) at alpha = 0.1, 0.2, 0.3, 0.5, 0.7; each on the scene-4 LIBERO-10 task x 20
# and task 23 x 20 (concurrently on one GPU), protocol settings, subgoal logging. alpha = 0 is θ0
# (E29), alpha = 1 the checkpoint itself (113800, 113795). Interpolated slim checkpoints are kept
# under $AVLA_RUNS/interp/113795/ for the offline drift (CPU) and as a weight-interpolation baseline.
# Usage: sbatch experiments/single_finetune/drift_tolerance.sbatch [ALPHA ...]   (default 0.1 0.2 0.3 0.5 0.7;
#   1.0 is the pipeline control: it must reproduce 113795's own task-23 success, 27/50)
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-drift-tol
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G   # 40 simulator processes ~102 GB RSS (memguard, 2026-10-26)
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export PYTHONPATH="$PWD" AVLA_SUBGOALS=1
SRC="$AVLA_RUNS/train/113795/checkpoints/last/pretrained_model"
ROOT="$AVLA_RUNS/interp/113795"
ALPHAS=("$@"); [ ${#ALPHAS[@]} -gt 0 ] || ALPHAS=(0.1 0.2 0.3 0.5 0.7)   # args override (115614 ran the default)
python scripts/interpolate_checkpoint.py "$SRC" "$ROOT" "${ALPHAS[@]}" || exit 1
for a in "${ALPHAS[@]}"; do
    FULL="$AVLA_RUNS/materialized/$SLURM_JOB_ID-a$a"
    python scripts/slim_checkpoint.py --materialize "$ROOT/a$a" "$FULL" || { echo "== a$a materialize FAILED"; continue; }
    ID="$SLURM_JOB_ID-t23-a$a"
    SLURM_JOB_ID="$ID" bash scripts/eval.sh libero_10 "[3]" 20 "$FULL" > "$ROOT/eval-a$a-l10.log" 2>&1 &
    p1=$!
    SLURM_JOB_ID="$ID" bash scripts/eval.sh libero_90 "[23]" 20 "$FULL" > "$ROOT/eval-a$a-l90.log" 2>&1 &
    p2=$!
    wait $p1 || echo "== a$a scene-4 probe FAILED"
    wait $p2 || echo "== a$a task 23 FAILED"
    grep -h "Success rate\|overall" "$ROOT"/eval-a$a-*.log | tail -2 || true
    rm -rf "$FULL"
done
echo "DRIFT TOLERANCE DONE $(date -Is)"
