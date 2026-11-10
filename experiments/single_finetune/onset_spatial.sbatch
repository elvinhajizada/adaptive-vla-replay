#!/bin/bash
# Closed-loop onset on Spatial: how fast does the task-23 fine-tune erase θ0's Spatial success?
# The onset run (forgetting_onset.sbatch, job 113822) kept a checkpoint every 100 steps; the scene-4
# probe was evaluated at steps 100 to 700, Spatial only offline. This evaluates Spatial 10 x 10 at
# steps 100, 200, 400, 700 and 1000 (the run's end point), all five at once on the one GPU (eval.sh
# runs a suite's tasks one after another and leaves the GPU mostly idle). 50 simulator processes,
# about 2.4 GB RSS each: within AVLA_MAX_ENVS (64).
# Outputs: $AVLA_RUNS/eval/libero_spatial/<job>-s<step>-r256/, logs $AVLA_RUNS/eval/<job>-s<step>.log.
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-onset-spatial
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G   # 50 simulator processes x about 2.4 GB RSS (memguard, 2026-10-26)
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
RUN="$AVLA_RUNS/train/${AVLA_ONSET_RUN:-113822}/checkpoints"
export AVLA_SUBGOALS=1
pids=(); full=()
for s in 000100 000200 000400 000700 001000; do
    FULL="$AVLA_RUNS/materialized/$SLURM_JOB_ID-s$s"
    PYTHONPATH="$PWD" python scripts/slim_checkpoint.py --materialize "$RUN/$s/pretrained_model" "$FULL" || exit 1
    full+=("$FULL")
    LOG="$AVLA_RUNS/eval/$SLURM_JOB_ID-s$s.log"
    SLURM_JOB_ID="$SLURM_JOB_ID-s$s" bash scripts/eval.sh libero_spatial all 10 "$FULL" > "$LOG" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "== an eval FAILED (see $AVLA_RUNS/eval/$SLURM_JOB_ID-s*.log)"; done
grep -H "overall:" "$AVLA_RUNS/eval/$SLURM_JOB_ID"-s*.log
rm -rf "${full[@]}"
echo "ONSET SPATIAL DONE $(date -Is)"
