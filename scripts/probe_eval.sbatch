#!/bin/bash
# Closed-loop retention of a (slim) fine-tuned checkpoint on old-skill probes, with subgoal logging.
# Materializes the checkpoint, runs each probe through eval.sh (protocol settings), removes the copy.
# Usage: sbatch [--job-name=...] scripts/probe_eval.sbatch POLICY_DIR LABEL SUITE:TASK_IDS:N_EPISODES [...]
#   e.g. ... train/113804/checkpoints/last/pretrained_model ft26 libero_10:[3]:50 libero_spatial:all:10
#   AVLA_PROBE_PARALLEL=1 runs the specs concurrently on the one GPU (eval.sh runs a suite's tasks
#   one after another, so this recovers most of the GPU), logs in $AVLA_RUNS/eval/<job>-<label>-<suite>.log;
#   only while the specs' episodes sum to at most AVLA_MAX_ENVS (default 64), else sequentially
# Outputs: $AVLA_RUNS/eval/<suite>/<job>-<label>-r256/. Probe references (θ0, render 256): scene-4
# probe LIBERO-10 task 3 42/50 (113038), Spatial 76/100 (113024), Goal 74/100 on the first 10 eps
# per task (113038).
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-probe-eval
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G   # about 2.4 GB RSS per simulator process; 50-episode batch ~122 GB (memguard, 2026-10-26)
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
SLIM="${1:?usage: probe_eval.sbatch POLICY_DIR LABEL SUITE:TASK_IDS:N ...}"; LABEL="${2:?label}"; shift 2
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
FULL="$AVLA_RUNS/materialized/$SLURM_JOB_ID-$LABEL"
if [ -f "$SLIM/model.safetensors" ]; then FULL="$SLIM"
else PYTHONPATH="$PWD" python scripts/slim_checkpoint.py --materialize "$SLIM" "$FULL" || exit 1; fi
export AVLA_SUBGOALS=1
# Parallel mode runs one simulator process per episode of every spec at once (batch = episodes per
# task). Above AVLA_MAX_ENVS processes it falls back to sequential: our Slurm site does not enforce --mem, and
# four suites with 50-episode batches ran the node out of memory on 2026-10-26 (scripts/memguard.py).
if [ "${AVLA_PROBE_PARALLEL:-0}" = 1 ]; then
    ENVS=0; for spec in "$@"; do ENVS=$(( ENVS + ${spec##*:} )); done
    if [ "$ENVS" -gt "${AVLA_MAX_ENVS:-64}" ]; then
        echo "== $ENVS concurrent simulator processes > AVLA_MAX_ENVS=${AVLA_MAX_ENVS:-64}: running the specs one after another"
        AVLA_PROBE_PARALLEL=0
    fi
fi
pids=()
for spec in "$@"; do
    IFS=: read -r SUITE TASKS N <<< "$spec"
    if [ "${AVLA_PROBE_PARALLEL:-0}" = 1 ]; then   # all specs at once on the one GPU, one log each
        LOG="$AVLA_RUNS/eval/$SLURM_JOB_ID-$LABEL-$SUITE.log"; mkdir -p "$(dirname "$LOG")"
        SLURM_JOB_ID="$SLURM_JOB_ID-$LABEL" bash scripts/eval.sh "$SUITE" "$TASKS" "$N" "$FULL" > "$LOG" 2>&1 &
        pids+=($!)
    else
        SLURM_JOB_ID="$SLURM_JOB_ID-$LABEL" bash scripts/eval.sh "$SUITE" "$TASKS" "$N" "$FULL" || echo "== $spec FAILED"
    fi
done
for p in "${pids[@]}"; do wait "$p" || echo "== a probe eval FAILED (see $AVLA_RUNS/eval/$SLURM_JOB_ID-$LABEL-*.log)"; done
[ ${#pids[@]} -gt 0 ] && grep -h "overall:" "$AVLA_RUNS/eval/$SLURM_JOB_ID-$LABEL"-*.log
[ "$FULL" != "$SLIM" ] && rm -rf "$FULL"
echo "PROBE EVAL DONE $LABEL $(date -Is)"
