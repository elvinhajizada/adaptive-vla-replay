#!/bin/bash
# Single-stage replay pilot: θ0 + one new task with θ0-data replay at budget K, then the
# per-stage evals on the last checkpoint.
#
# Usage: sbatch --job-name=avla-pilot-<label> experiments/replay/pilot_stage.sbatch ENV_ID K LABEL [OLD]
#   ENV_ID  libero_90 env id of the new task (26 or 23 in the pilot)
#   K       replay frames per θ0 task (14, 140, 700) or "all"
#   LABEL   run label, e.g. t26-k140
#   OLD     θ0 suites in the buffer (default all; "libero_spatial,libero_object,libero_10" for the
#           Goal hold-out run)
# Seeds: buffer and sampler seed 0, training seed 1000 (the single-task fine-tunes used 1000, so the
# Sequential baselines 114663 / 113795 are paired on everything but the replay). Replay seeds
# override both: AVLA_BUF_SEED (buffer and sampler) and AVLA_SEED (training, read by train.sh).
# Checkpoints at step 100, mid-stage and the last; all slimmed at the end. Evals at protocol
# settings with subgoal logging: the new task x 50 (batch 50) alone, then the three probes
# concurrently on the same GPU (eval.sh runs a suite's tasks one after another). AVLA_STAGE_PROBES=0 skips
# the probes (E50: final probes only).
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G   # about 2.4 GB RSS per simulator process; 50-episode batch ~122 GB (memguard, 2026-10-26)
#SBATCH --time=05:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
ENV_ID="${1:?usage: pilot_stage.sbatch ENV_ID K LABEL [OLD]}"; K="${2:?K}"; LABEL="${3:?LABEL}"; OLD="${4:-all}"
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export PYTHONPATH="$PWD"

SPEC="$AVLA_RUNS/replay_specs/$SLURM_JOB_ID-$LABEL.json"
mkdir -p "$(dirname "$SPEC")"
STEPS=$(python scripts/replay_spec.py --current "$ENV_ID" --k "$K" --old "$OLD" --seed "${AVLA_BUF_SEED:-0}" --out "$SPEC" | tail -1)
MID=$(( STEPS / 2 ))
echo "== pilot $LABEL: env $ENV_ID, k $K, old $OLD, $STEPS steps, checkpoints at 100, $MID, $STEPS"

AVLA_REPLAY_SPEC="$SPEC" AVLA_SAVE_STEPS="100,$MID" bash scripts/train.sh - "$STEPS" "$STEPS" 64
RUN="$AVLA_RUNS/train/$SLURM_JOB_ID"
CKPT="$RUN/checkpoints/last/pretrained_model"

export AVLA_SUBGOALS=1
ID="$SLURM_JOB_ID-$LABEL"
SLURM_JOB_ID="$ID" bash scripts/eval.sh libero_90 "[$ENV_ID]" 50 "$CKPT" || echo "== new-task eval FAILED"
pids=()
if [ "${AVLA_STAGE_PROBES:-1}" = 1 ]; then
for spec in "libero_10:[3]:20" "libero_spatial:[1,2,3,7,8]:10" "libero_goal:[1,2,4,7,8,9]:10"; do
    IFS=: read -r SUITE TASKS N <<< "$spec"
    SLURM_JOB_ID="$ID" bash scripts/eval.sh "$SUITE" "$TASKS" "$N" "$CKPT" > "$RUN/eval-$SUITE.log" 2>&1 &
    pids+=($!)
done
fi
for p in "${pids[@]}"; do wait "$p" || echo "== a probe eval FAILED (see $RUN/eval-*.log)"; done
grep -h "Success rate\|overall" "$RUN"/eval-*.log | tail -6 || true

python scripts/slim_checkpoint.py "$RUN"/checkpoints/[0-9]*/pretrained_model
echo "PILOT DONE $LABEL $(date -Is)"
