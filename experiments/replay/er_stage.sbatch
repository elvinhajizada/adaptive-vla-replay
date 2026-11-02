#!/bin/bash
# Random replay, one stage: fine-tune the previous stage's checkpoint on the next task of the
# sequence with replay of θ0's 40 tasks and the earlier new tasks, k frames each, then
# the per-stage evals on the last checkpoint: the task just learned x 50 (batch 50) alone,
# then the earlier new tasks x 20 and the three probes concurrently on the same GPU.
#
# Usage: sbatch --job-name=avla-er<K>-s<N> [--dependency=afterok:<prev job>] \
#            experiments/replay/er_stage.sbatch STAGE K PREV_POLICY_DIR PREV_SPEC
#   STAGE            2 to 5 (stage 1 is the single-stage pilot at the same k, e.g. 115608 for k 14)
#   K                replay frames per task
#   PREV_POLICY_DIR  the previous stage's policy dir (slim or full)
#   PREV_SPEC        the previous stage's replay spec: every task buffered there must keep its frames
# Order: Config.new_task_ids = 26, 18, 23, 27, 34. Seeds: buffer and sampler AVLA_BUF_SEED (default
# 0), training AVLA_SEED (default 1000), as the pilot; replay seeds 2 and 3 change both.
# Checkpoints at step 100, mid-stage and the last; all slimmed at the end. AVLA_STAGE_PROBES=0 keeps
# only the just-learned eval (E50: final probes only).
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G   # about 2.4 GB RSS per simulator process; 50-episode batch ~122 GB (memguard, 2026-10-26)
#SBATCH --time=05:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
STAGE="${1:?usage: er_stage.sbatch STAGE K PREV_POLICY_DIR PREV_SPEC}"; K="${2:?K}"
PREV="${3:?PREV_POLICY_DIR}"; PREV_SPEC="${4:?PREV_SPEC}"
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export PYTHONPATH="$PWD"

read -r ENV_ID EARLIER < <(python - "$STAGE" <<'PY'
import sys
from adaptive_vla.config import Config
cfg, stage = Config(), int(sys.argv[1])
assert 2 <= stage <= len(cfg.new_task_ids), stage
print(cfg.new_task_ids[stage - 1], ",".join(map(str, cfg.new_task_ids[:stage - 1])))
PY
)
LABEL="er-k$K-s$STAGE"
SPEC="$AVLA_RUNS/replay_specs/$SLURM_JOB_ID-$LABEL.json"
mkdir -p "$(dirname "$SPEC")"
STEPS=$(python scripts/replay_spec.py --current "$ENV_ID" --earlier "$EARLIER" --old all --k "$K" \
    --seed "${AVLA_BUF_SEED:-0}" --keep "$PREV_SPEC" --out "$SPEC" | tail -1)
MID=$(( STEPS / 2 ))
echo "== ER stage $STAGE: env $ENV_ID, k $K, earlier $EARLIER, $STEPS steps, from $PREV"

INIT="$PREV"
if [ ! -f "$PREV/model.safetensors" ]; then
    INIT="$AVLA_RUNS/materialized/$SLURM_JOB_ID-init"
    python scripts/slim_checkpoint.py --materialize "$PREV" "$INIT"
fi
AVLA_INIT="$INIT" AVLA_REPLAY_SPEC="$SPEC" AVLA_SAVE_STEPS="100,$MID" bash scripts/train.sh - "$STEPS" "$STEPS" 64
[ "$INIT" != "$PREV" ] && rm -rf "$INIT"
RUN="$AVLA_RUNS/train/$SLURM_JOB_ID"
CKPT="$RUN/checkpoints/last/pretrained_model"

export AVLA_SUBGOALS=1
ID="$SLURM_JOB_ID-$LABEL"
SLURM_JOB_ID="$ID" bash scripts/eval.sh libero_90 "[$ENV_ID]" 50 "$CKPT" > "$RUN/eval-learned.log" 2>&1 \
    || echo "== just-learned eval FAILED (see $RUN/eval-learned.log)"
pids=()
if [ "${AVLA_STAGE_PROBES:-1}" = 1 ]; then
SLURM_JOB_ID="$ID-earlier" bash scripts/eval.sh libero_90 "[$EARLIER]" 20 "$CKPT" > "$RUN/eval-earlier.log" 2>&1 &
pids+=($!)
for spec in "libero_10:[3]:20" "libero_spatial:[1,2,3,7,8]:10" "libero_goal:[1,2,4,7,8,9]:10"; do
    IFS=: read -r SUITE TASKS N <<< "$spec"
    SLURM_JOB_ID="$ID" bash scripts/eval.sh "$SUITE" "$TASKS" "$N" "$CKPT" > "$RUN/eval-$SUITE.log" 2>&1 &
    pids+=($!)
done
fi
for p in "${pids[@]}"; do wait "$p" || echo "== an eval FAILED (see $RUN/eval-*.log)"; done
grep -h "overall:" "$RUN"/eval-*.log || true

python scripts/slim_checkpoint.py "$RUN"/checkpoints/[0-9]*/pretrained_model
echo "ER STAGE $STAGE DONE -> $CKPT $(date -Is)"
