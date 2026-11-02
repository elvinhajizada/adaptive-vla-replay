#!/bin/bash
# M2 Sequential, one stage: fine-tune the previous stage's checkpoint on the next task of the
# sequence (8 passes, θ0's recipe, no replay), then the new-task row of the success matrix: the
# task just learned x 50 (batch 50), then every earlier task x 20, on one GPU. Probes run
# only after the last stage, as a separate probe_eval job.
#
# Usage: sbatch --job-name=avla-seq-s<N> [--dependency=afterok:<prev job>] \
#            experiments/m2/seq_stage.sbatch STAGE PREV_POLICY_DIR
#   STAGE            2 to 5 (stage 1 = 113804: task 26 from θ0, 8 passes, seed 1000, 44/50)
#   PREV_POLICY_DIR  the previous stage's policy dir (slim or full)
# Order: Config.new_task_ids = 26, 18, 23, 27, 34. Training seed 1000 (as stage 1).
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G   # about 2.4 GB RSS per simulator process; 50-episode batch ~122 GB (memguard, 2026-10-26)
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
STAGE="${1:?usage: seq_stage.sbatch STAGE PREV_POLICY_DIR}"; PREV="${2:?PREV_POLICY_DIR}"
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"   # the repo root: submit from it, or set AVLA_REPO
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD (submit from the repo root or set AVLA_REPO)" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export PYTHONPATH="$PWD"

read -r ENV_ID EARLIER TASK_INDEX STEPS < <(python - "$STAGE" <<'PY'
import os, sys
from libero.libero.benchmark import get_benchmark_dict
from adaptive_vla.config import Config
from adaptive_vla.libero_data import frames_for_tasks, steps_for_passes, task_index_for_instruction
cfg, stage = Config(), int(sys.argv[1])
assert 2 <= stage <= len(cfg.new_task_ids), stage
env_id, earlier = cfg.new_task_ids[stage - 1], cfg.new_task_ids[:stage - 1]
root = os.environ["AVLA_DSS"] + "/data/libero90_theta0"
idx = task_index_for_instruction(root, get_benchmark_dict()["libero_90"]().get_task(env_id).language)
steps = steps_for_passes(cfg.passes_per_task, frames_for_tasks(root, [idx]), cfg.batch_size)
print(env_id, "[" + ",".join(map(str, earlier)) + "]", idx, steps)
PY
)
echo "== Sequential stage $STAGE: env $ENV_ID (task_index $TASK_INDEX), $STEPS steps, from $PREV; earlier $EARLIER"

INIT="$PREV"
if [ ! -f "$PREV/model.safetensors" ]; then
    INIT="$AVLA_RUNS/materialized/$SLURM_JOB_ID-init"
    python scripts/slim_checkpoint.py --materialize "$PREV" "$INIT"
fi
AVLA_INIT="$INIT" bash scripts/train.sh "[$TASK_INDEX]" "$STEPS"
[ "$INIT" != "$PREV" ] && rm -rf "$INIT"
RUN="$AVLA_RUNS/train/$SLURM_JOB_ID"
CKPT="$RUN/checkpoints/last/pretrained_model"

export AVLA_SUBGOALS=1
ID="$SLURM_JOB_ID-seq-s$STAGE"
# One after the other: 50 + 20 simulator processes at once took about 178 GB (115606, 2026-10-26).
SLURM_JOB_ID="$ID" bash scripts/eval.sh libero_90 "[$ENV_ID]" 50 "$CKPT" > "$RUN/eval-learned.log" 2>&1 \
    || echo "== just-learned eval FAILED (see $RUN/eval-learned.log)"
SLURM_JOB_ID="$ID-earlier" bash scripts/eval.sh libero_90 "$EARLIER" 20 "$CKPT" > "$RUN/eval-earlier.log" 2>&1 \
    || echo "== earlier-task eval FAILED (see $RUN/eval-earlier.log)"
grep -h "Success rate\|overall" "$RUN"/eval-*.log | tail -8 || true

python scripts/slim_checkpoint.py "$CKPT"
echo "SEQ STAGE $STAGE DONE -> $CKPT $(date -Is)"
