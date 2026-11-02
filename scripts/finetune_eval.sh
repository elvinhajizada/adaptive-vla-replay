#!/bin/bash
# Fine-tune θ0 on one LIBERO-90 task for STEPS steps (one full schedule), then evaluate the final
# checkpoint on that task at the protocol settings (eval.sh: render 256, seed 1000, init states).
# The checkpoint is slimmed after the eval (scripts/slim_checkpoint.py).
# Used for the step-count sweep (1k/2k/3k on task 23) and the learnability check of the other chosen tasks.
#
# Usage: bash scripts/finetune_eval.sh ENV_ID STEPS|PASSESp [N_EPISODES]
#   ENV_ID      libero_90 env task id (config.py new_task_ids); mapped to the dataset task_index by
#               instruction string, never by integer id (docs/LIBERO.md, section 5)
#   STEPS       training steps; the scheduler fits warmup + cosine decay into this length;
#               or PASSES with a trailing p (e.g. 8p): steps = ceil(passes x task frames / 64),
#               the M2 protocol (fixed passes per task, since 2026-10-22)
#   N_EPISODES  eval episodes (default 50); batch = N_EPISODES
set -euo pipefail
ENV_ID="${1:?usage: finetune_eval.sh ENV_ID STEPS|PASSESp [N_EPISODES]}"
LENGTH="${2:?usage: finetune_eval.sh ENV_ID STEPS|PASSESp [N_EPISODES]}"
BATCH=64   # train.sh default
N_EP="${3:-50}"

cd "${AVLA_REPO:-$(dirname "${BASH_SOURCE[0]}")/..}"
set +u; source scripts/env.sh; set -u
read -r TASK_INDEX STEPS < <(python - "$ENV_ID" "$LENGTH" "$BATCH" <<'PY'
import os, sys
from libero.libero.benchmark import get_benchmark_dict
from adaptive_vla.libero_data import frames_for_tasks, steps_for_passes, task_index_for_instruction
env_id, length, batch = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
root = os.environ["AVLA_DSS"] + "/data/libero90_theta0"
lang = get_benchmark_dict()["libero_90"]().get_task(env_id).language
idx = task_index_for_instruction(root, lang)
frames = frames_for_tasks(root, [idx])
steps = steps_for_passes(float(length[:-1]), frames, batch) if length.endswith("p") else int(length)
print(f"env {env_id}: {lang!r} -> task_index {idx}, {frames} frames; {length} -> {steps} steps "
      f"({steps * batch / frames:.2f} passes at batch {batch})", file=sys.stderr)
print(idx, steps)
PY
)

bash scripts/train.sh "[$TASK_INDEX]" "$STEPS"
CKPT="$AVLA_RUNS/train/${SLURM_JOB_ID:?run inside a Slurm job}/checkpoints/last/pretrained_model"
bash scripts/eval.sh libero_90 "[$ENV_ID]" "$N_EP" "$CKPT"
# Keep only the delta vs θ0 (about 195 MB instead of 1.6 GB, lossless, verified before deleting);
# rebuild with: python scripts/slim_checkpoint.py --materialize "$CKPT" OUT_DIR
PYTHONPATH="$PWD" python scripts/slim_checkpoint.py "$CKPT"
echo "FINETUNE+EVAL DONE env $ENV_ID length $LENGTH steps $STEPS"
