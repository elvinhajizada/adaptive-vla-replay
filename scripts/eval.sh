#!/bin/bash
# One-command LIBERO eval of a SmolVLA policy (theta0 by default) through the pinned lerobot-eval.
#
# Usage: bash scripts/eval.sh SUITE [TASK_IDS] [N_EPISODES] [POLICY_DIR]
#   SUITE       libero_spatial | libero_object | libero_goal | libero_10 | libero_90
#   TASK_IDS    "all" (default) or a list like "[0,1,2]"
#   N_EPISODES  per task (default 10)
#   POLICY_DIR  checkpoint dir (default: theta0 snapshot at the pinned revision)
# Env var AVLA_RENDER=<px> overrides the render size (default 256, the protocol).
# Env var AVLA_SUBGOALS=1 also logs per-episode goal predicates (secondary metric; success is
# unchanged, see adaptive_vla/subgoal_env.py): writes subgoals.jsonl + subgoals_summary.json.
#
# Protocol (docs/LIBERO.md, section 10; decided 2026-10-19, render 2026-10-22): init_states=true,
# hard resets, seed 1000, render 256x256 (θ0's training-frame size; the env default 360 cost
# 12 pts on Spatial, jobs 113009 vs 113024), and batch_size = N_EPISODES, so that
# each task's episodes run in one batch and runs are paired across conditions (LeRobot docs).
# Output: $AVLA_RUNS/eval/<SUITE>/<run id>/ with eval_info.json, videos, run_info.json.
set -euo pipefail
SUITE="${1:?usage: eval.sh SUITE [TASK_IDS] [N_EPISODES] [POLICY_DIR]}"
TASK_IDS="${2:-all}"
N_EP="${3:-10}"
AVLA_RENDER="${AVLA_RENDER:-256}"
SEED=1000

cd "${AVLA_REPO:-$(dirname "${BASH_SOURCE[0]}")/..}"
set +u; source scripts/env.sh; set -u
THETA0="$HF_HOME/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/$(python -c 'from adaptive_vla.config import Config; print(Config().pretrained_revision)')"
POLICY="${4:-$THETA0}"
RUN_ID="${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}"
[ -n "${AVLA_RENDER:-}" ] && RUN_ID="$RUN_ID-r$AVLA_RENDER"
OUT="$AVLA_RUNS/eval/$SUITE/$RUN_ID"

TASK_ARG=()
[ "$TASK_IDS" != "all" ] && TASK_ARG=(--env.task_ids="$TASK_IDS")
RENDER_ARG=()
[ -n "${AVLA_RENDER:-}" ] && RENDER_ARG=(--env.observation_height="$AVLA_RENDER" --env.observation_width="$AVLA_RENDER")
EVAL_ENTRY=(lerobot-eval)
if [ "${AVLA_SUBGOALS:-0}" = 1 ]; then
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # async env workers import adaptive_vla
  EVAL_ENTRY=(python -m adaptive_vla.subgoal_env)
fi
CMD=("${EVAL_ENTRY[@]}"
  --policy.path="$POLICY"
  --policy.device=cuda
  --env.type=libero
  --env.task="$SUITE"
  "${TASK_ARG[@]}"
  "${RENDER_ARG[@]}"
  --env.init_states=true
  --env.hard_reset=true
  --eval.n_episodes="$N_EP"
  --eval.batch_size="$N_EP"
  --seed="$SEED"
  --output_dir="$OUT")

echo "== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
python -c "import torch; print('torch', torch.__version__, 'visible GPUs', torch.cuda.device_count())"
echo "== ${CMD[*]}"

START=$(date +%s)
"${CMD[@]}"
END=$(date +%s)

# Provenance next to the results.
python scripts/write_run_info.py "$OUT" "$POLICY" "$START" "$END" "${CMD[*]}"
echo "== eval wall-clock $((END - START))s  -> $OUT"
python - "$OUT/eval_info.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for t in d.get("per_task", []):
    s = t["metrics"]["successes"]
    print(f"task {t['task_id']:2d}: {sum(s):2d}/{len(s)}")
o = d.get("overall", {})
print(f"overall: {o.get('pc_success')}%  ({o.get('n_success')}/{o.get('n_episodes')})  CI95 {o.get('pc_success_ci95')}")
EOF
echo "EVAL DONE"
