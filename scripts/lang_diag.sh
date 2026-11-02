#!/bin/bash
# Language check: diagnostic on LIBERO-Goal (true / blank / swapped instruction).
# Protocol and the "swapped" construction: adaptive_vla/lang_diag.py (module docstring).
#
# Usage: bash scripts/lang_diag.sh TASKS [PAIRS] [N_EPISODES] [POLICY_DIR]
#   TASKS       LIBERO-Goal task ids used as A, e.g. "8,9,7"
#   PAIRS       "cycle" (default: A -> next task in TASKS) or an explicit list "8:9,9:7,7:8"
#   N_EPISODES  per task and condition (default 10); 0 = env selftest only, no policy rollouts
#   POLICY_DIR  checkpoint dir (default: theta0 snapshot at the pinned revision)
# Env vars: AVLA_RENDER=<px> render size (as in eval.sh; default 256, the protocol),
#           AVLA_DEVICE=cuda|cpu (default cuda), AVLA_BLANK=<text> blank instruction (default "").
#
# Same eval settings as scripts/eval.sh: init_states=true, hard resets, seed 1000,
# batch_size = N_EPISODES, offline, MUJOCO_GL=egl. Every run starts with an env selftest
# (2 async envs per cell, no policy) that checks instructions, installed goals and state sizes.
# Output: $AVLA_RUNS/lang_diag/<run id>/ with eval_info.json, lang_diag_spec.json,
# lang_diag_summary.json, videos/<condition>_<task>/, run_info.json.
set -euo pipefail
TASKS="${1:?usage: lang_diag.sh TASKS [PAIRS] [N_EPISODES] [POLICY_DIR]}"
PAIRS="${2:-cycle}"
N_EP="${3:-10}"
DEVICE="${AVLA_DEVICE:-cuda}"
BLANK="${AVLA_BLANK:-}"
SEED=1000

cd "${AVLA_REPO:-$(dirname "${BASH_SOURCE[0]}")/..}"
set +u; source scripts/env.sh; set -u
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # async env workers import adaptive_vla
THETA0="$HF_HOME/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/$(python -c 'from adaptive_vla.config import Config; print(Config().pretrained_revision)')"
POLICY="${4:-$THETA0}"
AVLA_RENDER="${AVLA_RENDER:-256}"
RUN_ID="${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}"
[ -n "${AVLA_RENDER:-}" ] && RUN_ID="$RUN_ID-r$AVLA_RENDER"
[ "$DEVICE" != cuda ] && RUN_ID="$RUN_ID-$DEVICE"
OUT="$AVLA_RUNS/lang_diag/$RUN_ID"

RENDER_ARG=()
[ -n "${AVLA_RENDER:-}" ] && RENDER_ARG=(--env.observation_height="$AVLA_RENDER" --env.observation_width="$AVLA_RENDER")
COMMON=(--tasks="$TASKS" --pairs="$PAIRS" --blank="$BLANK"
  --policy.path="$POLICY"
  --policy.device="$DEVICE"
  --env.type=libero
  --env.task=libero_goal
  "${RENDER_ARG[@]}"
  --env.init_states=true
  --env.hard_reset=true
  --seed="$SEED")

echo "== host $(hostname)  job ${SLURM_JOB_ID:-none}  device $DEVICE  cpus ${SLURM_CPUS_PER_TASK:-?}  $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
[ "$DEVICE" = cuda ] && nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
python -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads(), 'visible GPUs', torch.cuda.device_count())"

echo "== env selftest (no policy rollouts)"
python -m adaptive_vla.lang_diag --selftest "${COMMON[@]}" \
  --eval.n_episodes=2 --eval.batch_size=2 --output_dir="$OUT/selftest"
if [ "$N_EP" = 0 ]; then echo "LANG_DIAG SELFTEST ONLY, DONE"; exit 0; fi

CMD=(python -m adaptive_vla.lang_diag "${COMMON[@]}"
  --eval.n_episodes="$N_EP"
  --eval.batch_size="$N_EP"
  --output_dir="$OUT")
echo "== ${CMD[*]}"
START=$(date +%s)
"${CMD[@]}"
END=$(date +%s)
python scripts/write_run_info.py "$OUT" "$POLICY" "$START" "$END" "${CMD[*]}"
echo "== eval wall-clock $((END - START))s  -> $OUT"
echo "LANG_DIAG DONE"
