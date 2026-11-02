#!/bin/bash
# One-command fine-tune of theta0 on LIBERO-90 task(s) from the converted dataset (pinned lerobot-train).
#
# Usage: bash scripts/train.sh TASK_INDICES STEPS [SAVE_FREQ] [BATCH]
#   TASK_INDICES  nvidia task_index list, e.g. "[12]" or "[3,7]" (episodes are looked up)
#   STEPS         training steps
#   SAVE_FREQ     checkpoint every N steps (default: STEPS, i.e. only the final one)
#   BATCH         batch size (default 64, the LeRobot LIBERO docs recipe)
# Env var AVLA_LR=<peak lr> overrides θ0's peak LR (1e-4); the final LR scales with it
# (peak / 40, θ0's ratio 1e-4 : 2.5e-6), so the schedule keeps its shape. Unset = θ0's recipe.
# Env var AVLA_LOSS_DIMS=arm|grip supervises only those action dims (scripts/masked_train.py; E26).
# Env var AVLA_REPLAY_SPEC=<spec.json> trains with experience replay (scripts/replay_train.py, M2):
#   the data becomes the merged libero_all_theta0, TASK_INDICES is ignored (pass "-"), and the
#   spec decides every batch; AVLA_SAVE_STEPS=a,b,... keeps checkpoints at those steps only.
# Env vars AVLA_WARM_ADAM=<N> (N zero-LR updates on θ0's data first, so Adam starts warm) and
# AVLA_STOP_STEP=<S> (stop after the step-S checkpoint) run scripts/optim_train.py (E35).
# Env var AVLA_SEED overrides the training seed (default 1000; the M2 seeds vary it).
# Env var AVLA_INIT starts from a checkpoint instead of θ0; AVLA_TRAIN_ARGS appends raw
# lerobot-train arguments (plumbing only, e.g. "--policy.device=cpu" for a CPU smoke run).
#
# Recipe: SmolVLA defaults from the pinned LeRobot (action expert only, frozen vision,
# AdamW lr 1e-4, 1k warmup, cosine decay to 30k). The data is libero90_theta0, which carries
# theta0's normalizer stats (docs/LIBERO.md, section 5, step 4); scripts/check_normalizer.py
# verifies that every saved checkpoint still has them.
# Output: $AVLA_RUNS/train/<run id>/ (lerobot output dir) + run_info.json + normalizer check.
set -euo pipefail
TASKS="${1:?usage: train.sh TASK_INDICES STEPS [SAVE_FREQ] [BATCH]}"
STEPS="${2:?usage: train.sh TASK_INDICES STEPS [SAVE_FREQ] [BATCH]}"
SAVE_FREQ="${3:-$STEPS}"
BATCH="${4:-64}"
SEED="${AVLA_SEED:-1000}"
# Data workers: at most 8. 8 workers feed a batch every 0.18 s vs the 0.58 s a step takes, even
# with three loaders at once (0.37 s); more workers only add spawn start-up and CPU contention
# (experiments/m1/dataloader_bench.py, job 5683198, 2026-10-22).
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} < 8 ? ${SLURM_CPUS_PER_TASK:-8} : 8 ))

cd "${AVLA_REPO:-$(dirname "${BASH_SOURCE[0]}")/..}"
set +u; source scripts/env.sh; set -u
THETA0="$HF_HOME/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/$(python -c 'from adaptive_vla.config import Config; print(Config().pretrained_revision)')"
RUN_ID="${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}"
OUT="$AVLA_RUNS/train/$RUN_ID"
# Start policy: θ0 unless AVLA_INIT names a loadable checkpoint (the previous M2 stage, materialized).
INIT="${AVLA_INIT:-$THETA0}"

if [ -n "${AVLA_REPLAY_SPEC:-}" ]; then
  DATA="$AVLA_DSS/data/libero_all_theta0"
  DATA_ARGS=(--dataset.repo_id=local/libero_all_theta0 --dataset.root="$DATA")
  echo "== replay spec $AVLA_REPLAY_SPEC: $(python -c "
import json; s = json.load(open('$AVLA_REPLAY_SPEC'))
print(s['current_key'], 'k', s['k'], 'seed', s['seed'], 'steps', s['steps'], s['ledger'])")"
else
  DATA="$AVLA_DSS/data/libero90_theta0"
  EPISODES=$(python -c "
from adaptive_vla.libero_data import episodes_for_tasks
eps = episodes_for_tasks('$DATA', $TASKS)
assert eps, 'no episodes for tasks $TASKS'
print('[' + ','.join(map(str, eps)) + ']')")
  echo "== tasks $TASKS -> $(python -c "print(len($EPISODES))") episodes"
  DATA_ARGS=(--dataset.repo_id=local/libero90_theta0 --dataset.root="$DATA" --dataset.episodes="$EPISODES")
fi

CMD=(lerobot-train
  --policy.path="$INIT"
  --policy.push_to_hub=false
  "${DATA_ARGS[@]}"
  --dataset.video_backend=pyav
  --batch_size="$BATCH"
  --steps="$STEPS"
  --save_freq="$SAVE_FREQ"
  --log_freq=50
  --num_workers="$WORKERS"
  --seed="$SEED"
  --wandb.enable=false
  --output_dir="$OUT")
if [ -n "${AVLA_LR:-}" ]; then
  CMD+=(--policy.optimizer_lr="$AVLA_LR" --policy.scheduler_decay_lr="$(python -c "print($AVLA_LR / 40)")")
fi
if [ -n "${AVLA_TRAIN_ARGS:-}" ]; then
  read -r -a EXTRA <<< "$AVLA_TRAIN_ARGS"
  CMD+=("${EXTRA[@]}")
fi
if [ -n "${AVLA_LOSS_DIMS:-}" ]; then
  CMD=(python scripts/masked_train.py "${CMD[@]:1}")
fi
if [ -n "${AVLA_WARM_ADAM:-}${AVLA_STOP_STEP:-}" ]; then
  [ -z "${AVLA_LOSS_DIMS:-}${AVLA_REPLAY_SPEC:-}" ] || { echo "AVLA_WARM_ADAM/AVLA_STOP_STEP combine with neither AVLA_LOSS_DIMS nor AVLA_REPLAY_SPEC" >&2; exit 2; }
  CMD=(python scripts/optim_train.py "${CMD[@]:1}")
fi
if [ -n "${AVLA_REPLAY_SPEC:-}" ]; then
  [ -z "${AVLA_LOSS_DIMS:-}" ] || { echo "AVLA_REPLAY_SPEC and AVLA_LOSS_DIMS do not combine" >&2; exit 2; }
  CMD=(python scripts/replay_train.py "${CMD[@]:1}")
fi

echo "== host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || echo "(no GPU visible)"
echo "== ${CMD[*]}" | cut -c1-600

START=$(date +%s)
"${CMD[@]}"
END=$(date +%s)
echo "== train wall-clock $((END - START))s for $STEPS steps (incl. startup)"

python scripts/check_normalizer.py "$OUT" "$THETA0"
REPLAY_INFO=()
if [ -n "${AVLA_REPLAY_SPEC:-}" ]; then
  cp "$AVLA_REPLAY_SPEC" "$OUT/replay_spec.json"
  REPLAY_INFO=(replay_spec="$OUT/replay_spec.json" save_steps="${AVLA_SAVE_STEPS:-last}")
fi
python scripts/write_run_info.py "$OUT" "$THETA0" "$START" "$END" "${CMD[*]}" \
  tasks="$TASKS" steps="$STEPS" batch_size="$BATCH" peak_lr="${AVLA_LR:-default}" loss_dims="${AVLA_LOSS_DIMS:-all}" \
  seed="$SEED" init="$INIT" warm_adam="${AVLA_WARM_ADAM:-0}" stop_step="${AVLA_STOP_STEP:-none}" "${REPLAY_INFO[@]}"
echo "TRAIN DONE -> $OUT"
