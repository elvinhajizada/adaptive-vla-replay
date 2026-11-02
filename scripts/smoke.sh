#!/bin/bash
# Smoke test body (called by smoke_gpu.sbatch / smoke_cpu.sbatch).
# Checks: device visible, LIBERO builds offline under EGL, θ0 runs end-to-end through
# lerobot-eval (LIBERO-Spatial task 0, 2 episodes). Not a reported number.
# Usage: bash scripts/smoke.sh {cuda|cpu}
set -euo pipefail
DEVICE="${1:?usage: smoke.sh cuda|cpu}"

cd "${AVLA_REPO:-$(dirname "${BASH_SOURCE[0]}")/..}"
set +u; source scripts/env.sh; set -u
OUT="$AVLA_RUNS/smoke/${SLURM_JOB_ID:-local}-$DEVICE"
THETA0="$HF_HOME/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/$(python -c 'from adaptive_vla.config import Config; print(Config().pretrained_revision)')"

echo "== host $(hostname)  job ${SLURM_JOB_ID:-none}  device $DEVICE  cpus ${SLURM_CPUS_PER_TASK:-?}  $(date -Is)"
if [ "$DEVICE" = cuda ]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python -c "import torch; print('torch', torch.__version__, 'threads', torch.get_num_threads(), 'cuda', torch.cuda.is_available())"

echo "== LIBERO env reset (offline, MUJOCO_GL=$MUJOCO_GL)"
python - <<'EOF'
import time
t = time.time()
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
env = LiberoEnv(task_suite=suite, task_id=0, task_suite_name="libero_spatial")
obs, info = env.reset(seed=0)
print("task:", env.task_description)
print("obs:", {k: getattr(v, "shape", None) for k, v in obs.get("pixels", obs).items()})
print(f"reset ok in {time.time() - t:.1f}s")
env.close()
EOF

echo "== lerobot-eval theta0, libero_spatial task 0, 2 episodes"
START=$(date +%s)
lerobot-eval \
  --policy.path="$THETA0" \
  --policy.device="$DEVICE" \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=2 \
  --seed=1000 \
  --output_dir="$OUT"
echo "== eval wall-clock $(( $(date +%s) - START ))s"
echo "SMOKE DONE"
