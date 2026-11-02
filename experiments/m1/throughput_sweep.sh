#!/bin/bash
# M1 throughput: eval throughput sweep for theta0 on one A100 (episodes per GPU-hour vs batch size,
# CPU count and parallel tasks). Not a success-rate result: configs use different episode sets.
#
# Usage: bash experiments/m1/throughput_sweep.sh   (inside a GPU allocation; see throughput_sweep.sbatch)
# Env var AVLA_RENDER=<px> (default 256, the size chosen for the language diagnostic).
#
# Configs (LIBERO-Spatial, theta0, init_states=true, hard resets, seed 1000, n_action_steps=1):
#   b10_c12   task 0, 10 episodes, batch 10, pinned to 6 cores + HT siblings (= old 12-CPU jobs)
#   b10       task 0, 10 episodes, batch 10, all allocated CPUs
#   b20       task 0, 20 episodes, batch 20
#   b50       task 0, 50 episodes, batch 50 (the planned final-eval size)
#   par2      tasks 0,1, 10 episodes each, batch 10, 2 tasks in parallel (--env.max_parallel_tasks)
#   par4      tasks 0-3, 10 episodes each, batch 10, 4 tasks in parallel
# Confound: a batch runs until its slowest episode ends, and failures run to the 280-step limit,
# so success rate shifts time per episode. Success is reported next to each timing.
# GPU utilization of our GPU is sampled every 2 s (nvidia-smi, matched by UUID).
# Output: $AVLA_RUNS/throughput/<run id>/<config>/eval_info.json, gpu_util.log, summary.json.
set -euo pipefail
cd "${AVLA_REPO:-$(dirname "${BASH_SOURCE[0]}")/../..}"
set +u; source scripts/env.sh; set -u
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
RENDER="${AVLA_RENDER:-256}"
THETA0="$HF_HOME/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/$(python -c 'from adaptive_vla.config import Config; print(Config().pretrained_revision)')"
RUN_ID="${SLURM_JOB_ID:-local-$(date +%Y%m%d-%H%M%S)}-r$RENDER"
OUT="$AVLA_RUNS/throughput/$RUN_ID"
mkdir -p "$OUT"

echo "== host $(hostname)  job ${SLURM_JOB_ID:-none}  cpus ${SLURM_CPUS_PER_TASK:-?}  render $RENDER  $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  affinity $(taskset -cp $$ | cut -d: -f2)"
UUID=$(python -c "import torch; print('GPU-' + str(torch.cuda.get_device_properties(0).uuid))")
echo "our GPU: $UUID"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv

# 6 physical cores + their hyperthread siblings out of our allocation (the old 12-CPU footprint).
SMALL=$(python - <<'EOF'
import os
allowed = sorted(os.sched_getaffinity(0))
cores, picked = [], []
for c in allowed:
    sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
    if sib not in cores:
        cores.append(sib)
for sib in cores[:6]:
    for part in sib.split(","):
        a, _, b = part.partition("-")
        picked += [x for x in range(int(a), int(b or a) + 1) if x in allowed]
print(",".join(map(str, picked)))
EOF
)
echo "b10_c12 pinned to CPUs $SMALL"

nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 \
  | stdbuf -oL grep "$UUID" | while read -r line; do echo "$(date +%s) $line"; done > "$OUT/gpu_util.log" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

run() {   # run NAME TASK_IDS N_EP BATCH PARALLEL [taskset cpu list]
    local name=$1 tasks=$2 n=$3 batch=$4 par=$5 pin=${6:-}
    local pre=()
    [ -n "$pin" ] && pre=(taskset -c "$pin")
    echo "== $name: tasks $tasks, $n episodes/task, batch $batch, parallel tasks $par ${pin:+(pinned)}"
    local start=$(date +%s)
    "${pre[@]}" lerobot-eval \
        --policy.path="$THETA0" --policy.device=cuda \
        --env.type=libero --env.task=libero_spatial --env.task_ids="$tasks" \
        --env.observation_height="$RENDER" --env.observation_width="$RENDER" \
        --env.init_states=true --env.hard_reset=true --env.max_parallel_tasks="$par" \
        --eval.n_episodes="$n" --eval.batch_size="$batch" \
        --seed=1000 --output_dir="$OUT/$name" > "$OUT/$name.log" 2>&1 \
        || { echo "!! $name failed, see $OUT/$name.log"; tail -20 "$OUT/$name.log"; }
    local end=$(date +%s)
    echo "$name $start $end $tasks $n $batch $par ${pin:+12}" >> "$OUT/configs.txt"
    echo "   wall-clock $((end - start))s"
}

START=$(date +%s)
run b10_c12 "[0]"       10 10 1 "$SMALL"
run b10     "[0]"       10 10 1
run b20     "[0]"       20 20 1
run b50     "[0]"       50 50 1
run par2    "[0,1]"     10 10 2
run par4    "[0,1,2,3]" 10 10 4
END=$(date +%s)
kill $SMI_PID 2>/dev/null || true

python - "$OUT" "${SLURM_CPUS_PER_TASK:-0}" <<'EOF'
import json, sys
from pathlib import Path
out, ncpu = Path(sys.argv[1]), int(sys.argv[2])
util = []
for line in (out / "gpu_util.log").read_text().splitlines():
    t, rest = line.split(" ", 1)
    _, u, m = [x.strip() for x in rest.split(",")]
    util.append((int(t), float(u), float(m)))
rows = []
for line in (out / "configs.txt").read_text().splitlines():
    name, start, end, tasks, n, batch, par, *pin = line.split()
    start, end = int(start), int(end)
    f = out / name / "eval_info.json"
    if not f.exists():
        rows.append({"config": name, "failed": True}); continue
    o = json.load(open(f))["overall"]
    w = [u for t, u, _ in util if start <= t <= end]
    mem = [m for t, _, m in util if start <= t <= end]
    rows.append({
        "config": name, "tasks": tasks, "episodes": o["n_episodes"], "batch": int(batch),
        "parallel_tasks": int(par), "cpus": int(pin[0]) if pin else ncpu,
        "wall_s": end - start, "eval_s": round(o["eval_s"], 1),
        "s_per_episode": round(o["eval_s"] / o["n_episodes"], 1),
        "episodes_per_gpu_h": round(3600 * o["n_episodes"] / o["eval_s"], 1),
        "success_pc": round(o["pc_success"], 1),
        "gpu_util_mean": round(sum(w) / len(w), 1) if w else None,
        "gpu_mem_max_mib": max(mem) if mem else None,
    })
json.dump(rows, open(out / "summary.json", "w"), indent=2)
print(f"{'config':8s} {'eps':>4s} {'batch':>5s} {'par':>3s} {'cpus':>4s} {'s/ep':>6s} {'ep/GPU-h':>8s} "
      f"{'success':>7s} {'GPU util':>8s} {'GPU MiB':>8s}")
for r in rows:
    if r.get("failed"):
        print(f"{r['config']:8s} FAILED"); continue
    print(f"{r['config']:8s} {r['episodes']:4d} {r['batch']:5d} {r['parallel_tasks']:3d} {r['cpus']:4d} "
          f"{r['s_per_episode']:6.1f} {r['episodes_per_gpu_h']:8.1f} {r['success_pc']:6.1f}% "
          f"{r['gpu_util_mean'] if r['gpu_util_mean'] is not None else float('nan'):7.1f}% "
          f"{r['gpu_mem_max_mib'] or 0:8.0f}")
EOF
python scripts/write_run_info.py "$OUT" "$THETA0" "$START" "$END" "throughput_sweep.sh render=$RENDER"
echo "== sweep wall-clock $((END - START))s  -> $OUT"
echo "THROUGHPUT SWEEP DONE"
