#!/bin/bash
# Noise-tolerance test (E58): how fast does θ0's success fall on each old task when Gaussian noise is
# added to its arm actions (AVLA_ACTION_NOISE, adaptive_vla/subgoal_env.py)? Protocol settings
# otherwise (render 256, init states, hard resets, seed 1000, batch = episodes per task).
# Usage: sbatch experiments/setup/noise_tolerance.sbatch "SIGMA ..." SUITE:TASK_IDS:N [...]
#   e.g. ... "0.5 1 2" libero_spatial:[7]:10 libero_goal:[7]:10       (calibration)
# Per noise level the specs run concurrently on the one GPU; outputs in
# $AVLA_RUNS/eval/<suite>/<job>-n<sigma>-r256/.
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-noise
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=80G   # two specs of 10 episodes: 20 simulator processes at about 2.4 GB, plus two policies
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
SIGMAS="${1:?usage: noise_tolerance.sbatch \"SIGMA ...\" SUITE:TASK_IDS:N ...}"; shift
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" AVLA_SUBGOALS=1
JOB="$SLURM_JOB_ID"
for s in $SIGMAS; do
    pids=()
    for spec in "$@"; do
        IFS=: read -r SUITE TASKS N <<< "$spec"
        LOG="$AVLA_RUNS/eval/$JOB-n$s-$SUITE.log"; mkdir -p "$(dirname "$LOG")"
        AVLA_ACTION_NOISE="$s" SLURM_JOB_ID="$JOB-n$s" bash scripts/eval.sh "$SUITE" "$TASKS" "$N" > "$LOG" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || echo "== an eval at sigma $s FAILED"; done
    echo "== sigma $s"; grep -h "action noise:\|^task \|overall:" "$AVLA_RUNS/eval/$JOB-n$s"-*.log
done
echo "NOISE TOLERANCE DONE $(date -Is)"
