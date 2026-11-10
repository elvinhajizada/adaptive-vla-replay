#!/bin/bash
# Repeatability test: why do two identical closed-loop evals of θ0 disagree on some episodes?
# Four evals of θ0 on LIBERO-Spatial tasks 1, 2, 3 x 10 at the protocol settings, one after another
# on the same GPU, each logging every observation hash and action (AVLA_STEPLOG=1):
#   A, B  the protocol as is (the pinned eval_main sets cudnn.benchmark=True)
#   C, D  AVLA_DETERMINISTIC=1: no cuDNN autotuning, torch deterministic algorithms
# Then experiments/setup/repeatability_compare.py on A-B, C-D and A-C: where each pair first
# diverges (policy, simulator or reset) and how many outcomes flip.
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-repeat
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G   # 10 simulator processes at about 2.4 GB RSS each, plus the policy process
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
SUITE=libero_spatial; TASKS="[1,2,3]"; N=10
JOB="$SLURM_JOB_ID"; OUT="$AVLA_RUNS/eval/$SUITE"

python -m adaptive_vla.subgoal_env --selftest --env.task=$SUITE --env.task_ids="[1]" \
    --output_dir="$AVLA_RUNS/eval/selftest-$JOB" || { echo "selftest FAILED"; exit 1; }

export AVLA_SUBGOALS=1 AVLA_STEPLOG=1
for run in A B C D; do
    if [ "$run" = C ] || [ "$run" = D ]; then
        export AVLA_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
    fi
    SLURM_JOB_ID="$JOB-$run" bash scripts/eval.sh "$SUITE" "$TASKS" "$N" || echo "== run $run FAILED"
done

for pair in "A B" "C D" "A C"; do
    set -- $pair
    echo "== compare $1 vs $2"
    python experiments/setup/repeatability_compare.py "$OUT/$JOB-$1-r256" "$OUT/$JOB-$2-r256" \
        --out "$OUT/$JOB-compare-$1$2.json"
done
rm -rf "$AVLA_RUNS/eval/selftest-$JOB"
echo "REPEATABILITY DONE $(date -Is)"
