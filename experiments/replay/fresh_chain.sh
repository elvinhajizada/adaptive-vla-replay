#!/bin/bash
# Fresh-buffer arm (E59): continue an existing k-frame stage-1 run through stages 2 to 5 with every
# task's buffer drawn anew at each stage from frames it has not held before (er_stage.sbatch with
# AVLA_REFRESH=1), then the final Spatial and Goal probes. Same tasks, steps, mixing and seeds as the
# fixed-buffer chain (er_seed_chain.sh), so the two share stage 1 and differ only in buffer reuse.
# Lean (E59): stages train only (no per-stage evals, 24 GB), and the final job runs only the two probes
# the reading rule uses, concurrently.
#   bash experiments/replay/fresh_chain.sh K BUF_SEED TRAIN_SEED S1_JOB
# E59: 14 0 1000 115608; 14 1 1001 115626 (a third seed, if needed: 14 2 1002 115632).
set -euo pipefail
K="${1:?usage: fresh_chain.sh K BUF_SEED TRAIN_SEED S1_JOB}"; BUF="${2:?BUF_SEED}"; TRAIN="${3:?TRAIN_SEED}"
S1="${4:?S1_JOB}"
cd "${AVLA_REPO:-$(dirname "${BASH_SOURCE[0]}")/../..}"
set +u; source scripts/env.sh; set -u   # Slurm site settings and AVLA_RUNS
R="$AVLA_RUNS"
export AVLA_BUF_SEED="$BUF" AVLA_SEED="$TRAIN" AVLA_REFRESH=1   # sbatch exports the environment to every job
export AVLA_STAGE_PROBES=0 AVLA_LEARNED_EVAL=0
TAG="k$K-fresh-b$BUF-t$TRAIN"
spec=$(ls "$R"/replay_specs/"$S1"-*.json)        # exactly one spec per run; fails loudly otherwise
grep -q "\"k\": \"$K\"" "$spec" && grep -q "\"seed\": $BUF," "$spec" \
    || { echo "$spec is not k $K, buffer seed $BUF"; exit 1; }
[ -f "$R/train/$S1/checkpoints/last/pretrained_model/config.json" ] || { echo "no checkpoint for $S1"; exit 1; }
ids="s1:$S1"; prev=$S1
for s in 2 3 4 5; do
    dep="--dependency=afterok:$prev"; [ "$prev" = "$S1" ] && dep=""
    j=$(${AVLA_SBATCH:-sbatch} --parsable --job-name=avla-er-$TAG-s$s --mem=24G --time=01:30:00 $dep \
        experiments/replay/er_stage.sbatch $s "$K" "$R/train/$prev/checkpoints/last/pretrained_model" "$spec" | cut -d';' -f1)
    ids="$ids s$s:$j"; spec="$R/replay_specs/$j-er-k$K-fresh-s$s.json"; prev=$j
done
j=$(AVLA_PROBE_PARALLEL=1 ${AVLA_SBATCH:-sbatch} --parsable --job-name=avla-er-$TAG-final --dependency=afterok:$prev \
    scripts/probe_eval.sbatch "$R/train/$prev/checkpoints/last/pretrained_model" "er-$TAG-s5-final" \
    "libero_spatial:all:20" "libero_goal:all:20" | cut -d';' -f1)
echo "$TAG $ids final:$j"
