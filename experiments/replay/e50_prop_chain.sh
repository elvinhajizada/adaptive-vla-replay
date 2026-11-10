#!/bin/bash
# E50: one random-replay sequence with the 2% budget split in proportion to task length (2.05% of
# every task's frames) instead of 140 frames per task. Buffer seed 0, training seed 1000, as E39.
# Stages as er_seed_chain.sh but with AVLA_STAGE_PROBES=0 (only the just-learned x 50 per stage);
# the final probes are E39's θ0 probes plus E47's far-scene LIBERO-10 tasks, split over two jobs
# (the earlier new tasks x 50 are left out to save GPU time; the slim checkpoint allows them later).
# Seed 2 (E51): buffer seed 1, training seed 1001, as E42.
# Run wherever you call sbatch (or with AVLA_SBATCH=scripts/run_local.sh, here):
#   bash experiments/replay/e50_prop_chain.sh [BUF TRAIN [FROM PREV_JOB]]   (default 0 1000)
# FROM PREV_JOB resumes a stopped chain at stage FROM (2 to 5) from the finished stage FROM-1 job
# PREV_JOB (E51: 1 1001 3 115942).
set -euo pipefail
K=p2.05; BUF=${1:-0}; TRAIN=${2:-1000}; FROM=${3:-1}; PREV_JOB=${4:-}
cd "${AVLA_REPO:-$(dirname "${BASH_SOURCE[0]}")/../..}"
set +u; source scripts/env.sh; set -u   # Slurm site settings and AVLA_RUNS
R="$AVLA_RUNS"
export AVLA_BUF_SEED="$BUF" AVLA_SEED="$TRAIN" AVLA_STAGE_PROBES=0   # sbatch exports these to every job
TAG="k$K-b$BUF-t$TRAIN"
if [ "$FROM" = 1 ]; then
    j=$(${AVLA_SBATCH:-sbatch} --parsable --job-name=avla-er-$TAG-s1 experiments/replay/pilot_stage.sbatch 26 "$K" "t26-$TAG" | cut -d';' -f1)
    spec="$R/replay_specs/$j-t26-$TAG.json"; ids="s1:$j"; prev=$j; dep=--dependency=afterok:$j
else
    prev=${PREV_JOB:?PREV_JOB is needed with FROM}; ids="resume:$prev"
    if [ "$FROM" = 2 ]; then spec="$R/replay_specs/$prev-t26-$TAG.json"
    else spec="$R/replay_specs/$prev-er-k$K-s$((FROM - 1)).json"; fi
    [ -f "$spec" ] && [ -d "$R/train/$prev/checkpoints/last/pretrained_model" ] \
        || { echo "no spec or checkpoint for job $prev: $spec" >&2; exit 1; }
    dep=   # PREV_JOB has finished; Slurm may have forgotten it
fi
for s in $(seq "$((FROM > 1 ? FROM : 2))" 5); do
    j=$(${AVLA_SBATCH:-sbatch} --parsable --job-name=avla-er-$TAG-s$s $dep \
        experiments/replay/er_stage.sbatch $s "$K" "$R/train/$prev/checkpoints/last/pretrained_model" "$spec" | cut -d';' -f1)
    ids="$ids s$s:$j"; spec="$R/replay_specs/$j-er-k$K-s$s.json"; prev=$j; dep=--dependency=afterok:$j
done
CKPT="$R/train/$prev/checkpoints/last/pretrained_model"
# One suite per label per job: two LIBERO-10 specs under one label would write the same eval dir.
a=$(${AVLA_SBATCH:-sbatch} --parsable --job-name=avla-er-$TAG-final --dependency=afterok:$prev scripts/probe_eval.sbatch \
    "$CKPT" "er-$TAG-s5-final" "libero_10:[2,3,9]:50" "libero_spatial:all:20" "libero_goal:all:20" | cut -d';' -f1)
b=$(${AVLA_SBATCH:-sbatch} --parsable --job-name=avla-er-$TAG-final-far --dependency=afterok:$prev scripts/probe_eval.sbatch \
    "$CKPT" "er-$TAG-s5-final-far" "libero_10:[0,1,4,6,7]:20" | cut -d';' -f1)
echo "$TAG $ids final:$a,$b"
