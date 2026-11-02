#!/bin/bash
# Submit one full random-replay sequence for one seed: stage 1 by pilot_stage.sbatch (task 26,
# from θ0), stages 2 to 5 by er_stage.sbatch, then the final probes, chained with afterok.
# Seed 1 of E37 reused the pilot (buffer 0, training 1000); seeds 2 and 3 use (1, 1001), (2, 1002).
# Run wherever you call sbatch (without Slurm: AVLA_SBATCH=scripts/run_local.sh runs the jobs here,
# one after another): bash experiments/replay/er_seed_chain.sh K BUF_SEED TRAIN_SEED [S1_JOB]
#   S1_JOB  an existing stage-1 run at the same K and seeds (a pilot, e.g. 115609 for k 140, E39):
#           its checkpoint and replay spec start the chain and no stage-1 job is submitted
set -euo pipefail
K="${1:?usage: er_seed_chain.sh K BUF_SEED TRAIN_SEED [S1_JOB]}"; BUF="${2:?BUF_SEED}"; TRAIN="${3:?TRAIN_SEED}"
S1="${4:-}"
cd "${AVLA_REPO:-$(dirname "${BASH_SOURCE[0]}")/../..}"
set +u; source scripts/env.sh; set -u   # Slurm site settings and AVLA_RUNS
R="$AVLA_RUNS"
export AVLA_BUF_SEED="$BUF" AVLA_SEED="$TRAIN"      # sbatch exports the environment to every job
TAG="k$K-b$BUF-t$TRAIN"
if [ -n "$S1" ]; then
    spec=$(ls "$R"/replay_specs/"$S1"-*.json)        # exactly one spec per run; fails loudly otherwise
    [ -f "$R/train/$S1/checkpoints/last/pretrained_model/config.json" ] || { echo "no checkpoint for $S1"; exit 1; }
    j=$S1
else
    j=$(${AVLA_SBATCH:-sbatch} --parsable --job-name=avla-er-$TAG-s1 experiments/replay/pilot_stage.sbatch 26 "$K" "t26-$TAG" | cut -d';' -f1)
    spec="$R/replay_specs/$j-t26-$TAG.json"
fi
ids="s1:$j"; prev=$j
for s in 2 3 4 5; do
    dep="--dependency=afterok:$prev"; [ "$prev" = "$S1" ] && dep=""
    j=$(${AVLA_SBATCH:-sbatch} --parsable --job-name=avla-er-$TAG-s$s $dep \
        experiments/replay/er_stage.sbatch $s "$K" "$R/train/$prev/checkpoints/last/pretrained_model" "$spec" | cut -d';' -f1)
    ids="$ids s$s:$j"; spec="$R/replay_specs/$j-er-k$K-s$s.json"; prev=$j
done
j=$(AVLA_PROBE_PARALLEL=1 ${AVLA_SBATCH:-sbatch} --parsable --job-name=avla-er-$TAG-final --dependency=afterok:$prev \
    scripts/probe_eval.sbatch "$R/train/$prev/checkpoints/last/pretrained_model" "er-$TAG-s5-final" \
    "libero_90:[26,18,23,27]:50" "libero_spatial:all:20" "libero_goal:all:20" "libero_10:[2,3,9]:50" | cut -d';' -f1)
echo "$TAG $ids final:$j"
