#!/bin/bash
# Submit the no-replay sequence (E32) in one go: stage 1 = task 26 from θ0 (scripts/finetune_eval.sbatch,
# 8 passes, eval x 50), stages 2 to 5 by seq_stage.sbatch (tasks 18, 23, 27, 34), then the final probes,
# chained with afterok. Training seed 1000, as E32.
# E32 itself was submitted stage by stage, and its final probes used fewer episodes (10 per Spatial and
# Goal task, 20 per LIBERO-10 task); this chain runs the same final probes as er_seed_chain.sh.
# Run wherever you call sbatch (or with AVLA_SBATCH=scripts/run_local.sh, here): bash experiments/replay/seq_chain.sh
set -euo pipefail
cd "${AVLA_REPO:-$(dirname "${BASH_SOURCE[0]}")/../..}"
set +u; source scripts/env.sh; set -u   # Slurm site settings and AVLA_RUNS
R="$AVLA_RUNS"
j=$(${AVLA_SBATCH:-sbatch} --parsable --job-name=avla-seq-s1 scripts/finetune_eval.sbatch 26 8p | cut -d';' -f1)
ids="s1:$j"; prev=$j
for s in 2 3 4 5; do
    j=$(${AVLA_SBATCH:-sbatch} --parsable --job-name=avla-seq-s$s --dependency=afterok:$prev \
        experiments/replay/seq_stage.sbatch $s "$R/train/$prev/checkpoints/last/pretrained_model" | cut -d';' -f1)
    ids="$ids s$s:$j"; prev=$j
done
j=$(AVLA_PROBE_PARALLEL=1 ${AVLA_SBATCH:-sbatch} --parsable --job-name=avla-seq-final --dependency=afterok:$prev \
    scripts/probe_eval.sbatch "$R/train/$prev/checkpoints/last/pretrained_model" "seq-s5-final" \
    "libero_90:[26,18,23,27]:50" "libero_spatial:all:20" "libero_goal:all:20" "libero_10:[2,3,9]:50" | cut -d';' -f1)
echo "seq $ids final:$j"
