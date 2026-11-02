#!/bin/bash
# Language check: theta0 language diagnostic on LIBERO-Goal, 3 tasks x 10 episodes x 3 conditions
# (true, blank, swapped). About 90 episodes. Body: scripts/lang_diag.sh.
# Decided 2026-10-21: tasks 8, 9, 7 with cyclic pairs (8->9, 9->7, 7->8); blank = "";
# render 256x256 (theta0's training-frame size), chosen before the Spatial 256 check (job 113024)
# finished. Override with AVLA_RENDER=<px> sbatch scripts/lang_diag.sbatch.
# batch_size=10 runs 10 envs in parallel (async subprocesses), hence 12 CPUs.
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-langdiag
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
export AVLA_RENDER="${AVLA_RENDER:-256}"
bash "${AVLA_REPO:-$SLURM_SUBMIT_DIR}/scripts/lang_diag.sh" 8,9,7 cycle 10
