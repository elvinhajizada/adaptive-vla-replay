#!/bin/bash
# E38: theta0 on LIBERO-Object at the eval protocol (render 256), to fill the reference table in
# docs/LIBERO.md section 7. Not a probe: the probe set stays as fixed before any replay result.
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-eval-obj256
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G   # Spatial at 10 episodes per task peaked at 30.6 GB RSS (job 113024)
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
bash "${AVLA_REPO:-$SLURM_SUBMIT_DIR}/scripts/eval.sh" libero_object all 10
