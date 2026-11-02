#!/bin/bash
# Language-check plumbing on a CPU node. Not a result.
# 1) env selftest for all 9 cells of the proposed design (tasks 8,9,7, cyclic pairs), no policy;
# 2) θ0 on one cell per condition (A=8, B=9), 1 episode each: true, blank, swapped.
# About 36 min per episode on 16 cores. Body: scripts/lang_diag.sh.
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-langdiag-cpu
#SBATCH --partition=<your-partition>
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:30:00
#SBATCH --output=slurm-%j.out
export OMP_NUM_THREADS=16
export AVLA_DEVICE=cpu
bash "${AVLA_REPO:-$SLURM_SUBMIT_DIR}/scripts/lang_diag.sh" 8,9,7 cycle 0
bash "${AVLA_REPO:-$SLURM_SUBMIT_DIR}/scripts/lang_diag.sh" 8 8:9 1
