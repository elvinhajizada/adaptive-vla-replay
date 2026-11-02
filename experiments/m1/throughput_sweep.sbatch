#!/bin/bash
# M1 throughput: theta0 eval throughput sweep on one A100 (batch 10/20/50, CPU count, parallel tasks).
# 150 episodes in total. Body: experiments/m1/throughput_sweep.sh.
# 48 Slurm CPUs = 24 physical cores (the node counts hyperthreads: 96 CPUs = 48 cores), so batch
# 50 is oversubscribed on purpose: it shows where the CPU side saturates.
# Slurm settings: a template from the cluster these runs used. Set the partition
# (and account or QOS if your site needs them) before submitting.
#SBATCH --job-name=avla-throughput
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out
bash "${AVLA_REPO:-$SLURM_SUBMIT_DIR}/experiments/m1/throughput_sweep.sh"
