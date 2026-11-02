#!/bin/bash
# Stand-in for sbatch on a machine without Slurm: runs a launcher (*.sbatch) here and now, then
# returns. Jobs run one after another in the order they are submitted.
#
# Usage: bash scripts/run_local.sh [--option=value ...] LAUNCHER [ARGS ...]
#        export AVLA_SBATCH=scripts/run_local.sh   # makes the chain scripts in experiments/replay use it
# sbatch options are accepted in --option=value form and ignored, except --parsable (print only the
# job id). A --dependency on an earlier job is met by construction, and a failed job stops the
# chain: this script exits with the job's status. The job id is numeric (milliseconds since 1970),
# like a Slurm id, so run directories and labels keep their usual form.
# The launcher sees SLURM_JOB_ID, SLURM_SUBMIT_DIR (the current directory) and SLURM_CPUS_PER_TASK
# (its own #SBATCH --cpus-per-task, capped at this machine's cores). Its output goes to the file its
# #SBATCH --output line names (%j = job id), or to slurm-<id>.out here. No memory watchdog runs.
set -euo pipefail

PARSABLE=0
while [ $# -gt 0 ] && [[ "$1" == -* ]]; do
    case "$1" in
        --parsable) PARSABLE=1 ;;
        --*=*) ;;
        *) echo "run_local.sh: use --option=value forms; got '$1'" >&2; exit 2 ;;
    esac
    shift
done
LAUNCHER="${1:?usage: run_local.sh [--option=value ...] LAUNCHER [ARGS ...]}"; shift
[ -f "$LAUNCHER" ] || { echo "run_local.sh: no such launcher: $LAUNCHER" >&2; exit 2; }

ID=$(date +%s%3N)
CPUS=$(sed -n 's/^#SBATCH --cpus-per-task=\([0-9]*\).*/\1/p' "$LAUNCHER" | head -1)
NPROC=$(nproc)
CPUS=${CPUS:-1}; [ "$CPUS" -le "$NPROC" ] || CPUS=$NPROC
LOG=$(sed -n 's/^#SBATCH --output=\([^ ]*\).*/\1/p' "$LAUNCHER" | head -1)
LOG=${LOG:-slurm-%j.out}; LOG=${LOG//%j/$ID}
mkdir -p "$(dirname "$LOG")"

echo "== run_local: $LAUNCHER $* as job $ID, $CPUS CPUs, log $LOG" >&2
status=0
SLURM_JOB_ID=$ID SLURM_SUBMIT_DIR="$PWD" SLURM_CPUS_PER_TASK=$CPUS \
    bash "$LAUNCHER" "$@" > "$LOG" 2>&1 || status=$?
echo "== run_local: job $ID finished with status $status" >&2
if [ "$PARSABLE" = 1 ]; then echo "$ID"; else echo "Submitted batch job $ID (ran locally)"; fi
exit $status
