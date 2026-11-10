#!/bin/bash
# Repeatability follow-up (E54): 116056 found two identical evals differing in their observations
# from reset on, never first in the policy. Two more protocol evals of θ0 (LIBERO-Spatial task 1 x 10)
# with per-part observation hashes and the simulator state (qpos, qvel) logged at every step, to
# tell physics from rendering: is the simulator state already different, or only the images?
# Slurm settings: a template from the cluster these runs used. Set the partition
# here or with SBATCH_PARTITION (README, "Running on Slurm").
#SBATCH --job-name=avla-repeat-parts
#SBATCH --partition=<your-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G   # 10 simulator processes at about 2.4 GB RSS each, plus the policy process
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
set -uo pipefail
cd "${AVLA_REPO:-$SLURM_SUBMIT_DIR}"
[ -f scripts/env.sh ] || { echo "not a repo root: $PWD" >&2; exit 1; }
set +u; source scripts/env.sh; set -u
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" AVLA_SUBGOALS=1 AVLA_STEPLOG=1
SUITE=libero_spatial; JOB="$SLURM_JOB_ID"; OUT="$AVLA_RUNS/eval/$SUITE"
for run in E F; do
    SLURM_JOB_ID="$JOB-$run" bash scripts/eval.sh "$SUITE" "[1]" 10 || echo "== run $run FAILED"
done
echo "== compare E vs F"
python experiments/setup/repeatability_compare.py "$OUT/$JOB-E-r256" "$OUT/$JOB-F-r256" --out "$OUT/$JOB-compare-EF.json"
python - "$OUT/$JOB-compare-EF.json" <<'EOF'
import json, sys
for e in json.load(open(sys.argv[1]))["episodes_detail"]:
    print(e["init_state"], e["kind"], e["step"], "sim differs:", e.get("sim_state_differs"),
          "first sim diff:", e.get("first_sim_diff"), "parts:", e.get("parts_differ"))
EOF
echo "REPEATABILITY PARTS DONE $(date -Is)"
