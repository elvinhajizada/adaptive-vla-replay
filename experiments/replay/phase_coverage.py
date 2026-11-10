"""E48b: are the step-2 frames of LIBERO-10 tasks under-represented in the replay buffers? (CPU)

  python experiments/replay/phase_coverage.py SPEC [SPEC ...]      SPEC = label=path/to/stage-5 replay spec

Step 2 is read off the gripper (θ0 convention, action[6] < 0 open, > 0 closed), per demo:
  - tasks 3 and 9 (put X in, then close) and the living-room tasks 0, 1, 4, 6, 7 (two pick-and-place
    steps): rows after the first grasp-and-release (second toggle), in demo order;
  - task 2 (turn on the stove, then the moka pot): rows from the last closing of the gripper.
Prints the toggle-count distribution (to check the rule fits the demos), the step-2 share of all
demo frames, and per buffer the step-2 share and count among its frames and the demos it covers.
Writes results/phase_coverage.csv (one row per task and buffer).
"""
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from adaptive_vla.libero_data import frame_groups  # noqa: E402

ROOT = os.path.join(os.environ["AVLA_DSS"], "data", "libero_all_theta0")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "results", "phase_coverage.csv")
TASKS = {2: "turn on the stove and put the moka pot on it",
         3: "put the black bowl in the bottom drawer of the cabinet and close it",
         9: "put the yellow and white mug in the microwave and close it",
         0: "put both the alphabet soup and the tomato sauce in the basket",
         1: "put both the cream cheese box and the butter in the basket",
         4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
         6: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
         7: "put both the alphabet soup and the cream cheese box in the basket"}


def step2_rows(task, rows, grip):
    """Global row indices of one demo's step 2, and the demo's toggle count."""
    g = grip[rows] > 0
    tog = np.flatnonzero(g[1:] != g[:-1]) + 1
    if task == 2:
        closes = [t for t in tog if g[t]]
        start = closes[-1] if closes else len(rows)
    else:
        start = tog[1] if len(tog) >= 2 else len(rows)
    return set(rows[start:].tolist()), len(tog)


def main() -> None:
    specs = dict(a.split("=", 1) for a in sys.argv[1:])
    groups = frame_groups(ROOT)
    t = pq.read_table(os.path.join(ROOT, "data", "chunk-000", "file-000.parquet"), columns=["index", "action"])
    assert (t.column("index").to_numpy() == np.arange(t.num_rows)).all(), "rows are not in global index order"
    grip = np.stack(t.column("action").to_numpy(zero_copy_only=False))[:, 6]
    bufs = {lab: json.load(open(p))["replay"] for lab, p in specs.items()}
    out = []
    for tid, instr in TASKS.items():
        key = f"libero_10:{instr}"
        demos = [np.arange(r.start, r.stop) for r in groups[key]]
        step2, ntog, per_demo = set(), Counter(), []
        for rows in demos:
            s, n = step2_rows(tid, rows, grip)
            step2 |= s
            ntog[n] += 1
            per_demo.append(s)
        total = sum(len(r) for r in demos)
        base = len(step2) / total
        print(f"\nLIBERO-10 task {tid}: {len(demos)} demos, {total} frames; toggles per demo {dict(sorted(ntog.items()))}")
        print(f"  step 2 = {len(step2)} frames, base share {base:.2f}")
        for lab, rep in bufs.items():
            sel = rep[key]
            hit = [r for r in sel if r in step2]
            demos_cov = sum(1 for s in per_demo if s & set(sel))
            print(f"  {lab}: {len(sel)} frames, step 2 {len(hit)} ({len(hit) / len(sel):.2f} vs base {base:.2f}); "
                  f"step 2 of {demos_cov} of {len(demos)} demos covered")
            out.append({"task_id": tid, "demos": len(demos), "frames": total, "step2_frames": len(step2),
                        "base_share": round(base, 4), "buffer": lab, "buffer_frames": len(sel),
                        "step2_in_buffer": len(hit), "step2_share_in_buffer": round(len(hit) / len(sel), 4),
                        "demos_with_step2_covered": demos_cov})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
