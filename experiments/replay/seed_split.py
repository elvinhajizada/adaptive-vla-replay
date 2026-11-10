"""E61: at 0.2% (k = 14), is the seed spread from the buffer seed or from the training seed?

A 2 x 2 of final checkpoints: buffer seed {0, 1} x training seed {1000, 1001}. Two cells are E37's
seeds 1 and 2 (both seeds changed together), two are the crossed runs. The buffer seed sets the
whole data stream: which frames are buffered, which replay frames fill each batch, and the per-epoch
order of the new task's frames (adaptive_vla.replay.MixedSampler). The training seed sets the rest of
training's randomness, chiefly the noise and time draws of the flow-matching loss.

Reads results/runs.csv. Per cell: final success on Spatial and Goal (10 x 20), LIBERO-10 tasks 2, 3, 9
(x 50) and the earlier new tasks (x 50). Main effect of a factor = mean over the other factor of the
difference between its two levels; 95% intervals resample episodes within each task and cell (episode
noise only: one run per cell, so the interaction cannot be told apart from run-to-run noise).
Secondary: per-task agreement (Spearman ρ over the 20 Spatial and Goal tasks) between cells that share
the buffer seed and between cells that share the training seed.

  python experiments/replay/seed_split.py      (torch-free, seconds on a CPU node)

Writes results/seed_split.csv.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "seed_split.csv")
# (buffer seed, training seed) -> stage-5 training job of the k = 14 sequence
CELLS = {(0, 1000): 115623, (1, 1001): 115630,    # E37 seeds 1 and 2
         (0, 1001): 116069, (1, 1000): 116075}    # crossed (E61)
SUITES = {"spatial": ("libero_spatial", range(10)), "goal": ("libero_goal", range(10)),
          "libero10": ("libero_10", (2, 3, 9)), "new_tasks": ("libero_90", (26, 18, 23, 27))}
N_BOOT = 4000


def job_of(policy: str) -> int:
    parts = policy.split("/")
    return int(parts[1]) if len(parts) > 2 and parts[0] == "train" and parts[1].isdigit() else -1


def ones(s: int, n: int) -> np.ndarray:
    return np.r_[np.ones(s), np.zeros(n - s)]


def ranks(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):                      # average ranks for ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b) -> float:
    return float(np.corrcoef(ranks(np.asarray(a, float)), ranks(np.asarray(b, float)))[0, 1])


def main() -> None:
    cells = defaultdict(lambda: [0, 0])          # (job, suite, task) -> [success, episodes], final probes
    with open(os.path.join(ROOT, "results", "runs.csv")) as f:
        for r in csv.DictReader(f):
            if "final" in r["run"]:
                key = (job_of(r["policy"]), r["suite"], int(r["task_id"]))
                cells[key][0] += int(r["n_success"])
                cells[key][1] += int(r["n_episodes"])

    vecs = {}                                    # (cell, metric) -> per-task episode vectors
    for cell, job in CELLS.items():
        for m, (suite, tasks) in SUITES.items():
            v = [ones(*cells[(job, suite, t)]) for t in tasks]
            if any(len(x) == 0 for x in v):
                sys.exit(f"missing final probe: cell {cell}, job {job}, {suite}")
            vecs[(cell, m)] = v

    rng = np.random.default_rng(0)
    point = {k: np.mean(np.concatenate(v)) for k, v in vecs.items()}
    draws = {k: np.array([np.mean(np.concatenate([rng.choice(x, len(x)) for x in v])) for _ in range(N_BOOT)])
             for k, v in vecs.items()}
    for cell in CELLS:                           # derived metric: Spatial minus Goal
        point[(cell, "spatial_minus_goal")] = point[(cell, "spatial")] - point[(cell, "goal")]
        draws[(cell, "spatial_minus_goal")] = draws[(cell, "spatial")] - draws[(cell, "goal")]

    rows = []
    metrics = ["spatial", "goal", "spatial_minus_goal", "libero10", "new_tasks"]
    print("Cells, final success % (buffer seed, training seed):")
    for cell in CELLS:
        print(f"  {cell}: " + ", ".join(f"{m} {100 * point[(cell, m)]:.1f}" for m in metrics))
        for m in metrics:
            lo, hi = np.quantile(draws[(cell, m)], [0.025, 0.975])
            rows.append({"quantity": "cell", "buffer_seed": cell[0], "training_seed": cell[1], "metric": m,
                         "estimate": round(point[(cell, m)], 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)})

    def effect(src, m):                          # main effect, level A minus level B, mean over the other factor
        if src == "buffer":
            pairs = [((0, t), (1, t)) for t in (1000, 1001)]
        else:
            pairs = [((b, 1000), (b, 1001)) for b in (0, 1)]
        return (np.mean([point[(a, m)] - point[(b, m)] for a, b in pairs]),
                np.mean([draws[(a, m)] - draws[(b, m)] for a, b in pairs], axis=0))

    print("\nMain effects, pts [95% interval, episode noise] (buffer: 0 minus 1; training: 1000 minus 1001):")
    for m in metrics:
        parts = []
        for src in ("buffer", "training"):
            est, d = effect(src, m)
            lo, hi = np.quantile(d, [0.025, 0.975])
            parts.append(f"{src} {100 * est:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
            rows.append({"quantity": f"main_effect_{src}", "buffer_seed": "", "training_seed": "", "metric": m,
                         "estimate": round(est, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)})
        print(f"  {m:20s} " + "; ".join(parts))

    # Per-task agreement over the 20 Spatial and Goal tasks
    def per_task(cell):
        job = CELLS[cell]
        out = []
        for suite in ("libero_spatial", "libero_goal"):
            out += [cells[(job, suite, t)][0] / cells[(job, suite, t)][1] for t in range(10)]
        return out

    same_buffer = [spearman(per_task((b, 1000)), per_task((b, 1001))) for b in (0, 1)]
    same_training = [spearman(per_task((0, t)), per_task((1, t))) for t in (1000, 1001)]
    print(f"\nPer-task agreement, Spearman ρ over 20 Spatial and Goal tasks: same buffer seed "
          f"{same_buffer[0]:.2f}, {same_buffer[1]:.2f}; same training seed {same_training[0]:.2f}, {same_training[1]:.2f}")
    for name, vals in (("same_buffer", same_buffer), ("same_training", same_training)):
        for i, v in enumerate(vals):
            rows.append({"quantity": f"spearman_{name}", "buffer_seed": "" if name == "same_training" else i,
                         "training_seed": "" if name == "same_buffer" else (1000, 1001)[i], "metric": "per_task_spatial_goal",
                         "estimate": round(v, 4), "ci95_lo": "", "ci95_hi": ""})

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
