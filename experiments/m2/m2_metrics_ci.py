"""M2 metrics with bootstrap CIs: NBT, normalized forgetting, final probes.

Reads results/runs.csv (counts per eval run x task). Episodes are 0/1, so each cell is rebuilt as a
vector of n_success ones and n_episodes - n_success zeros and resampled by metrics.bootstrap_ci
(episode noise only; seed spread is printed separately). Normalized forgetting is given twice: last
row from the stage-5 per-stage evals (x 20) and from the final-checkpoint evals (x 50).

  python experiments/m2/m2_metrics_ci.py            (torch-free; runs in seconds on a CPU node)

Also writes results/m2_metrics_ci.csv (one row per condition, seed and metric: estimate and 95%
interval), adding two quantities: retention = final success / θ0's on the final-probe protocol, with
θ0's episodes resampled too, per seed and for the seed mean; and the learning cost, success on the task
just learned minus the mean of the no-replay references (Sequential and M1's single-task fine-tunes
from θ0, E48), per task and averaged over the five tasks. All intervals are episode noise only.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from adaptive_vla.metrics import bootstrap_ci, nbt, normalized_forgetting  # noqa: E402

TASKS = [26, 18, 23, 27, 34]
RUNS = {   # condition: training job of each stage (stage 1 first)
    "Sequential": [113804, 115603, 115604, 115605, 115606],
    "ER k14 b0": [115608, 115620, 115621, 115622, 115623],
    "ER k14 b1": [115626, 115627, 115628, 115629, 115630],
    "ER k14 b2": [115632, 115633, 115634, 115635, 115636],
    "ER k140 b0": [115609, 115758, 115759, 115760, 115761],
    "ER k140 b1": [115786, 115787, 115788, 115789, 115790],     # E42
    "ER kall b0": [115611, 115792, 115793, 115794, 115795],     # E43, full replay
    "ER kall b1": [115815, 115816, 115817, 115818, 115819],     # E46, full replay seed 2
}
# Final probes and just-learned evals only (no per-stage probes, so no NBT or normalized forgetting).
RUNS_FINAL_ONLY = {
    "ER p2.05 b0": [115873, 115874, 115875, 115876, 115877],   # E50, 2.05% of every task's frames
}
T975 = {1: 12.706, 2: 4.303, 3: 3.182}   # t(0.975, df)
PROBES = [("libero_spatial", None), ("libero_goal", None), ("libero_10", {2, 3, 9})]
# θ0 on the final-probe protocol, every run of it pooled: E41, the M1 Goal run (10 x 20) and the M1
# LIBERO-10 runs (x 50; 113801 for task 3). Spatial has only E41. Episodes per task must match the probe.
THETA0_FINAL_RUNS = {"115785-theta0-final-r256", "113038-goal-r256", "113038-l10kitchen-r256", "113801-r256"}
THETA0_EPISODES = {"libero_spatial": 20, "libero_goal": 20, "libero_10": 50}
# No-replay references for learning (E48a): M1 single-task fine-tunes from θ0, by task.
M1_REF = {18: 113803, 23: 113795, 27: 113805, 34: 113806}
N_BOOT = 2000
OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "results", "m2_metrics_ci.csv")


def job_of(policy: str) -> int:
    parts = policy.split("/")
    return int(parts[1]) if len(parts) > 2 and parts[0] == "train" and parts[1].isdigit() else -1


def ones(s: int, n: int) -> np.ndarray:
    return np.r_[np.ones(s), np.zeros(n - s)]


def boot_mean(vecs, rng):
    """One bootstrap draw of the pooled success rate, episodes resampled within each task."""
    return np.mean(np.concatenate([rng.choice(v, len(v)) for v in vecs]))


def interval(draws):
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def report_rows(cells, theta0, rng):
    """Rows of results/m2_metrics_ci.csv: final probes, retention and learning cost, with 95% intervals."""
    rows = []

    def add(cond, seed, metric, est, lo, hi, n):
        rows.append({"condition": cond, "seed": seed, "metric": metric, "estimate": round(est, 4),
                     "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4), "n_episodes": n})

    budgets = defaultdict(list)                        # "ER k14" -> [(seed label, jobs)]
    # Conditions added later draw from their own generator, so earlier intervals stay as published.
    rng_extra = np.random.default_rng(2)
    extra = {c.rsplit(" ", 1)[0] for c in RUNS_FINAL_ONLY}
    for cond, jobs in {**RUNS, **RUNS_FINAL_ONLY}.items():
        if cond != "Sequential":
            budgets[cond.rsplit(" ", 1)[0]].append((cond.rsplit(" ", 1)[1], jobs))
    for suite, ids in PROBES:
        t0 = [ones(*theta0[(suite, t)]) for t in range(10) if (ids is None or t in ids) and (suite, t) in theta0]
        p0 = np.mean(np.concatenate(t0))
        t0_draws = np.array([boot_mean(t0, rng) for _ in range(N_BOOT)])
        add("theta0", "", f"final_{suite}", p0, *interval(t0_draws), sum(map(len, t0)))
        for budget, seeds in budgets.items():
            seed_draws, seed_points, n_all = [], [], 0
            for seed, jobs in seeds:
                vecs = [ones(*cells[(jobs[-1], suite, t, True)]) for t in range(10)
                        if (ids is None or t in ids) and cells[(jobs[-1], suite, t, True)][1]]
                p = np.mean(np.concatenate(vecs))
                g = rng_extra if budget in extra else rng
                d = np.array([boot_mean(vecs, g) for _ in range(N_BOOT)])
                n = sum(map(len, vecs))
                n_all += n
                add(budget, seed, f"final_{suite}", p, *interval(d), n)
                add(budget, seed, f"retention_{suite}", p / p0, *interval(d / t0_draws), n)
                seed_draws.append(d)
                seed_points.append(p)
            m, md = np.mean(seed_points), np.mean(seed_draws, axis=0)
            add(budget, "mean", f"final_{suite}", m, *interval(md), n_all)
            add(budget, "mean", f"retention_{suite}", m / p0, *interval(md / t0_draws), n_all)

    # learning cost: just-learned success (x 50) minus the mean no-replay reference, per task
    seq = RUNS["Sequential"]
    refs = []
    for i, t in enumerate(TASKS):
        r = [ones(*cells[(seq[i], "libero_90", t, False)])]
        if t in M1_REF:
            r.append(ones(*cells[(M1_REF[t], "libero_90", t, False)]))
        assert all(len(v) == 50 for v in r), (t, [len(v) for v in r])
        refs.append(r)
    ref_draws = np.array([[np.mean([rng.choice(v, len(v)).mean() for v in r]) for r in refs] for _ in range(N_BOOT)])
    ref_point = np.array([np.mean([v.mean() for v in r]) for r in refs])
    for i, t in enumerate(TASKS):
        add("no replay", "reference", f"learned_task{t}", ref_point[i], *interval(ref_draws[:, i]),
            sum(len(v) for v in refs[i]))
    for budget, seeds in budgets.items():
        diag = [[ones(*cells[(jobs[i], "libero_90", t, False)]) for _, jobs in seeds] for i, t in enumerate(TASKS)]
        assert all(len(v) == 50 for d in diag for v in d), budget
        point = np.array([np.mean([v.mean() for v in d]) for d in diag])
        g = rng_extra if budget in extra else rng
        draws = np.array([[np.mean([g.choice(v, len(v)).mean() for v in d]) for d in diag] for _ in range(N_BOOT)])
        for i, t in enumerate(TASKS):
            add(budget, "mean", f"learned_task{t}", point[i], *interval(draws[:, i]), 50 * len(seeds))
            add(budget, "mean", f"learning_cost_task{t}", point[i] - ref_point[i],
                *interval(draws[:, i] - ref_draws[:, i]), 50 * len(seeds))
        add(budget, "mean", "learning_cost_mean", float(np.mean(point - ref_point)),
            *interval((draws - ref_draws).mean(axis=1)), 250 * len(seeds))
    return rows


def main() -> None:
    cells = defaultdict(lambda: [0, 0])   # (job, suite, task_id, final?) -> [success, episodes]
    theta0 = {}                           # (suite, task_id) -> (success, episodes), final-probe protocol
    with open("results/runs.csv") as f:
        for r in csv.DictReader(f):
            key = (job_of(r["policy"]), r["suite"], int(r["task_id"]), "final" in r["run"])
            cells[key][0] += int(r["n_success"])
            cells[key][1] += int(r["n_episodes"])
            st = (r["suite"], int(r["task_id"]))
            if (r["run"] in THETA0_FINAL_RUNS and int(r["n_episodes"]) == THETA0_EPISODES.get(st[0])
                    and (st[0] != "libero_10" or st[1] in {2, 3, 9})):
                s, n = theta0.get(st, (0, 0))
                theta0[st] = (s + int(r["n_success"]), n + int(r["n_episodes"]))

    csv_rows = []
    print(f"{'condition':12s} {'NBT':>18s} {'NF (stage-5 x20)':>20s} {'NF (final x50)':>20s}")
    for cond, jobs in RUNS.items():
        k = len(TASKS)
        out = np.full((k, k), None, dtype=object)
        out_final = np.full((k, k), None, dtype=object)
        for i, job in enumerate(jobs):
            for j in range(i + 1):
                s, n = cells[(job, "libero_90", TASKS[j], False)]
                assert n > 0, (cond, i, j)
                out[i, j] = out_final[i, j] = ones(s, n)
        for j in range(k - 1):            # last row from the final-checkpoint evals, if they exist
            s, n = cells[(jobs[-1], "libero_90", TASKS[j], True)]
            if n:
                out_final[k - 1, j] = ones(s, n)
        has_final = all(out_final[k - 1, j] is not None and len(out_final[k - 1, j]) == 50 for j in range(k - 1))
        cond_label, seed_label = (cond, "") if cond == "Sequential" else cond.rsplit(" ", 1)
        for metric, mat, ok in (("nbt", out, True), ("norm_forgetting_stage5_x20", out, True),
                                ("norm_forgetting_final_x50", out_final, has_final)):
            if ok:
                est, lo, hi = bootstrap_ci(nbt if metric == "nbt" else normalized_forgetting, mat)
                csv_rows.append({"condition": cond_label, "seed": seed_label, "metric": metric,
                                 "estimate": round(est, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                                 "n_episodes": int(sum(len(v) for v in mat.flat if v is not None))})
        fmt = "{:.2f} [{:.2f}, {:.2f}]"
        row = [fmt.format(*bootstrap_ci(nbt, out)), fmt.format(*bootstrap_ci(normalized_forgetting, out)),
               fmt.format(*bootstrap_ci(normalized_forgetting, out_final)) if has_final else "n/a"]
        print(f"{cond:12s} {row[0]:>18s} {row[1]:>20s} {row[2]:>20s}")
        for i in range(k):                # the success matrix itself, % (rows: after stage i + 1)
            print(" " * 14 + " ".join(f"{100 * out[i, j].mean():5.0f}" if j <= i else "     " for j in range(k)))

    print("\nFinal-checkpoint probes, success % [95% CI, episode noise]:")
    rng = np.random.default_rng(0)
    seeds = defaultdict(list)
    for cond, jobs in {**RUNS, **RUNS_FINAL_ONLY}.items():
        parts = []
        for suite, ids in PROBES:
            per_task = [cells[(jobs[-1], suite, t, True)] for t in range(10) if ids is None or t in ids]
            per_task = [c for c in per_task if c[1]]
            if not per_task:
                parts.append(f"{suite} n/a")
                continue
            vecs = [ones(s, n) for s, n in per_task]
            p = np.mean(np.concatenate(vecs))
            boot = [np.mean(np.concatenate([rng.choice(v, len(v)) for v in vecs])) for _ in range(2000)]
            lo, hi = np.quantile(boot, [0.025, 0.975])
            parts.append(f"{suite} {100 * p:.1f} [{100 * lo:.1f}, {100 * hi:.1f}] (n={sum(len(v) for v in vecs)})")
            seeds[(cond.rsplit(' ', 1)[0], suite)].append(100 * p)
        print(f"  {cond:12s} " + "; ".join(parts))

    print("\nSeed spread (conditions with more than one seed): mean, SD, 95% t-interval of the mean")
    for (cond, suite), v in seeds.items():
        if len(v) > 1:
            v = np.array(v)
            sd = v.std(ddof=1)
            half = T975[len(v) - 1] * sd / np.sqrt(len(v))
            print(f"  {cond:10s} {suite:15s} n {len(v)}, mean {v.mean():.1f}, SD {sd:.1f}, "
                  f"[{v.mean() - half:.1f}, {v.mean() + half:.1f}]")


    csv_rows += report_rows(cells, theta0, np.random.default_rng(1))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nwrote {OUT_CSV} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
