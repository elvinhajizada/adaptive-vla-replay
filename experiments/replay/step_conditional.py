"""Is the second step of a two-step task lost beyond what losing the first implies? (torch-free, CPU)

  python experiments/replay/step_conditional.py

Step-2 success needs step 1 on the kitchen tasks (stove on, then the pot on it; bowl in the drawer,
then close it; mug in the microwave, then close it), so step 2's share falls whenever step 1's does.
This compares retention of step 1, P(step 1), with retention of step 2 given step 1, P(step 2 |
step 1), per episode from each final probe's subgoals.jsonl (end-of-episode predicates). If small
budgets only lose step 1 more often and the loss compounds, the two retentions agree; a loss specific
to step 2 shows as a lower conditional retention. Far-scene tasks (E47) use instruction order; their
two goals can be reached in either order, so there the conditional reads as "the other goal, once
one is reached".

Retention = condition / θ0 on the same protocol. 95% intervals: bootstrap over episodes within each
(run, task) cell, θ0's episodes resampled too (2,000 draws). Writes results/step_conditional.csv.
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL = os.path.join(os.environ.get("AVLA_RUNS", ""), "eval", "libero_10")
OUT = os.path.join(ROOT, "results", "step_conditional.csv")
KITCHEN, FAR = (2, 3, 9), (0, 1, 4, 6, 7)
# Final-probe eval runs per condition (seeds pooled): kitchen tasks 2, 3, 9 and far-scene tasks (E47, E50).
RUNS = {
    "theta0": {"kitchen": ["115785-theta0-final-r256", "113801-r256"], "far": ["115841-e47-farscene-theta0-r256"]},
    "k14": {"kitchen": ["115638-er14-s5-final-redo-r256", "115639-er-k14-b1-t1001-s5-final-r256",
                        "115640-er-k14-b2-t1002-s5-final-r256"], "far": []},
    "k140": {"kitchen": ["115762-er-k140-b0-t1000-s5-final-r256", "115791-er-k140-b1-t1001-s5-final-r256"],
             "far": ["115842-e47-farscene-k140-b0-r256", "115843-e47-farscene-k140-b1-r256"]},
    "p2.05": {"kitchen": ["115878-er-kp2.05-b0-t1000-s5-final-r256"],
              "far": ["115879-er-kp2.05-b0-t1000-s5-final-far-r256"]},
    "kall": {"kitchen": ["115796-er-kall-b0-t1000-s5-final-r256", "115820-er-kall-b1-t1001-s5-final-r256"],
             "far": ["115844-e47-farscene-kall-b0-r256", "115845-e47-farscene-kall-b1-r256"]},
}
# Kitchen: demo order (as task_analysis.STEPS). Far: instruction order (as task_analysis.FAR).
ORDER = {
    2: ["turnon flat_stove_1", "on moka_pot_1 flat_stove_1_cook_region"],
    3: ["in akita_black_bowl_1 white_cabinet_1_bottom_region", "close white_cabinet_1_bottom_region"],
    9: ["in white_yellow_mug_1 microwave_1_heating_region", "close microwave_1"],
    0: ["in alphabet_soup_1 basket_1_contain_region", "in tomato_sauce_1 basket_1_contain_region"],
    1: ["in cream_cheese_1 basket_1_contain_region", "in butter_1 basket_1_contain_region"],
    4: ["on porcelain_mug_1 plate_1", "on white_yellow_mug_1 plate_2"],
    6: ["on porcelain_mug_1 plate_1", "on chocolate_pudding_1 living_room_table_plate_right_region"],
    7: ["in alphabet_soup_1 basket_1_contain_region", "in cream_cheese_1 basket_1_contain_region"],
}
N_BOOT = 2000


def load(run, tasks):
    """(task) -> list of episode arrays [step1, step2] (end-of-episode predicates)."""
    out = defaultdict(list)
    seen = set()
    with open(os.path.join(EVAL, run, "subgoals.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            t = e["task_id"]
            if t not in tasks or (t, e["init_state"]) in seen:
                continue
            seen.add((t, e["init_state"]))
            g = dict(zip(e["goals"], e["final"]))
            out[t].append([bool(g[ORDER[t][0]]), bool(g[ORDER[t][1]])])
    return out


def cells(cond, group):
    tasks = KITCHEN if group == "kitchen" else FAR
    cs = []                                      # list of (task, array n x 2)
    for run in RUNS[cond][group]:
        for t, eps in load(run, tasks).items():
            cs.append((t, np.array(eps, dtype=bool)))
    return cs


def stats(cs, task=None):
    x = np.concatenate([a for t, a in cs if task is None or t == task])
    s1, s2 = x[:, 0].mean(), x[:, 1].mean()
    both = (x[:, 0] & x[:, 1]).sum()
    s21 = both / x[:, 0].sum() if x[:, 0].sum() else np.nan
    return len(x), s1, s2, s21


def resample(cs, rng):
    return [(t, a[rng.integers(0, len(a), len(a))]) for t, a in cs]


def main() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for group in ("kitchen", "far"):
        ref = cells("theta0", group)
        tasks = KITCHEN if group == "kitchen" else FAR
        for cond in RUNS:
            cs = cells(cond, group)
            if not cs:
                continue
            for task in [None, *tasks]:
                n, s1, s2, s21 = stats(cs, task)
                _, r1_, r2_, r21_ = stats(ref, task)
                rec = {"group": group, "task": "pooled" if task is None else task, "condition": cond, "n_episodes": n,
                       "step1": round(s1, 3), "step2": round(s2, 3), "step2_given_step1": round(s21, 3)}
                if cond != "theta0":
                    pt = (s1 / r1_, s21 / r21_ if r21_ else np.nan)
                    draws = []
                    for _ in range(N_BOOT):
                        _, b1, _, b21 = stats(resample(cs, rng), task)
                        _, c1, _, c21 = stats(resample(ref, rng), task)
                        draws.append((b1 / c1 if c1 else np.nan, b21 / c21 if c21 else np.nan))
                    d = np.array(draws, dtype=float)
                    lo, hi = np.nanpercentile(d, 2.5, axis=0), np.nanpercentile(d, 97.5, axis=0)
                    dd = d[:, 0] - d[:, 1]
                    rec.update({"ret_step1": round(pt[0], 3), "ret_step1_lo": round(lo[0], 3),
                                "ret_step1_hi": round(hi[0], 3), "ret_step2_given_1": round(pt[1], 3),
                                "ret_step2_given_1_lo": round(lo[1], 3), "ret_step2_given_1_hi": round(hi[1], 3),
                                "diff_lo": round(np.nanpercentile(dd, 2.5), 3),
                                "diff_hi": round(np.nanpercentile(dd, 97.5), 3)})
                rows.append(rec)
                if task is None or cond == "theta0":
                    print(f"{group:7s} {str(rec['task']):6s} {cond:6s} n {n:3d}  step1 {s1:.2f}  step2 {s2:.2f}"
                          f"  step2|step1 {s21:.2f}" + (f"   ret step1 {rec['ret_step1']:.2f} "
                          f"[{rec['ret_step1_lo']:.2f}, {rec['ret_step1_hi']:.2f}]  ret step2|1 "
                          f"{rec['ret_step2_given_1']:.2f} [{rec['ret_step2_given_1_lo']:.2f}, "
                          f"{rec['ret_step2_given_1_hi']:.2f}]  diff [{rec['diff_lo']:.2f}, {rec['diff_hi']:.2f}]"
                          if cond != "theta0" else ""))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
