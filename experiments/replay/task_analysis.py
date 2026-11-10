"""E45: where does replay fail? Failure anatomy of two-step tasks and skill margin (torch-free).

  python experiments/replay/task_analysis.py [results/runs.csv]

Reads the final-checkpoint probes of the replay sequences (rows whose run label contains "final",
keyed by the stage-5 checkpoint) and θ0's reference runs from results/runs.csv. Writes
results/failure_anatomy.csv and results/skill_margin.csv and prints both summaries.

(a) Failure anatomy: for task 23 and LIBERO-10 tasks 2, 3, 9 (two goal predicates each), the share
    of episodes where each predicate holds at the end, in the order a demo achieves them (`order` =
    demo); θ0 from E41, and for task 3 from 113801 (x 50 with subgoals). Also the far-scene LIBERO-10
    tasks 0, 1, 4, 6, 7 of E47 (x 20 per checkpoint), predicates in the order the instruction names
    them (`order` = instruction; the basket tasks have no fixed demo order).
(b) Skill margin: per Spatial and Goal task, θ0's success in the setup runs (the margin: 113024
    Spatial 10 x 10, 113038 Goal 10 x 20) against retention = final success / θ0's success in the
    E41 reference run (a different run, so their noise is independent). Spearman ρ between margin
    and the mean retention over the replay runs, with a permutation test over tasks (20 points).
"""
import csv
import os
import random
import sys
from collections import defaultdict

import numpy as np

# Stage-5 training job -> condition (E37, E39, E42, E43, E46).
FINAL = {"115623": "k14-b0", "115630": "k14-b1", "115636": "k14-b2", "115761": "k140-b0",
         "115790": "k140-b1", "115795": "kall-b0", "115819": "kall-b1"}
# E50 and E51 (proportional 2.05% split, two seeds): failure anatomy only, kept out of the margin analysis.
ANATOMY_ONLY = {"115877": "kp2.05-b0", "116053": "kp2.05-b1"}
# Goal predicates of the two-step tasks in the order the demos achieve them.
STEPS = {
    ("libero_90", 23): ["close white_cabinet_1_bottom_region", "open white_cabinet_1_top_region"],
    ("libero_10", 2): ["turnon flat_stove_1", "on moka_pot_1 flat_stove_1_cook_region"],
    ("libero_10", 3): ["in akita_black_bowl_1 white_cabinet_1_bottom_region", "close white_cabinet_1_bottom_region"],
    ("libero_10", 9): ["in white_yellow_mug_1 microwave_1_heating_region", "close microwave_1"],
}
# E47: far-scene two-step LIBERO-10 tasks, predicates in the order the instruction names them.
FAR = {
    0: ["in alphabet_soup_1 basket_1_contain_region", "in tomato_sauce_1 basket_1_contain_region"],
    1: ["in cream_cheese_1 basket_1_contain_region", "in butter_1 basket_1_contain_region"],
    4: ["on porcelain_mug_1 plate_1", "on white_yellow_mug_1 plate_2"],
    6: ["on porcelain_mug_1 plate_1", "on chocolate_pudding_1 living_room_table_plate_right_region"],
    7: ["in alphabet_soup_1 basket_1_contain_region", "in cream_cheese_1 basket_1_contain_region"],
}
E47 = "e47-farscene-"                  # run label prefix part; suffix = theta0 or the condition
MARGIN_RUNS = {"libero_spatial": "113024-r256", "libero_goal": "113038-goal-r256"}
THETA0_REF = "theta0-final"            # E41 run label part
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")


def condition(row):
    if E47 in row["run"]:
        lab = row["run"].split(E47, 1)[1].rsplit("-r256", 1)[0]
        return "theta0-E47" if lab == "theta0" else lab
    if row["policy"] == "theta0":
        if row["run"] == "113801-r256":
            return "theta0-113801"
        return "theta0-E41" if THETA0_REF in row["run"] else None
    parts = row["policy"].split("/")
    return {**FINAL, **ANATOMY_ONLY}.get(parts[1]) if "final" in row["run"] and len(parts) > 1 else None


def subgoals(s):
    return {k: float(v) for k, v in (p.rsplit("=", 1) for p in s.split(";") if p)} if s else {}


def rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):                      # average ranks for ties
        r[x == v] = r[x == v].mean()
    return r


def spearman(x, y):
    return float(np.corrcoef(rank(np.asarray(x, float)), rank(np.asarray(y, float)))[0, 1])


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "runs.csv")
    rows = list(csv.DictReader(open(path)))
    final = defaultdict(dict)                   # condition -> (suite, task) -> row
    margin = {}
    for r in rows:
        key = (r["suite"], int(r["task_id"]))
        c = condition(r)
        if c:
            final[c][key] = r
        if MARGIN_RUNS.get(r["suite"]) == r["run"]:
            margin[key] = int(r["n_success"]) / int(r["n_episodes"])
    conds = [c for c in ["theta0-E41", *FINAL.values()] if c in final]
    anat_conds = [c for c in ["theta0-E41", "theta0-113801", "theta0-E47", *FINAL.values(), *ANATOMY_ONLY.values()] if c in final]
    tasks = [(key, preds, "demo") for key, preds in STEPS.items()] + \
        [(("libero_10", t), preds, "instruction") for t, preds in FAR.items()]

    # (a) failure anatomy
    anat = []
    print("(a) two-step tasks: success, then each step in demo or instruction order (share of episodes at the end)")
    for key, preds, order in tasks:
        for c in anat_conds:
            r = final[c].get(key)
            if not r:
                continue
            g = subgoals(r["subgoals_final"])
            vals = [g.get(p) for p in preds]
            anat.append({"condition": c, "suite": key[0], "task_id": key[1], "n_episodes": r["n_episodes"],
                         "success": r["success_pct"], "order": order, "step1": preds[0], "step1_share": vals[0],
                         "step2": preds[1], "step2_share": vals[1]})
            fmt = lambda v: "  n/a" if v is None else f"{100 * v:5.0f}"  # noqa: E731
            print(f"  {key[0]:10s} {key[1]:2d} {c:11s} success {float(r['success_pct']):5.0f}"
                  f"  step 1 {fmt(vals[0])}  step 2 {fmt(vals[1])}")
    with open(os.path.join(OUT, "failure_anatomy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(anat[0]))
        w.writeheader()
        w.writerows(anat)

    # (b) skill margin
    ref = "theta0-E41" if "theta0-E41" in final else None
    replay = [c for c in conds if not c.startswith("theta0")]
    out, xs, ys = [], [], []
    for key in sorted(margin):
        den = final[ref][key] if ref and key in final[ref] else None
        d = int(den["n_success"]) / int(den["n_episodes"]) if den else margin[key]
        rets = {c: int(final[c][key]["n_success"]) / int(final[c][key]["n_episodes"]) / d
                for c in replay if key in final[c] and d > 0}
        if not rets:
            continue
        row = {"suite": key[0], "task_id": key[1], "theta0_margin": round(margin[key], 3),
               "theta0_ref": round(d, 3), "ref_source": "E41" if den else "margin run (coupled)",
               **{f"retention_{c}": round(v, 3) for c, v in rets.items()},
               "retention_mean": round(float(np.mean(list(rets.values()))), 3)}
        out.append(row)
        xs.append(margin[key])
        ys.append(row["retention_mean"])
    rho = spearman(xs, ys)
    rng = random.Random(0)
    perm = [spearman(rng.sample(xs, len(xs)), ys) for _ in range(10000)]
    p = (1 + sum(abs(v) >= abs(rho) for v in perm)) / (1 + len(perm))
    per_run = {c: spearman([margin[(o["suite"], o["task_id"])] for o in out if f"retention_{c}" in o],
                           [o[f"retention_{c}"] for o in out if f"retention_{c}" in o]) for c in replay}
    print(f"\n(b) {len(out)} Spatial + Goal tasks; retention denominator: {'E41' if ref else 'margin run (coupled!)'}")
    print(f"  Spearman ρ(θ0 margin, mean retention) = {rho:+.2f}, permutation p = {p:.3f}")
    print("  per run: " + ", ".join(f"{c} {v:+.2f}" for c, v in per_run.items()))
    fields = list(dict.fromkeys(k for o in out for k in o))
    with open(os.path.join(OUT, "skill_margin.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
