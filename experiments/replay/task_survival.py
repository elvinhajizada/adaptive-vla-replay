"""E56: why do some of θ0's tasks survive fine-tuning better than others? (analysis of existing runs)

  python experiments/replay/task_survival.py

Per Spatial and Goal task (20 tasks), retention = success / θ0's success in
- the single fine-tunes from θ0 (Table 2: task 23, task 23 at LR 1e-5, task 26, task 27; 10
  episodes per task; θ0 = the setup runs 113024 Spatial, 113038 Goal), and
- the seven replay sequences of results/skill_margin.csv (final probes, 20 episodes; θ0 = E41).
Then two questions:
(1) Is survival a property of the old task? Spearman ρ of per-task retention between runs: same
    condition, other seed (a reliability ceiling) vs other budget vs other new task. Kendall's W over
    all informative runs, with a permutation test.
(2) Which task properties go with survival? Spearman ρ (permutation p over the 20 tasks) of mean
    retention with: θ0's success, demo length, share of frames with the gripper closed, grasps per
    demo (open -> close switches), objects shared with the new tasks (BDDL objects of interest, by
    object class), and word overlap with the closest new-task instruction.
(3) Does demo length go with survival within each suite, not only across the two?
Writes results/task_survival.csv and results/length_within_suite.csv, and prints a summary.
Torch-free (numpy, pyarrow).
"""
import csv
import itertools
import os
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ["AVLA_DSS"]) / "data"
BDDL = Path(__import__("libero.libero", fromlist=["x"]).__file__).parent / "bddl_files"
SUITES = ("libero_spatial", "libero_goal")
# Single fine-tunes from θ0: label -> {suite: eval run}.
SINGLE = {"ft23": {"libero_spatial": "113802-ft23-1k-r256", "libero_goal": "113802-ft23-1k-r256"},
          "ft23-lr1e-5": {"libero_spatial": "113823-lr1e-5-r256", "libero_goal": "115949-lr1e-5-r256"},
          "ft26": {"libero_spatial": "114663-ft26-r256", "libero_goal": "114663-ft26-r256"},
          "ft27": {"libero_spatial": "115948-ft27-r256", "libero_goal": "114664-ft27-r256"}}
THETA0_SETUP = {"libero_spatial": "113024-r256", "libero_goal": "113038-goal-r256"}
NEW_TASKS = {18: "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
             23: "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
             26: "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
             27: "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
             34: "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug"}
SINGLE_NEW = {"ft23": 23, "ft23-lr1e-5": 23, "ft26": 26, "ft27": 27}


def ranks(x):
    x = np.asarray(x, float)
    order = x.argsort(kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):            # average ranks for ties
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def perm_p(a, b, n=10000, seed=0):
    rng, obs = np.random.default_rng(seed), abs(spearman(a, b))
    b = np.asarray(b, float)
    return float((sum(abs(spearman(a, rng.permutation(b))) >= obs for _ in range(n)) + 1) / (n + 1))


def kendall_w(m):
    """m: runs x tasks. W in [0, 1] (no tie correction) and a permutation p (shuffle each run)."""
    def w(x):
        r = np.array([ranks(row) for row in x])
        k, n = r.shape
        s = ((r.sum(0) - r.sum(0).mean()) ** 2).sum()
        return 12 * s / (k ** 2 * (n ** 3 - n))
    rng, obs = np.random.default_rng(1), w(m)
    null = [w(np.array([rng.permutation(row) for row in m])) for _ in range(5000)]
    return obs, (sum(v >= obs for v in null) + 1) / (len(null) + 1)


def read_runs():
    out = {}
    with open(REPO / "results/runs.csv") as f:
        for r in csv.DictReader(f):
            out[(r["run"], r["suite"], int(r["task_id"]))] = (int(r["n_success"]), int(r["n_episodes"]), r["task"])
    return out


def objects_of_interest(path: Path):
    text = path.read_text()
    block = re.search(r"\(:obj_of_interest(.*?)\)", text, re.S).group(1).split()
    # object class: drop the instance index and region suffixes (wine_rack_1_top_region -> wine_rack)
    return {re.sub(r"_\d+(_.*)?$", "", o) for o in block}


def bddl_for(suite, instruction):
    name = instruction.strip().replace(" ", "_") + ".bddl"
    hits = [p for p in (BDDL / suite).glob("*.bddl") if p.name.endswith(name)]
    assert len(hits) == 1, (suite, instruction, hits)
    return hits[0]


def demo_stats(suite):
    """Per task_index: mean frames per demo, share of frames with the gripper closed, grasps per demo."""
    root = DATA / f"{suite}_theta0"
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas()   # LeRobot v3: index = instruction
    names = tasks["task"] if "task" in tasks.columns else tasks.index
    instr = {int(i): str(t) for i, t in zip(tasks["task_index"], names)}
    frames = []
    for p in sorted((root / "data").rglob("*.parquet")):
        frames.append(pq.read_table(p, columns=["episode_index", "task_index", "action"]).to_pandas())
    import pandas as pd
    df = pd.concat(frames, ignore_index=True)
    grip = np.stack(df["action"].to_numpy())[:, 6]
    assert set(np.unique(np.round(grip))) <= {-1.0, 1.0}, "gripper not in θ0's -1/+1 convention"
    df["closed"] = grip > 0
    out = {}
    for ti, g in df.groupby("task_index"):
        per_ep = g.groupby("episode_index")
        grasps = per_ep["closed"].apply(lambda c: int(((~c.to_numpy()[:-1]) & c.to_numpy()[1:]).sum()))
        out[instr[int(ti)].strip()] = {"demo_frames": float(per_ep.size().mean()),
                                       "closed_share": float(g["closed"].mean()),
                                       "grasps_per_demo": float(grasps.mean())}
    return out


def words(s):
    return set(re.findall(r"[a-z]+", s.lower())) - {"the", "a", "it", "and", "of", "on", "in", "to", "up"}


def main():
    runs = read_runs()
    margin = {}
    with open(REPO / "results/skill_margin.csv") as f:
        rows = list(csv.DictReader(f))
    replay_cols = [c for c in rows[0] if c.startswith("retention_") and c != "retention_mean"]
    new_objs = {tid: objects_of_interest(BDDL / "libero_90" / f"{n}.bddl") for tid, n in NEW_TASKS.items()}
    new_instr = {tid: n.split("_", 2)[2].replace("_", " ") for tid, n in NEW_TASKS.items()}
    stats = {s: demo_stats(s) for s in SUITES}

    table = []
    for r in rows:
        suite, tid = r["suite"], int(r["task_id"])
        s0, n0, instr = runs[(THETA0_SETUP[suite], suite, tid)]
        rec = {"suite": suite, "task_id": tid, "task": instr, "theta0_setup": s0 / n0,
               "theta0_ref": float(r["theta0_ref"])}
        for lab, by_suite in SINGLE.items():
            s, n, _ = runs[(by_suite[suite], suite, tid)]
            rec[f"single_{lab}"] = (s / n) / (s0 / n0) if s0 else float("nan")
        for c in replay_cols:
            rec[c.replace("retention_", "replay_")] = float(r[c])
        objs = objects_of_interest(bddl_for(suite, instr))
        rec["objects"] = " ".join(sorted(objs))
        rec["shared_any_new"] = int(any(objs & o for o in new_objs.values()))
        rec["shared_task27"] = int(bool(objs & new_objs[27]))
        rec["word_overlap"] = max(len(words(instr) & words(v)) / len(words(instr) | words(v)) for v in new_instr.values())
        rec.update(stats[suite][instr.strip()])
        table.append(rec)

    single = [f"single_{k}" for k in SINGLE]
    replay = [c.replace("retention_", "replay_") for c in replay_cols]
    for rec in table:
        rec["single_mean"] = float(np.nanmean([rec[c] for c in ("single_ft23-lr1e-5", "single_ft26", "single_ft27")]))
        rec["replay_mean"] = float(np.mean([rec[c] for c in replay]))

    out = REPO / "results/task_survival.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    col = lambda c: np.array([t[c] for t in table], float)
    print("== (1) Is survival a property of the old task? Spearman ρ of per-task retention, 20 tasks")
    groups = {"same budget, other seed": [], "other budget": []}
    for a, b in itertools.combinations(replay, 2):
        same = a.split("-")[0] == b.split("-")[0]
        groups["same budget, other seed" if same else "other budget"].append(spearman(col(a), col(b)))
    groups["other new task (single ft26, ft27, lr1e-5)"] = [
        spearman(col(a), col(b)) for a, b in itertools.combinations(("single_ft23-lr1e-5", "single_ft26", "single_ft27"), 2)]
    groups["single fine-tune vs replay mean"] = [spearman(col(c), col("replay_mean"))
                                                 for c in ("single_ft23-lr1e-5", "single_ft26", "single_ft27")]
    for g, v in groups.items():
        print(f"  {g:45s} mean ρ {np.nanmean(v):+.2f}  (n pairs {len(v)}, range {np.nanmin(v):+.2f} to {np.nanmax(v):+.2f})")
    informative = replay + ["single_ft26", "single_ft27"]
    W, p = kendall_w(np.array([col(c) for c in informative]))
    print(f"  Kendall's W over {len(informative)} runs (7 replay, ft26, ft27): {W:.2f}, permutation p = {p:.4f}")
    for s in SUITES:
        m = [t["replay_mean"] for t in table if t["suite"] == s]
        ms = [t["single_mean"] for t in table if t["suite"] == s]
        print(f"  {s}: mean replay retention {np.mean(m):.2f}, mean single-fine-tune retention {np.nanmean(ms):.2f}")

    print("\n== (2) Which properties go with survival? Spearman ρ (permutation p), 20 tasks")
    props = ["theta0_setup", "demo_frames", "closed_share", "grasps_per_demo", "shared_any_new", "word_overlap"]
    for target in ("replay_mean", "single_mean"):
        print(f"  target {target}:")
        for pr in props:
            x = col(pr)
            print(f"    {pr:16s} ρ {spearman(x, col(target)):+.2f}  p {perm_p(x, col(target)):.3f}")
        x = col("shared_task27")
        print(f"    {'shared_task27':16s} vs single_ft27: ρ {spearman(x, col('single_ft27')):+.2f}  "
              f"p {perm_p(x, col('single_ft27')):.3f}  (tasks sharing an object with task 27: {int(x.sum())})")

    print("\n== (3) Demo length within each suite: Spearman ρ (permutation p), 10 tasks each")
    within = []
    for s in SUITES:
        sub = [t for t in table if t["suite"] == s]
        x = np.array([t["demo_frames"] for t in sub], float)
        for target in ("replay_mean", "single_mean"):
            y = np.array([t[target] for t in sub], float)
            rec = {"suite": s, "target": target, "n_tasks": len(sub), "rho": round(spearman(x, y), 3),
                   "perm_p": round(perm_p(x, y), 4), "n_at_zero": int((y == 0).sum())}
            within.append(rec)
            print(f"  {s:15s} {target:12s} ρ {rec['rho']:+.2f}  p {rec['perm_p']:.3f}  tasks at 0: {rec['n_at_zero']}")
    with open(REPO / "results/length_within_suite.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(within[0]))
        w.writeheader()
        w.writerows(within)

    print("\n== Tasks by mean replay retention (single: lr1e-5 / ft26 / ft27)")
    for t in sorted(table, key=lambda t: -t["replay_mean"]):
        print(f"  {t['replay_mean']:4.2f}  {t['single_ft23-lr1e-5']:4.2f} {t['single_ft26']:4.2f} {t['single_ft27']:4.2f}  "
              f"θ0 {t['theta0_setup']:.2f}  len {t['demo_frames']:5.0f}  closed {t['closed_share']:.2f}  "
              f"grasps {t['grasps_per_demo']:.1f}  shared {t['shared_any_new']}  {t['suite'][7:]} {t['task_id']}: {t['task']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
