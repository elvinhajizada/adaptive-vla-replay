"""Offline coverage of replay buffers (a first check for a coverage selector; CPU, no video).

Question: is there a within-task selection lever at k = 14, 40, 140 frames per task? Compares, per
replayed task, uniform random buffers with greedy k-center (farthest-point) buffers, and measures
the actual replay buffers (E37 k = 14 seeds b0/b1/b2, E39 k = 140) and whether their coverage predicts
retention across seeds.

Features (a proxy for selector features, no images): θ0-normalized robot state (8) and the
θ0-normalized 50-step action chunk from each frame, PCA to 24 dims; both blocks scaled to unit mean
squared norm and weighted equally (w = 0.5). A visual block (SigLIP
embedding) is left out: decoding 280k frames is a separate job.
Phases: segments between gripper toggles (θ0 convention, action[6] < 0 open, > 0 closed); a task's
phases are the segment ordinals present in at least half its demos. Transition frames: within 5
steps (0.25 s) of a toggle.

  python experiments/coverage/coverage_offline.py OUT_DIR SPEC [SPEC ...]
  SPEC = label=path/to/stage-5 replay spec json (every buffered task is in it)
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from adaptive_vla.libero_data import frame_groups  # noqa: E402

ROOT = os.path.join(os.environ["AVLA_DSS"], "data", "libero_all_theta0")
KS = (14, 40, 140)
N_RANDOM = 50
H, W_TRANS, PCA_DIM = 50, 5, 24
FINAL_RUNS = {"k14-b0": ("115623", "115624", "115638"), "k14-b1": ("115630", "115639"),
              "k14-b2": ("115636", "115640"), "k140-b0": ("115761", "115762")}


def load(keys):
    groups = frame_groups(ROOT)
    t = pq.read_table(os.path.join(ROOT, "data", "chunk-000", "file-000.parquet"),
                      columns=["index", "episode_index", "observation.state", "action"])
    idx = t.column("index").to_numpy()
    assert (idx == np.arange(len(idx))).all(), "rows are not in global index order"
    ep = t.column("episode_index").to_numpy()
    state = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
    act = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
    stats = json.load(open(os.path.join(ROOT, "meta", "stats.json")))
    norm = lambda x, k: (x - np.array(stats[k]["mean"], np.float32)) / np.array(stats[k]["std"], np.float32)
    rows = {k: np.concatenate([np.arange(r.start, r.stop) for r in groups[k]]) for k in keys}
    return groups, ep, norm(state, "observation.state"), norm(act, "action"), act[:, 6], rows


def chunks(act_n, ep_end, rows):
    """50-step action chunk from each row, padded with the episode's last action (as the dataset)."""
    idx = np.minimum(rows[:, None] + np.arange(H)[None, :], ep_end[rows][:, None])
    return act_n[idx].reshape(len(rows), -1)


def phases(grip, ep, rows):
    """Per row: segment ordinal (gripper toggles so far in the episode) and transition flag."""
    seg = np.zeros(len(rows), int)
    trans = np.zeros(len(rows), bool)
    by_ep = defaultdict(list)
    for n, r in enumerate(rows):
        by_ep[ep[r]].append(n)
    for e, ns in by_ep.items():
        ns = np.array(ns)
        g = grip[rows[ns]] > 0
        tog = np.flatnonzero(g[1:] != g[:-1]) + 1
        seg[ns] = np.searchsorted(tog, np.arange(len(ns)), side="right")
        for t in tog:
            trans[ns[max(0, t - W_TRANS):t + W_TRANS]] = True
    return seg, trans


def nn_dist(X, sel):
    d = np.full(len(X), np.inf, np.float32)
    for s in sel:
        d = np.minimum(d, ((X - X[s]) ** 2).sum(1))
    return np.sqrt(d)


def kcenter(X, k):
    first = int(np.argmin(((X - X.mean(0)) ** 2).sum(1)))
    sel, d = [first], ((X - X[first]) ** 2).sum(1)
    while len(sel) < k:
        s = int(np.argmax(d))
        sel.append(s)
        d = np.minimum(d, ((X - X[s]) ** 2).sum(1))
    return np.array(sel)


def metrics(X, seg, trans, ep_local, task_phases, sel):
    d = nn_dist(X, sel)
    return {"radius": float(d.max()), "mean_nn": float(d.mean()),
            "phase_cov": float(np.mean([np.any(seg[sel] == p) for p in task_phases])),
            "trans_share": float(trans[sel].mean()), "demos_cov": float(len(set(ep_local[sel])) / len(set(ep_local)))}


def main() -> None:
    out_dir = sys.argv[1]
    specs = dict(a.split("=", 1) for a in sys.argv[2:])
    os.makedirs(out_dir, exist_ok=True)
    bufs = {lab: json.load(open(p))["replay"] for lab, p in specs.items()}
    keys = sorted(set().union(*[set(b) for b in bufs.values()]))
    groups, ep, state_n, act_n, grip, rows = load(keys)
    last = np.r_[np.flatnonzero(ep[1:] != ep[:-1]), len(ep) - 1]   # last row of each episode run
    ep_end = np.repeat(last, np.diff(np.r_[-1, last]))
    print(f"== {len(keys)} replayed tasks, {sum(len(r) for r in rows.values())} frames")

    # PCA of action chunks and block scales, fitted once on θ0's frames (S2c option A)
    rng = np.random.default_rng(0)
    theta0_rows = np.concatenate([rows[k] for k in keys if not k.startswith("libero_90:")])
    fit = rng.choice(theta0_rows, 40000, replace=False)
    C = chunks(act_n, ep_end, fit)
    mu = C.mean(0)
    _, sv, Vt = np.linalg.svd(C - mu, full_matrices=False)
    P = Vt[:PCA_DIM].T
    print(f"== action-chunk PCA: {PCA_DIM} dims keep {100 * (sv[:PCA_DIM] ** 2).sum() / (sv ** 2).sum():.1f}% variance")
    s_scale = np.sqrt((state_n[fit] ** 2).sum(1).mean())
    a_scale = np.sqrt((((C - mu) @ P) ** 2).sum(1).mean())

    per_task, actual = [], []
    for key in keys:
        r = rows[key]
        X = np.hstack([np.sqrt(0.5) * state_n[r] / s_scale,
                       np.sqrt(0.5) * ((chunks(act_n, ep_end, r) - mu) @ P) / a_scale]).astype(np.float32)
        seg, trans = phases(grip, ep, r)
        ep_local = ep[r]
        n_demos = len(set(ep_local))
        counts = defaultdict(set)
        for s, e in zip(seg, ep_local):
            counts[s].add(e)
        task_phases = [p for p, es in counts.items() if len(es) >= n_demos / 2]
        base = {"trans_base": float(trans.mean()), "n_frames": len(r), "n_demos": n_demos, "n_phases": len(task_phases)}
        for k in KS:
            rand = [metrics(X, seg, trans, ep_local, task_phases, rng.choice(len(r), k, replace=False))
                    for _ in range(N_RANDOM)]
            kc = metrics(X, seg, trans, ep_local, task_phases, kcenter(X, k))
            for m in kc:
                vals = np.array([x[m] for x in rand])
                per_task.append({"task": key, "k": k, "metric": m, "random_mean": vals.mean(),
                                 "random_p05": np.quantile(vals, 0.05), "random_p95": np.quantile(vals, 0.95),
                                 "kcenter": kc[m], **base})
        pos = {g: n for n, g in enumerate(r)}
        for lab, b in bufs.items():
            if key in b:
                sel = np.array([pos[g] for g in b[key]])
                actual.append({"buffer": lab, "task": key, "k": len(sel),
                               **metrics(X, seg, trans, ep_local, task_phases, sel), **base})

    with open(os.path.join(out_dir, "coverage_per_task.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_task[0]))
        w.writeheader()
        w.writerows(per_task)
    with open(os.path.join(out_dir, "coverage_actual_buffers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(actual[0]))
        w.writeheader()
        w.writerows(actual)

    print("\n== random vs greedy k-center, mean over tasks (random: mean of 50 draws)")
    print(f"{'k':>4s} {'metric':12s} {'random':>8s} {'k-center':>9s} {'ratio':>7s}  base rate / note")
    for k in KS:
        for m in ("radius", "mean_nn", "phase_cov", "demos_cov", "trans_share"):
            rows_m = [x for x in per_task if x["k"] == k and x["metric"] == m]
            rm, kc = np.mean([x["random_mean"] for x in rows_m]), np.mean([x["kcenter"] for x in rows_m])
            note = f"transition base rate {np.mean([x['trans_base'] for x in rows_m]):.3f}" if m == "trans_share" else ""
            if m == "phase_cov":
                miss = np.mean([x["random_p05"] < 1 for x in rows_m])
                note = f"{100 * miss:.0f}% of tasks: 5th-percentile random draw misses a phase"
            print(f"{k:4d} {m:12s} {rm:8.3f} {kc:9.3f} {rm / kc if kc else float('nan'):7.2f}  {note}")
    print(f"\n== phases per task: median {np.median([x['n_phases'] for x in per_task]):.0f}, "
          f"range {min(x['n_phases'] for x in per_task)} to {max(x['n_phases'] for x in per_task)}")

    # does an actual buffer's coverage predict retention across seeds? (within-task, k = 14 seeds)
    succ = defaultdict(dict)
    with open("results/runs.csv") as f:
        for row in csv.DictReader(f):
            if "final" not in row["run"]:
                continue
            for lab, jobs in FINAL_RUNS.items():
                if row["policy"].split("/")[1:2] == [jobs[0]]:
                    key = f"{row['suite']}:{row['task']}"
                    s, n = succ[lab].get(key, (0, 0))
                    succ[lab][key] = (s + int(row["n_success"]), n + int(row["n_episodes"]))
    cov = {(a["buffer"], a["task"]): a for a in actual}
    pts = []
    for key in keys:
        labs = [l for l in ("k14-b0", "k14-b1", "k14-b2") if key in succ[l] and (l, key) in cov]
        if len(labs) == 3:
            y = np.array([succ[l][key][0] / succ[l][key][1] for l in labs])
            for m in ("mean_nn", "radius", "phase_cov"):
                x = np.array([cov[(l, key)][m] for l in labs])
                pts.append((key, m, x - x.mean(), y - y.mean()))
    print(f"\n== k = 14 buffers vs final success, within task across 3 seeds ({len(pts) // 3} tasks)")
    for m in ("mean_nn", "radius", "phase_cov"):
        xs = np.concatenate([p[2] for p in pts if p[1] == m])
        ys = np.concatenate([p[3] for p in pts if p[1] == m])
        r = np.corrcoef(xs, ys)[0, 1] if xs.std() > 0 and ys.std() > 0 else float("nan")
        perm = []
        groups_m = [(p[2], p[3]) for p in pts if p[1] == m]
        for _ in range(5000):
            px = np.concatenate([rng.permutation(x) for x, _ in groups_m])
            perm.append(np.corrcoef(px, ys)[0, 1] if px.std() > 0 else 0)
        p = np.mean(np.abs(perm) >= abs(r)) if not np.isnan(r) else float("nan")
        print(f"  {m:10s} r = {r:+.2f} (permutation p = {p:.3f}; negative r for mean_nn/radius = better coverage, more success)")
    print("  confound: a seed changes the buffer and the training run together")

    print("\n== actual buffers, mean over buffered tasks")
    for lab in bufs:
        a = [x for x in actual if x["buffer"] == lab]
        print(f"  {lab:8s} k {a[0]['k']:4d}: radius {np.mean([x['radius'] for x in a]):.3f}, mean_nn "
              f"{np.mean([x['mean_nn'] for x in a]):.3f}, phase_cov {np.mean([x['phase_cov'] for x in a]):.3f}, "
              f"demos_cov {np.mean([x['demos_cov'] for x in a]):.3f}")
    print("COVERAGE OFFLINE DONE")


if __name__ == "__main__":
    main()
