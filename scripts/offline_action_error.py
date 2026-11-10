"""Graded forgetting: how far do a checkpoint's predicted actions move on old-task demo frames? (CPU)

Closed-loop success collapses to 0 after a task-23 fine-tune (jobs 113800, 113802), which hides
how much the policy actually changed. Here each policy predicts action chunks teacher-forced on a
fixed set of demo frames (same frames, same flow-matching noise for every policy), and we report,
in units of θ0's action std, mean |error| over the first 10 actions of the chunk:
  vs demo   against the demo's own next actions (θ0's error is the in-distribution floor);
  vs θ0     against θ0's prediction on the same frame: pure behavioural change, no demo noise.
Frame sets (θ0 trained on the first three; task 23 is the fine-tune task, a positive control):
  spatial  LIBERO-Spatial, 10 tasks x 2 demos x frames 0/30/60/90
  goal     LIBERO-Goal,    10 tasks x 2 demos x frames 0/30/60/90
  l10_s4   LIBERO-10 "put the black bowl in the bottom drawer ... and close it", 6 demos x 6 frames
  task23   LIBERO-90 task 23 (converted data), 6 demos x 6 frames
nvidia data is converted on the fly (wrist key -> image2, gripper 1 - 2g), like libero_data.

Usage: python scripts/offline_action_error.py [--csv=OUT.csv] [--frames=NAME=JSON ...] [--only-frames]
           LABEL=POLICY_DIR [...]   (θ0 always included)
--csv also writes the table as CSV (same columns as results/offline_action_error.csv, minus job).
--frames adds a frame set of the merged replay dataset (libero_all_theta0, θ0 conventions): JSON is a
  list of [row index, episode end row, instruction] (experiments/replay/buffer_memorization.py, E44);
  --only-frames drops the four fixed sets above.
--task-frames adds spatial_t and goal_t: per task 4 demos x 7 frames at 5, 20, ..., 95% of each demo's
  length (the fixed sets above stop at frame 90, before the end of the longer tasks), and
  --per-task-csv=OUT.csv writes one row per (model, frame set, task) (E57).
POLICY_DIR may be slim (adaptive_vla.checkpoints); it is materialized to a temp dir and removed.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from adaptive_vla import checkpoints as ck
from adaptive_vla.config import Config
from adaptive_vla.lerobot_glue import load_policy, nvidia_dir, open_dataset, theta0_dir
from adaptive_vla.libero_data import load_theta0_stats

IMG, WRIST, STATE = "observation.images.image", "observation.images.image2", "observation.state"
H = 10
GROUPS = {"trans": slice(0, 3), "rot": slice(3, 6), "grip": slice(6, 7)}
c = Config()
THETA0, NV = theta0_dir(c), nvidia_dir(c)
CONV = os.environ["AVLA_DSS"] + "/data/libero90_theta0"
L10_S4 = "put the black bowl in the bottom drawer of the cabinet and close it"
T23 = "close the bottom drawer of the cabinet and open the top drawer"
st = {k: torch.as_tensor(v) for k, v in load_theta0_stats(THETA0).items()}
a_std = st["action.std"]


def frame_set(root, nvidia, tasks_filter, n_demos, frames, relative=False):
    """frames: step indices, or with relative=True fractions of each demo's length."""
    ds = open_dataset(root)
    eps, picks = ds.meta.episodes, {}
    for i in range(len(eps)):
        e = eps[i]
        t = e["tasks"][0]
        if tasks_filter and t not in tasks_filter:
            continue
        if len(picks.setdefault(t, [])) < n_demos:
            picks[t].append((e["dataset_from_index"], e["length"]))
    obs, acts, tasks = [], [], []
    for t, lst in sorted(picks.items()):
        for start, n in lst:
            for f in ([int(x * (n - 1)) for x in frames] if relative else frames):
                if f >= n:
                    continue
                s = ds[start + f]
                rows = [ds.hf_dataset[start + min(f + j, n - 1)]["action"] for j in range(H)]
                a = torch.stack([torch.as_tensor(r) for r in rows]).float()
                if nvidia:
                    a[:, 6] = 1.0 - 2.0 * a[:, 6]
                obs.append({IMG: s[IMG], WRIST: s["observation.images.wrist_image" if nvidia else WRIST],
                            STATE: s[STATE]})
                acts.append(a)
                tasks.append(t)
    return obs, torch.stack(acts), tasks


def index_set(root, rows):
    """Frames of a converted dataset by row: rows = [[index, episode end (exclusive), instruction]]."""
    ds = open_dataset(root)
    obs, acts, tasks = [], [], []
    for i, end, t in rows:
        s = ds[i]
        a = torch.stack([torch.as_tensor(ds.hf_dataset[min(i + j, end - 1)]["action"]) for j in range(H)]).float()
        obs.append({IMG: s[IMG], WRIST: s[WRIST], STATE: s[STATE]})
        acts.append(a)
        tasks.append(t)
    return obs, torch.stack(acts), tasks


ARGS = sys.argv[1:]
MERGED = os.environ["AVLA_DSS"] + "/data/libero_all_theta0"
SETS = {} if "--only-frames" in ARGS else {
    "spatial": frame_set(f"{NV}/libero_spatial", True, None, 2, (0, 30, 60, 90)),
    "goal": frame_set(f"{NV}/libero_goal", True, None, 2, (0, 30, 60, 90)),
    "l10_s4": frame_set(f"{NV}/libero_10", True, {L10_S4}, 6, (0, 40, 80, 120, 160, 200)),
    "task23": frame_set(CONV, False, {T23}, 6, (0, 40, 80, 120, 160, 200)),
}
if "--task-frames" in ARGS:
    FRACS = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
    SETS["spatial_t"] = frame_set(f"{NV}/libero_spatial", True, None, 4, FRACS, relative=True)
    SETS["goal_t"] = frame_set(f"{NV}/libero_goal", True, None, 4, FRACS, relative=True)
for a in ARGS:
    if a.startswith("--frames="):
        name, path = a.split("=", 2)[1:]
        SETS[name] = index_set(MERGED, json.load(open(path)))
for k, (o, a, _) in SETS.items():
    print(f"frame set {k}: {len(o)} frames", flush=True)


@torch.no_grad()
def predict(policy, obs, tasks, bs=8):
    out = []
    for i in range(0, len(obs), bs):
        b = {k: torch.stack([o[k] for o in obs[i:i + bs]]) for k in (IMG, WRIST, STATE)}
        b["task"] = tasks[i:i + bs]
        policy.reset()
        g = torch.Generator().manual_seed(1000 + i)
        noise = torch.randn(len(b["task"]), cfg.chunk_size, cfg.max_action_dim, generator=g)
        chunk = policy.predict_action_chunk(pre(b), noise=noise)[..., :7]
        out.append((chunk * a_std + st["action.mean"])[:, :H])
    return torch.cat(out)


def err(p, ref):
    d = (p - ref).abs() / a_std
    return {g: float(d[..., sl].mean()) for g, sl in GROUPS.items()}


CSV_OUT = next((a.split("=", 1)[1] for a in ARGS if a.startswith("--csv=")), None)
models = [("theta0", THETA0)] + [a.split("=", 1) for a in ARGS if not a.startswith("--")]
tmp_root = Path(os.environ["AVLA_RUNS"]) / "materialized" / f"offline-{os.environ.get('SLURM_JOB_ID', 'local')}"
preds = {}
for label, pdir in models:
    path = Path(pdir)
    if (path / ck.SLIM_INFO).exists():
        path = ck.materialize(str(path), str(tmp_root / label))
    cfg, policy, pre = load_policy(str(path))   # predict() reads cfg and pre
    preds[label] = {k: predict(policy, o, t) for k, (o, _, t) in SETS.items()}
    del policy
    if str(path).startswith(str(tmp_root)):
        shutil.rmtree(path)
    print(f"done {label}", flush=True)
shutil.rmtree(tmp_root, ignore_errors=True)

print("\nmean |error| over the first 10 actions, in θ0 action-std units (trans / rot / grip)")
print(f"{'model':10s} {'set':8s} {'vs demo':>24s} {'vs θ0':>24s}")
for label, _ in models:
    for k, (_, acts, _) in SETS.items():
        d, z = err(preds[label][k], acts), err(preds[label][k], preds["theta0"][k])
        f = lambda e: " / ".join(f"{e[g]:.2f}" for g in GROUPS)  # noqa: E731
        print(f"{label:10s} {k:8s} {f(d):>24s} {f(z):>24s}")
if CSV_OUT:
    import csv
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "frame_set"] + [f"vs_{r}_{g}" for r in ("demo", "theta0") for g in GROUPS])
        for label, _ in models:
            for k, (_, acts, _) in SETS.items():
                d, z = err(preds[label][k], acts), err(preds[label][k], preds["theta0"][k])
                w.writerow([label, k] + [round(d[g], 4) for g in GROUPS] + [round(z[g], 4) for g in GROUPS])
PER_TASK = next((a.split("=", 1)[1] for a in ARGS if a.startswith("--per-task-csv=")), None)
if PER_TASK:
    import csv
    with open(PER_TASK, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "frame_set", "task", "n_frames"] + [f"vs_{r}_{g}" for r in ("demo", "theta0") for g in GROUPS])
        for label, _ in models:
            for k, (_, acts, tasks) in SETS.items():
                for t in sorted(set(tasks)):
                    ix = torch.tensor([i for i, x in enumerate(tasks) if x == t])
                    d = err(preds[label][k][ix], acts[ix])
                    z = err(preds[label][k][ix], preds["theta0"][k][ix])
                    w.writerow([label, k, t, len(ix)] + [round(d[g], 4) for g in GROUPS] + [round(z[g], 4) for g in GROUPS])
print("OFFLINE ACTION ERROR DONE")
