"""Two M1 convention checks on the LIBERO-90 side (CPU, no rollouts, not a result).

E04 (experiments/m1/eval_parity.py) checked image orientation on LIBERO-Spatial only. This closes
the two remaining gaps for the data we actually fine-tune on:

  A. Orientation on LIBERO-90. For each chosen new task (Config.new_task_ids), build the
     libero_90 env through the same LeRobot config and processors as lerobot-eval (render
     256), reset on the benchmark init states, and compare with frame 0 of that task's demos
     in the converted dataset (libero90_theta0): images nearest-demo MAE as delivered vs
     rotated 180 / flipped, per camera; state in units of theta0's normalizer std.
     PASS if "as is" is the best view for every task and camera, by at least 2x.

  B. Control rate / fps label. theta0 trained on lerobot/libero (fps label 10), we train on
     nvidia (label 20). In the pinned LeRobot the action chunk is fetched by row offset
     (resolve_delta_timestamps gives i / fps, get_delta_indices gives round(d * fps) = i), so
     the label cancels; it only matters for video decoding, which looks frames up by the
     row's timestamp. Checked here:
       B1  delta indices for theta0's config are 0..49 at fps 10 and at fps 20;
       B2  converted dataset: the 50-step action chunk lerobot-train builds equals the next
           50 rows (clamped, with matching action_is_pad), on sampled frames incl. ends;
       B3  timestamp == frame_index / fps on every row, in the converted dataset and in
           lerobot/libero (data parquets fetched to $AVLA_DSS/data/lerobot_libero_parquet_*),
           and each video stream's frame rate equals the label;
       B4  lerobot/libero is not subsampled: its episode lengths equal those of the nvidia
           Spatial/Object/Goal/10 suites (so one row = one 20 Hz control step in both).
     One row = one env.step was shown in E05 (experiments/m1/replay_check.py).

Usage: python experiments/m1/libero90_parity.py [--tasks 18,23,...] [--n-init 3] [--lerobot-libero DIR]
"""
import argparse
import glob
import json
import os
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.feature_utils import get_delta_indices
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401 (registers "smolvla")

from adaptive_vla.config import Config
from adaptive_vla.libero_data import load_theta0_stats, task_index_for_instruction

IMG, WRIST = "observation.images.image", "observation.images.image2"
STATE = "observation.state"
DIM_NAMES = ["x", "y", "z", "ax", "ay", "az", "f1", "f2"]
NVIDIA_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

c = Config()
ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default=",".join(map(str, c.new_task_ids)))
ap.add_argument("--n-init", type=int, default=3)
ap.add_argument("--lerobot-libero", default=None, help="dir with lerobot/libero meta/ + data/ parquets")
args = ap.parse_args()
tasks = [int(t) for t in args.tasks.split(",")]

hub = os.environ["HF_HOME"] + "/hub"
THETA0 = f"{hub}/models--HuggingFaceVLA--smolvla_libero/snapshots/{c.pretrained_revision}"
NVIDIA = f"{hub}/datasets--nvidia--LIBERO_LeRobot_v3/snapshots/{c.dataset_revision}"
DATA = os.environ["AVLA_DSS"] + "/data/libero90_theta0"
LL = args.lerobot_libero or sorted(glob.glob(os.environ["AVLA_DSS"] + "/data/lerobot_libero_parquet_*"))[-1]
s_std = torch.as_tensor(load_theta0_stats(THETA0)["observation.state.std"])
cfg = PreTrainedConfig.from_pretrained(THETA0)
cfg.device = "cpu"
np.set_printoptions(precision=2, suppress=True, linewidth=140)
fails = []

# ------------------------------------------------------------------ A. orientation on LIBERO-90
ds = LeRobotDataset("local/libero90_theta0", root=DATA, video_backend="pyav")
eps = ds.meta.episodes
demos = {}                          # instruction -> list of (from_index, length)
for i in range(len(eps)):
    e = eps[i]
    demos.setdefault(e["tasks"][0], []).append((e["dataset_from_index"], e["length"]))


def env_obs(task_id, n):
    env_cfg = LiberoEnvConfig(task="libero_90", task_ids=[task_id],
                              observation_height=c.render_size, observation_width=c.render_size)
    env_pre, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    vec = env_cfg.create_envs(n_envs=n, use_async_envs=False)["libero_90"][task_id]
    raw, _ = vec.reset(seed=list(range(1000, 1000 + n)))
    o = preprocess_observation(raw)
    o["task"] = list(vec.call("task_description"))
    o = env_pre(o)
    vec.close()
    return o


VIEWS = {"as is": lambda x: x, "rot180": lambda x: x.flip(-1, -2),
         "flipH": lambda x: x.flip(-2), "flipW": lambda x: x.flip(-1)}
print("=== A. orientation, LIBERO-90 env (render %d) vs libero90_theta0 frame 0" % c.render_size)
for t in tasks:
    o = env_obs(t, args.n_init)
    task = o["task"][0]
    ti = task_index_for_instruction(DATA, task)          # raises unless exactly one match
    ep_list = demos[task]
    d0 = [ds[s] for s, _ in ep_list]
    d0 = {k: torch.stack([f[k] for f in d0]) for k in (IMG, WRIST, STATE)}
    print(f"\n--- env task {t} (dataset task_index {ti}): {task}  ({len(ep_list)} demos, {args.n_init} init states)")

    z = (o[STATE][:, None, :] - d0[STATE][None]) / s_std
    near = z.norm(dim=-1).argmin(dim=1)
    zn = torch.stack([z[i, near[i]] for i in range(len(near))])
    print("  |z| to nearest demo, per dim", dict(zip(DIM_NAMES, zn.abs().mean(0).numpy().round(2))))
    print("  |z| demo-to-demo spread     ",
          dict(zip(DIM_NAMES, ((d0[STATE] - d0[STATE].mean(0)) / s_std).abs().mean(0).numpy().round(2))))

    for key, name in ((IMG, "agentview"), (WRIST, "wrist")):
        assert o[key].shape[-2:] == d0[key].shape[-2:], (o[key].shape, d0[key].shape)
        mae = {}
        for vname, v in VIEWS.items():
            mae[vname] = (v(o[key])[:, None] - d0[key][None]).abs().mean(dim=(2, 3, 4)).min(dim=1).values.mean().item()
        flipped = min(v for k, v in mae.items() if k != "as is")
        ok = mae["as is"] * 2 <= flipped
        print(f"  {name:9s} nearest-demo MAE: " + "  ".join(f"{k} {v:.3f}" for k, v in mae.items())
              + f"   {'ok' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"A task {t} {name}")

# ------------------------------------------------------------------ B. fps label and chunk timing
print("\n=== B. fps label, action chunk and timestamps")
idx10 = get_delta_indices({"action": [i / 10 for i in cfg.action_delta_indices]}, 10)["action"]
idx20 = get_delta_indices({"action": [i / 20 for i in cfg.action_delta_indices]}, 20)["action"]
b1 = idx10 == idx20 == list(range(cfg.chunk_size))
print(f"  B1 delta indices at fps 10 == fps 20 == 0..{cfg.chunk_size - 1}: {b1}")
if not b1:
    fails.append("B1")

# B2: the chunk lerobot-train builds for theta0's config, on the converted dataset.
dt = resolve_delta_timestamps(cfg, ds.meta)
print(f"  B2 resolve_delta_timestamps keys {sorted(dt)}; action deltas {dt['action'][:3]} ... {dt['action'][-1]:.2f} s")
dsc = LeRobotDataset("local/libero90_theta0", root=DATA, video_backend="pyav", delta_timestamps=dt)
rng = np.random.default_rng(0)
acts = torch.as_tensor(np.stack(ds.hf_dataset.with_format("numpy")["action"])).float()
n_chk, b2_bad = 0, 0
for s, n in [p for k in demos for p in demos[k][:2]][:40]:        # 2 episodes of 20 tasks
    for f in sorted({0, int(rng.integers(0, n)), n - 30, n - 1}):
        if f < 0:
            continue
        smp = dsc[s + f]
        rows = [s + min(f + j, n - 1) for j in range(cfg.chunk_size)]
        pad = torch.tensor([f + j >= n for j in range(cfg.chunk_size)])
        same = torch.equal(smp["action"], acts[rows]) and torch.equal(smp["action_is_pad"], pad)
        n_chk += 1
        b2_bad += not same
print(f"  B2 chunk == next {cfg.chunk_size} rows (+ pad flags) on {n_chk} frames of 40 episodes: "
      f"{n_chk - b2_bad}/{n_chk}")
if b2_bad:
    fails.append("B2")


def ts_check(root, fps, name):
    worst, rows = 0.0, 0
    for p in sorted(Path(root, "data").rglob("*.parquet")):
        t = pq.read_table(p, columns=["timestamp", "frame_index"])
        d = np.abs(t["timestamp"].to_numpy() - t["frame_index"].to_numpy() / fps)
        worst, rows = max(worst, float(d.max())), rows + len(d)
    ok = worst < 1e-4
    print(f"  B3 {name}: fps label {fps}, max |timestamp - frame_index/fps| = {worst:.2e} over {rows} rows"
          f"   {'ok' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"B3 {name}")
    return rows


ts_check(DATA, ds.meta.fps, "libero90_theta0")
ll_info = json.loads(Path(LL, "meta", "info.json").read_text())
print(f"  lerobot/libero @ {Path(LL, 'REVISION').read_text().strip()}: codebase {ll_info['codebase_version']}, "
      f"fps {ll_info['fps']}, video fps "
      + str({k: v.get('info', {}).get('video.fps') for k, v in ll_info['features'].items() if v['dtype'] == 'video'}))
ll_rows = ts_check(LL, ll_info["fps"], "lerobot/libero")

for key in ds.meta.video_keys:
    vids = sorted(Path(DATA, "videos", key).rglob("*.mp4"))
    with av.open(str(vids[0])) as cont:
        rate = float(cont.streams.video[0].average_rate)
    ok = abs(rate - ds.meta.fps) < 1e-3
    print(f"  B3 libero90_theta0 {key}: video stream rate {rate:g} vs label {ds.meta.fps}   {'ok' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"B3 video {key}")

# B4: lerobot/libero episode lengths vs nvidia's four suites.
e = np.concatenate([pq.read_table(p, columns=["episode_index"])["episode_index"].to_numpy()
                    for p in sorted(Path(LL, "data").rglob("*.parquet"))])
ll_len = np.sort(np.bincount(e)[np.unique(e)])
nv_len = np.sort(np.concatenate([np.asarray(LeRobotDatasetMetadata(f"nvidia/{s}", root=f"{NVIDIA}/{s}").episodes["length"])
                                 for s in NVIDIA_SUITES]))
b4 = len(ll_len) == len(nv_len) and np.array_equal(ll_len, nv_len)
print(f"  B4 episode lengths, lerobot/libero ({len(ll_len)} eps, {ll_rows} rows) vs nvidia 4 suites "
      f"({len(nv_len)} eps, {nv_len.sum()} rows): {'identical' if b4 else 'DIFFER'}"
      f"; median {np.median(ll_len):.0f} vs {np.median(nv_len):.0f}")
if not b4:
    fails.append("B4")

print("\n" + ("LIBERO90 PARITY PASS" if not fails else f"LIBERO90 PARITY FAIL: {fails}"))
