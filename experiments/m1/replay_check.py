"""M1 eval diagnostic: does env.step execute actions like the demos' controller? (CPU, no policy)

Complements experiments/m1/eval_parity.py (observations). For each LIBERO-Spatial task, build envs
through the same LeRobot config path lerobot-eval uses, reset on benchmark init states 0..n-1,
pair each with the training demo whose frame-0 agentview image is closest (most similar
layout), then replay that demo's recorded actions open-loop (gripper converted to theta0's
-1 open / +1 close, which the eval passes to the env unchanged).

The robot's start pose is the same in every demo (eval_parity: within 0.03 std), so until
contact the EEF path depends only on the controller. Reported per replay:
  pos err    |env EEF - demo EEF| in cm at steps 0, 10, 30, 60 and at the demo's end;
  gain       least-squares slope of env per-step EEF displacement on the demo's, steps 0-39
             (1.0 = the env moves exactly as much per action as the demo did);
  rot err    |axis-angle difference| at step 30, radians;
  grip       mean |finger qpos difference| over the replay, mm;
  success    whether open-loop replay solved the task (bonus: layouts differ slightly).

Usage: python experiments/m1/replay_check.py [--tasks 0,1,...] [--n-init 3]
"""
import argparse
import os

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation

from adaptive_vla.config import Config

IMG, STATE = "observation.images.image", "observation.state"
CHECK_STEPS = (0, 10, 30, 60)
GAIN_STEPS = 40

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--n-init", type=int, default=3)
args = ap.parse_args()
tasks = [int(t) for t in args.tasks.split(",")]

c = Config()
hub = os.environ["HF_HOME"] + "/hub"
SPATIAL = f"{hub}/datasets--nvidia--LIBERO_LeRobot_v3/snapshots/{c.dataset_revision}/libero_spatial"
np.set_printoptions(precision=2, suppress=True, linewidth=140)

ds = LeRobotDataset("nvidia/libero_spatial", root=SPATIAL)
eps = ds.meta.episodes
demos = {}                          # instruction -> list of (from_index, length)
for i in range(len(eps)):
    e = eps[i]
    demos.setdefault(e["tasks"][0], []).append((e["dataset_from_index"], e["length"]))


def demo_numeric(start, length):
    rows = ds.hf_dataset.select(range(start, start + length))
    state = np.stack([np.asarray(r, dtype=np.float64) for r in rows[STATE]])
    action = np.stack([np.asarray(r, dtype=np.float32) for r in rows["action"]])
    action[:, 6] = 1.0 - 2.0 * action[:, 6]           # nvidia 1 open / 0 close -> theta0 -1 / +1
    assert np.isin(action[:, 6], (-1.0, 1.0)).all()
    return state, action


def to_policy_obs(raw, task_str, env_pre):
    o = preprocess_observation(raw)
    o["task"] = task_str
    return env_pre(o)


rows_all = []
for t in tasks:
    env_cfg = LiberoEnvConfig(task="libero_spatial", task_ids=[t], observation_height=256, observation_width=256)
    env_pre, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=None)  # non-XVLA: LIBERO default
    n = args.n_init
    vec = env_cfg.create_envs(n_envs=n, use_async_envs=False)["libero_spatial"][t]
    raw, _ = vec.reset(seed=list(range(1000, 1000 + n)))
    task_str = list(vec.call("task_description"))
    o = to_policy_obs(raw, task_str, env_pre)
    ep_list = demos[task_str[0]]

    # pair each init state with the demo whose frame-0 agentview is closest
    d_img = torch.stack([ds[s][IMG] for s, _ in ep_list])                     # (N, 3, 256, 256)
    mae = (o[IMG][:, None] - d_img[None]).abs().mean(dim=(2, 3, 4))           # (n, N)
    pick = mae.argmin(dim=1).tolist()
    pairs = [demo_numeric(*ep_list[j]) for j in pick]
    T = max(len(a) for _, a in pairs)

    env_states = [[o[STATE][i].numpy().astype(np.float64)] for i in range(n)]
    live, success = [True] * n, [False] * n
    for k in range(T):
        act = np.zeros((n, 7), dtype=np.float32)
        for i, (_, a) in enumerate(pairs):
            act[i] = a[min(k, len(a) - 1)]
        raw, _, term, trunc, info = vec.step(act)
        o = to_policy_obs(raw, task_str, env_pre)
        # success as lerobot-eval reads it (gymnasium >= 1.0: final_info is a dict of arrays)
        src = info.get("final_info", info)
        step_success = np.asarray(src.get("is_success", np.zeros(n, dtype=bool))).astype(bool)
        for i in range(n):
            if not live[i]:
                continue
            success[i] |= bool(step_success[i])
            env_states[i].append(o[STATE][i].numpy().astype(np.float64))
            # NEXT_STEP autoreset: after a terminal step the next obs would be a fresh reset
            if term[i] or trunc[i] or k + 1 >= len(pairs[i][1]):
                live[i] = False
        if not any(live):
            break
    vec.close()

    print(f"\n=== Spatial task {t}: {task_str[0]}")
    for i, (ds_state, _) in enumerate(pairs):
        es = np.stack(env_states[i])
        L = min(len(es), len(ds_state))
        es, dst = es[:L], ds_state[:L]
        pos = np.linalg.norm(es[:, :3] - dst[:, :3], axis=1) * 100            # cm
        at = {s: pos[min(s, L - 1)] for s in CHECK_STEPS}
        g = min(GAIN_STEPS, L - 1)
        de, dd = np.diff(es[:g + 1, :3], axis=0).ravel(), np.diff(dst[:g + 1, :3], axis=0).ravel()
        gain = float(de @ dd / (dd @ dd))
        rot = float(np.linalg.norm(es[min(30, L - 1), 3:6] - dst[min(30, L - 1), 3:6]))
        grip = float(np.abs(es[:, 6:8] - dst[:, 6:8]).mean() * 1000)          # mm
        row = dict(task=t, init=i, demo=pick[i], img_mae=float(mae[i, pick[i]]), len=L,
                   **{f"pos@{s}": at[s] for s in CHECK_STEPS}, pos_end=pos[-1], gain=gain, rot30=rot,
                   grip_mm=grip, success=success[i])
        rows_all.append(row)
        print(f"  init {i} <- demo {pick[i]:2d} (img MAE {row['img_mae']:.3f}, {L} steps): "
              "pos err cm " + " ".join(f"@{s} {at[s]:.2f}" for s in CHECK_STEPS) + f" end {pos[-1]:.2f} | "
              f"gain {gain:.3f} | rot@30 {rot:.3f} rad | grip {grip:.2f} mm | success {success[i]}")

print("\n=== over all replays")
for key in ("pos@0", "pos@10", "pos@30", "pos@60", "pos_end", "gain", "rot30", "grip_mm"):
    v = np.array([r[key] for r in rows_all])
    print(f"  {key:8s} median {np.median(v):.3f}  min {v.min():.3f}  max {v.max():.3f}")
print(f"  open-loop success {sum(r['success'] for r in rows_all)}/{len(rows_all)}")
print("REPLAY DONE")
