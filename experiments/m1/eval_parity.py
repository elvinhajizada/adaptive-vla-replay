"""M1 eval diagnostic: do the eval's observations match theta0's training frames? (CPU, no rollouts)

The Spatial eval gave 64% vs a ~80% community figure. A rollout on CPU takes ~36 min, so
this checks the open-loop pieces instead, on LIBERO-Spatial:

  A. Observations. Build envs through the same LeRobot config path lerobot-eval uses,
     reset on the benchmark init states, and run preprocess_observation + the LIBERO env
     processor. Compare with frame 0 of the training demos of the same task (nvidia
     libero_spatial = the same demos as lerobot/libero, theta0's training set):
       state  per dim, in units of theta0's normalizer std (axis-angle wrap shows up here);
       images nearest-demo MAE as delivered vs rotated 180 / flipped (orientation check),
              and 360 vs 256 render after resizing to 256.
  B. Actions (theta0, fixed noise, num_steps as in the checkpoint):
       B1 teacher-forced on demo frames: first 10 predicted actions vs the demo's, the
          in-distribution error floor;
       B2 env obs vs the nearest demo's frame 0: if the env inputs were equivalent to
          training inputs, the two predictions should agree about as well as B1.

Usage: python experiments/m1/eval_parity.py [--tasks 0,1,...] [--n-init 3]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from adaptive_vla.config import Config
from adaptive_vla.libero_data import gripper_to_theta0, load_theta0_stats

IMG, WRIST = "observation.images.image", "observation.images.image2"
STATE = "observation.state"
DIM_NAMES = ["x", "y", "z", "ax", "ay", "az", "f1", "f2"]
GROUPS = {"trans": slice(0, 3), "rot": slice(3, 6), "grip": slice(6, 7)}
H_CHECK = 10                        # predicted vs demo actions compared over this many steps

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--n-init", type=int, default=3)
args = ap.parse_args()
tasks = [int(t) for t in args.tasks.split(",")]

c = Config()
hub = os.environ["HF_HOME"] + "/hub"
THETA0 = f"{hub}/models--HuggingFaceVLA--smolvla_libero/snapshots/{c.pretrained_revision}"
SPATIAL = f"{hub}/datasets--nvidia--LIBERO_LeRobot_v3/snapshots/{c.dataset_revision}/libero_spatial"
st = {k: torch.as_tensor(v) for k, v in load_theta0_stats(THETA0).items()}
s_std, a_mean, a_std = st["observation.state.std"], st["action.mean"], st["action.std"]
torch.manual_seed(0)
np.set_printoptions(precision=2, suppress=True, linewidth=140)

# ------------------------------------------------------------------ training demos
ds = LeRobotDataset("nvidia/libero_spatial", root=SPATIAL)
eps = ds.meta.episodes
demos = {}                          # instruction -> list of (from_index, length)
for i in range(len(eps)):
    e = eps[i]
    demos.setdefault(e["tasks"][0], []).append((e["dataset_from_index"], e["length"]))


def demo_frame(idx):
    s = ds[idx]
    a = s["action"].clone()
    a[GROUPS["grip"]] = torch.as_tensor(gripper_to_theta0(a[GROUPS["grip"]].numpy()))
    return {IMG: s[IMG], WRIST: s["observation.images.wrist_image"], STATE: s[STATE], "action": a}


def stack(frames):
    return {k: torch.stack([f[k] for f in frames]) for k in frames[0]}


def demo_actions(start, length, f):
    """Demo actions f .. f+H_CHECK-1 in theta0's convention (clamped at the episode end)."""
    rows = [ds.hf_dataset[start + min(f + j, length - 1)]["action"] for j in range(H_CHECK)]
    a = torch.stack([torch.as_tensor(r) for r in rows]).float()
    a[:, 6] = 1.0 - 2.0 * a[:, 6]
    return a


# ------------------------------------------------------------------ theta0, eval-path processors
cfg = PreTrainedConfig.from_pretrained(THETA0)
cfg.device = "cpu"
policy = SmolVLAPolicy.from_pretrained(THETA0, config=cfg).eval()
pre, _ = make_pre_post_processors(cfg, pretrained_path=THETA0,
                                  preprocessor_overrides={"device_processor": {"device": "cpu"}})


@torch.no_grad()
def predict(obs, task):
    """First H_CHECK actions of theta0's chunk in raw units, fixed noise per call."""
    b = {k: v.clone() for k, v in obs.items() if k in (IMG, WRIST, STATE)}
    b["task"] = [task] * b[STATE].shape[0]
    policy.reset()
    noise = torch.randn(b[STATE].shape[0], cfg.chunk_size, cfg.max_action_dim,
                        generator=torch.Generator().manual_seed(0))
    chunk = policy.predict_action_chunk(pre(b), noise=noise)[..., :7]
    return (chunk * a_std + a_mean)[:, :H_CHECK]


def err(pred, ref):
    """Mean |pred - ref| per action group, in units of theta0's action std."""
    d = ((pred - ref).abs() / a_std)
    return {g: d[..., sl].mean().item() for g, sl in GROUPS.items()}


def fmt(e):
    return "  ".join(f"{g} {v:.2f}" for g, v in e.items())


def env_obs(task_id, size, n):
    env_cfg = LiberoEnvConfig(task="libero_spatial", task_ids=[task_id],
                              observation_height=size, observation_width=size)
    env_pre, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    vec = env_cfg.create_envs(n_envs=n, use_async_envs=False)["libero_spatial"][task_id]
    raw, _ = vec.reset(seed=list(range(1000, 1000 + n)))
    o = preprocess_observation(raw)
    o["task"] = list(vec.call("task_description"))
    o = env_pre(o)
    vec.close()
    return o


def resize(x, size=256):
    return x if x.shape[-1] == size else F.interpolate(x, size=(size, size), mode="bilinear",
                                                      antialias=True, align_corners=False)


VIEWS = {"as is": lambda x: x, "rot180": lambda x: x.flip(-1, -2),
         "flipH": lambda x: x.flip(-2), "flipW": lambda x: x.flip(-1)}
summary = {"B1": [], "B2_256": [], "B2_360": [], "demo_spread": []}

for t in tasks:
    o256, o360 = env_obs(t, 256, args.n_init), env_obs(t, 360, args.n_init)
    task = o256["task"][0]
    ep_list = demos[task]
    d0 = stack([demo_frame(s) for s, _ in ep_list])     # frame 0 of every demo of this task
    print(f"\n=== Spatial task {t}: {task}  ({len(ep_list)} demos, {args.n_init} init states)")

    # A. state
    z = (o256[STATE][:, None, :] - d0[STATE][None]) / s_std          # (n_init, n_demos, 8)
    near = z.norm(dim=-1).argmin(dim=1)
    zn = torch.stack([z[i, near[i]] for i in range(len(near))])
    print("  env state (init 0)     ", o256[STATE][0].numpy())
    print("  demo frame-0 state mean", d0[STATE].mean(0).numpy())
    print("  |z| to nearest demo, per dim", dict(zip(DIM_NAMES, zn.abs().mean(0).numpy().round(2))))
    print("  |z| demo-to-demo spread     ",
          dict(zip(DIM_NAMES, ((d0[STATE] - d0[STATE].mean(0)) / s_std).abs().mean(0).numpy().round(2))))

    # A. images
    for key, name in ((IMG, "agentview"), (WRIST, "wrist")):
        row = []
        for vname, v in VIEWS.items():
            e = resize(v(o256[key]))
            mae = (e[:, None] - d0[key][None]).abs().mean(dim=(2, 3, 4)).min(dim=1).values.mean()
            row.append(f"{vname} {mae:.3f}")
        r = (resize(o360[key]) - o256[key]).abs().mean().item()
        print(f"  {name:9s} nearest-demo MAE: {'  '.join(row)}   | 360->256 vs 256: {r:.3f}")

    # B1. teacher-forced on 2 demos, frames 0, 30, 60, 90
    frames, refs = [], []
    for s, n in ep_list[:2]:
        for f in (0, 30, 60, 90):
            if f < n:
                frames.append(demo_frame(s + f))
                refs.append(demo_actions(s, n, f))
    b1 = err(predict(stack(frames), task), torch.stack(refs))
    print(f"  B1 theta0 on demo frames vs demo actions:     {fmt(b1)}")

    # B2. env obs vs nearest demo frame 0 (by state, as above)
    dn = {k: d0[k][near] for k in (IMG, WRIST, STATE)}
    p_demo = predict(dn, task)
    p256, p360 = predict(o256, task), predict(o360, task)
    print(f"  B2 env@256 vs nearest demo frame-0 prediction: {fmt(err(p256, p_demo))}")
    print(f"  B2 env@360 vs nearest demo frame-0 prediction: {fmt(err(p360, p_demo))}")
    others = {k: d0[k][(near + 1) % len(ep_list)] for k in (IMG, WRIST, STATE)}
    spread = err(predict(others, task), p_demo)
    print(f"  ref: prediction on a different demo's frame 0: {fmt(spread)}")
    print(f"  first action, env@256 init 0:  {p256[0, 0].numpy()}")
    print(f"  first action, nearest demo:    {p_demo[0, 0].numpy()}")
    for k, v in (("B1", b1), ("B2_256", err(p256, p_demo)), ("B2_360", err(p360, p_demo)), ("demo_spread", spread)):
        summary[k].append(v)

print("\n=== mean over tasks (|error| in theta0 action-std units)")
for k, rows in summary.items():
    print(f"  {k:12s}", fmt({g: float(np.mean([r[g] for r in rows])) for g in GROUPS}))
print("PARITY DONE")
