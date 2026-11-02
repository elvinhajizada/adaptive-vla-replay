"""lerobot-train with optimizer-state controls (E35: is the early drift of a fine-tune an artefact of a fresh Adam?).

AVLA_WARM_ADAM=N   before the first real update, run N updates with the learning rate set to 0 on
                   θ0's own training data (the four LIBERO suites of libero_all_theta0), so Adam's
                   moments start warm instead of empty. The weights do not move (checked); the LR
                   schedule and the step count of the real run are untouched.
AVLA_STOP_STEP=S   stop after the checkpoint at step S. The LR schedule still spans --steps, so a
                   step-100 checkpoint matches the first 100 steps of a full run.
Plain SGD needs no patch: train.sh's AVLA_TRAIN_ARGS passes --use_policy_training_preset=false and
the optimizer and scheduler arguments. Everything else is the pinned lerobot-train, with arguments
passed through. Called by scripts/train.sh when either variable is set.
"""
import os
import sys

import numpy as np
import torch
import lerobot.scripts.lerobot_train as lt
from lerobot.datasets.lerobot_dataset import LeRobotDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_vla.libero_data import frame_groups  # noqa: E402

THETA0_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ROOT = os.path.join(os.environ["AVLA_DSS"], "data", "libero_all_theta0")
WARM = int(os.environ.get("AVLA_WARM_ADAM", "0"))
STOP = int(os.environ.get("AVLA_STOP_STEP", "0"))
CAUGHT = {}
_make_dataloaders, _preprocess, _update = lt.make_dataloaders, lt._preprocess_dataset_batch, lt.update_policy


def make_dataloaders(cfg, dataset, *args, **kwargs):
    CAUGHT.update(cfg=cfg, dataset=dataset)
    return _make_dataloaders(cfg, dataset, *args, **kwargs)


def preprocess(batch, camera_keys, rename_map, preprocessor):
    CAUGHT.setdefault("pre", (camera_keys, rename_map, preprocessor))
    return _preprocess(batch, camera_keys, rename_map, preprocessor)


def warm(train_metrics, policy, optimizer, grad_clip_norm, accelerator):
    """N zero-LR updates on θ0's training frames: fills Adam's moments, leaves the weights alone."""
    cfg = CAUGHT["cfg"]
    ds = LeRobotDataset("local/libero_all_theta0", root=ROOT, video_backend=cfg.dataset.video_backend,
                        delta_timestamps=CAUGHT["dataset"].delta_timestamps)
    rows = [i for key, ranges in frame_groups(ROOT).items() if key.split(":")[0] in THETA0_SUITES
            for r in ranges for i in r]
    idx = np.random.default_rng(cfg.seed).choice(rows, size=WARM * cfg.batch_size, replace=False).tolist()
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, sampler=idx, num_workers=cfg.num_workers,
                                     pin_memory=True, drop_last=True)
    params = [p for p in policy.parameters() if p.requires_grad]
    before = torch.stack([p.detach().double().sum() for p in params])
    lrs = [g["lr"] for g in optimizer.param_groups]
    for g in optimizer.param_groups:
        g["lr"] = 0.0
    for i, batch in enumerate(dl):
        batch = _preprocess(batch, *CAUGHT["pre"])
        train_metrics, _ = _update(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator=accelerator)
        if (i + 1) % 50 == 0:
            print(f"== warm_adam: {i + 1}/{WARM} zero-LR updates, loss {train_metrics.loss.avg:.3f}", flush=True)
    train_metrics.reset_averages()          # the first logged loss covers real updates only
    for g, lr in zip(optimizer.param_groups, lrs):
        g["lr"] = lr
    after = torch.stack([p.detach().double().sum() for p in params])
    assert torch.equal(before, after), "zero-LR warm-up moved the weights"
    v = torch.cat([optimizer.state[p]["exp_avg_sq"].flatten() for p in params if p in optimizer.state])
    print(f"== warm_adam: done on {len(rows)} θ0 frames; weights unchanged; "
          f"sqrt(exp_avg_sq) median {v.sqrt().median().item():.3e}", flush=True)


CALLS = [0]


def update_policy(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator, *args, **kwargs):
    if CALLS[0] == 0 and WARM:
        warm(train_metrics, policy, optimizer, grad_clip_norm, accelerator)
    CALLS[0] += 1
    if STOP and CALLS[0] > STOP:              # the checkpoint at step STOP is already written
        print(f"== optim_train: stopping after step {STOP}", flush=True)
        raise SystemExit(0)
    return _update(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator, *args, **kwargs)


lt.make_dataloaders, lt._preprocess_dataset_batch, lt.update_policy = make_dataloaders, preprocess, update_policy
print(f"== optim_train: warm_adam={WARM} stop_step={STOP or 'none'}", flush=True)

if __name__ == "__main__":
    lt.main()
