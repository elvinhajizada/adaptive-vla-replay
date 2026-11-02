"""Is lerobot-train's first-batches stall a worker-count effect? Dataloader-only benchmark (CPU).

Step-5 runs (24 workers, three runs at once) spent 4.6 min on the first 50 steps (data_s 4.9 s
per step) vs 0.75 s/step in the 8-worker smoke run. lerobot-train spawns its workers
(multiprocessing_context="spawn"), so each one re-imports torch/LeRobot and rebuilds the dataset.
This builds the same dataset and DataLoader settings as the pinned lerobot-train (task 23,
action chunk via delta_timestamps, pyav, batch 64, shuffle, prefetch 4, persistent, spawn) and
times: first batch, batches 2-50, batches 51-150. Training needs about one batch per 0.58 s.

Usage: python experiments/m1/dataloader_bench.py WORKERS [N_BATCHES]
"""
import os
import sys
import time

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401  registers "smolvla"

from adaptive_vla.config import Config
from adaptive_vla.libero_data import episodes_for_tasks

if __name__ == "__main__":
    workers, n_batches = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 150
    t0 = time.time()
    theta0 = f"{os.environ['HF_HOME']}/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/{Config().pretrained_revision}"
    root = os.environ["AVLA_DSS"] + "/data/libero90_theta0"
    cfg = PreTrainedConfig.from_pretrained(theta0)
    meta = LeRobotDatasetMetadata("local/libero90_theta0", root=root)
    ds = LeRobotDataset("local/libero90_theta0", root=root, episodes=episodes_for_tasks(root, [13]),
                        delta_timestamps=resolve_delta_timestamps(cfg, meta), video_backend="pyav")
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=workers, drop_last=False,
                                     prefetch_factor=4 if workers else None, persistent_workers=workers > 0,
                                     multiprocessing_context="spawn" if workers else None)
    t_build = time.time() - t0
    mid = min(50, n_batches)
    marks, it, start = {}, iter(dl), time.time()
    for i in range(1, n_batches + 1):
        try:
            next(it)
        except StopIteration:
            it = iter(dl); next(it)
        if i in (1, mid, n_batches):
            marks[i] = time.time() - start
    steady = (marks[n_batches] - marks[mid]) / (n_batches - mid) if n_batches > mid else float("nan")
    print(f"workers {workers:2d}: build {t_build:5.1f}s | first batch {marks[1]:6.1f}s | "
          f"batches 2-{mid} {(marks[mid] - marks[1]) / max(mid - 1, 1):5.2f} s/batch | "
          f"{mid + 1}-{n_batches} {steady:5.2f} s/batch | first {mid} total {marks[mid]:6.1f}s", flush=True)
