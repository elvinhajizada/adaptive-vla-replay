"""Experience-replay buffer (condition 4).

Framework-agnostic: stores opaque sample handles (e.g. (task_id, demo_idx, step)
tuples, or dataset indices), not tensors. The real training loop dereferences a
sampled handle back to a (image, instruction, action) example via the dataset.
Keeping it handle-based makes the buffer cheap and torch-free, and keeps the
buffer's memory cost honest (handles, not frames).

Per-task cap with uniform-at-random eviction. This is the simple ER baseline the
2026 forget-resistance result found surprisingly strong on pretrained VLAs.

MixedSampler turns a buffer's groups plus the current task's frames into the training index
stream (1:1 mixing, uniform over tasks then frames). scripts/replay_train.py plugs it into
the pinned lerobot-train in place of its sampler.
"""
from typing import Any, Dict, Iterator, List, Optional
import math
import random

import numpy as np


class ReplayBuffer:
    def __init__(self, per_task_cap: Optional[int], seed: int = 0, per_task_frac: Optional[float] = None):
        """per_task_cap: k frames per task; None keeps every frame (the "all" budget).
        per_task_frac: instead, keep this fraction of each task's frames (E50), rounded, at least 1."""
        assert per_task_cap is None or per_task_frac is None, "give a cap or a fraction, not both"
        assert per_task_frac is None or 0 < per_task_frac <= 1, per_task_frac
        self.per_task_cap = per_task_cap
        self.per_task_frac = per_task_frac
        self._store: Dict[Any, List[Any]] = {}
        self._rng = random.Random(seed)

    def add(self, task_id: Any, samples: List[Any]) -> None:
        """Add samples for a task, keeping at most its cap via random subsample."""
        bucket = self._store.setdefault(task_id, [])
        bucket.extend(samples)
        cap = self.per_task_cap
        if self.per_task_frac is not None:
            cap = max(1, round(self.per_task_frac * len(bucket)))
        if cap is not None and len(bucket) > cap:
            self._store[task_id] = self._rng.sample(bucket, cap)

    def groups(self) -> Dict[Any, List[Any]]:
        """The stored handles per task (copies)."""
        return {t: list(b) for t, b in self._store.items()}

    def sample(self, n: int) -> List[Any]:
        """Uniformly sample n handles across all stored tasks (with replacement if needed)."""
        pool = [h for bucket in self._store.values() for h in bucket]
        if not pool:
            return []
        if n <= len(pool):
            return self._rng.sample(pool, n)
        return [self._rng.choice(pool) for _ in range(n)]

    def memory_report(self) -> Dict[Any, int]:
        return {t: len(b) for t, b in self._store.items()}

    def __len__(self) -> int:
        return sum(len(b) for b in self._store.values())


class MixedSampler:
    """Dataset-index stream for experience replay with 1:1 mixing, torch-free.

    Every consecutive block of `batch_size` indices is one training batch: `n_current` frames of
    the current task, drawn without replacement from a per-epoch permutation (the last batch of an
    epoch wraps around to the permutation's start, so every batch has the same mix), then
    `batch_size - n_current` replay frames, each drawn by picking a task uniformly and then one of
    its buffered frames uniformly (with replacement). An epoch is one pass over the current task;
    passes count current-task samples only, so `steps = ceil(passes * len(current) / n_current)`.

    With an empty `replay`, every slot is a current-task frame (plain fine-tuning, Sequential).
    Order is a pure function of (seed, epoch); each __iter__ call advances the epoch, matching the
    pinned LeRobot sampler that lerobot-train cycles over.
    """

    def __init__(self, current: List[int], replay: Dict[Any, List[int]], batch_size: int,
                 mix: float = 0.5, seed: int = 0):
        if len(current) == 0:
            raise ValueError("no current-task frames")
        if any(len(v) == 0 for v in replay.values()):
            raise ValueError("empty replay group")
        self.current = np.asarray(current, dtype=np.int64)
        self.keys = list(replay)
        self.groups = [np.asarray(replay[k], dtype=np.int64) for k in self.keys]
        self.batch_size = batch_size
        self.n_replay = int(round(batch_size * mix)) if self.groups else 0
        self.n_current = batch_size - self.n_replay
        if self.n_current <= 0:
            raise ValueError(f"mix {mix} leaves no current-task slots in a batch of {batch_size}")
        self.batches_per_epoch = math.ceil(len(self.current) / self.n_current)
        self.seed = seed
        self._epoch = 0

    def _rng(self, epoch: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.seed, epoch]))

    def batches(self, epoch: int) -> Iterator[np.ndarray]:
        """The epoch's batches as arrays of dataset indices (current frames first)."""
        rng = self._rng(epoch)
        perm = self.current[rng.permutation(len(self.current))]
        need = self.batches_per_epoch * self.n_current
        cur = np.concatenate([perm, perm[: need - len(perm)]])
        for b in range(self.batches_per_epoch):
            part = cur[b * self.n_current:(b + 1) * self.n_current]
            if self.n_replay:
                tasks = rng.integers(len(self.groups), size=self.n_replay)
                rep = np.array([g[rng.integers(len(g))] for g in (self.groups[t] for t in tasks)],
                               dtype=np.int64)
                part = np.concatenate([part, rep])
            yield part

    def __iter__(self) -> Iterator[int]:
        epoch = self._epoch
        self._epoch += 1
        for batch in self.batches(epoch):
            yield from (int(i) for i in batch)

    def __len__(self) -> int:
        return self.batches_per_epoch * self.batch_size
