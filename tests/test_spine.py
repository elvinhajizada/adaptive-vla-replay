"""CPU tests for the torch-free spine. NumPy only.

Run from the repo root: `python -m pytest tests/` or `python tests/test_spine.py`.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_vla import metrics  # noqa: E402
from adaptive_vla.config import Config  # noqa: E402
from adaptive_vla.replay import MixedSampler, ReplayBuffer  # noqa: E402


def test_metrics_hand_matrix():
    # Lower triangle valid; upper = NaN. Diagonal = success right after learning a task.
    S = np.array([
        [0.80, np.nan, np.nan],
        [0.40, 0.90,  np.nan],
        [0.20, 0.50,  0.85 ],
    ])
    zs = np.array([0.05, 0.05, 0.05])
    assert math.isclose(metrics.final_avg_success(S), (0.20 + 0.50 + 0.85) / 3, rel_tol=1e-9)
    # BWT = mean((0.20-0.80), (0.50-0.90)) = -0.50
    assert math.isclose(metrics.backward_transfer(S), -0.50, rel_tol=1e-9)
    # forgetting = mean((0.8-0.2), (0.9-0.5)) = 0.50
    assert math.isclose(metrics.forgetting(S), 0.50, rel_tol=1e-9)
    # forward learning = mean(0.75, 0.85, 0.80) = 0.80
    assert math.isclose(metrics.forward_learning(S, zs), 0.80, rel_tol=1e-9)


def test_metrics_single_task_is_nan():
    S = np.array([[0.7]])
    assert math.isnan(metrics.backward_transfer(S))
    assert math.isnan(metrics.forgetting(S))
    assert math.isnan(metrics.nbt(S))
    assert math.isnan(metrics.normalized_forgetting(S))


def test_nbt_and_normalized_forgetting_hand_matrix():
    S = np.array([
        [0.80, np.nan, np.nan],
        [0.40, 0.90,  np.nan],
        [0.20, 0.50,  0.85 ],
    ])
    # NBT_0 = mean(0.8-0.4, 0.8-0.2) = 0.5; NBT_1 = 0.9-0.5 = 0.4; NBT = 0.45
    assert math.isclose(metrics.nbt(S), 0.45, rel_tol=1e-9)
    # mean(0.6/0.8, 0.4/0.9)
    assert math.isclose(metrics.normalized_forgetting(S), (0.75 + 0.4 / 0.9) / 2, rel_tol=1e-9)
    S[0, 0] = 0.0                            # a task that learned nothing is skipped
    assert math.isclose(metrics.normalized_forgetting(S), 0.4 / 0.9, rel_tol=1e-9)


def test_probe_forgetting_and_gap_recovered():
    assert math.isclose(metrics.probe_normalized_forgetting(0.76, 0.38), 0.5, rel_tol=1e-9)
    assert math.isnan(metrics.probe_normalized_forgetting(0.0, 0.0))
    assert math.isclose(metrics.gap_recovered(0.5, 0.0, 0.8), 0.625, rel_tol=1e-9)
    assert math.isnan(metrics.gap_recovered(0.5, 0.3, 0.3))   # no gap to recover


def test_success_matrix_and_bootstrap():
    ones, zeros = np.ones(20), np.zeros(20)
    out = np.array([[ones, None], [zeros, ones]], dtype=object)
    S = metrics.success_matrix(out)
    assert S[0, 0] == 1.0 and S[1, 0] == 0.0 and math.isnan(S[0, 1])
    # Deterministic cells: the interval collapses onto the point.
    assert metrics.bootstrap_ci(metrics.nbt, out, n_boot=200) == (1.0, 1.0, 1.0)
    rng = np.random.default_rng(1)
    noisy = np.array([[rng.integers(0, 2, 50), None], [rng.integers(0, 2, 50), ones]], dtype=object)
    point, lo, hi = metrics.bootstrap_ci(metrics.nbt, noisy, n_boot=500, seed=3)
    assert lo < point < hi
    assert (point, lo, hi) == metrics.bootstrap_ci(metrics.nbt, noisy, n_boot=500, seed=3)  # seeded


def test_replay_buffer_cap_and_sample():
    buf = ReplayBuffer(per_task_cap=50, seed=0)
    buf.add("t0", list(range(100)))          # over cap -> subsample to 50
    assert len(buf) == 50
    buf.add("t1", list(range(20)))           # under cap -> all 20
    assert len(buf) == 70
    assert buf.memory_report() == {"t0": 50, "t1": 20}
    assert len(buf.sample(8)) == 8
    assert len(buf.sample(1000)) == 1000     # with replacement when n > pool


def test_replay_buffer_all_budget():
    buf = ReplayBuffer(per_task_cap=None, seed=0)
    buf.add("t0", list(range(100)))
    assert len(buf) == 100 and sorted(buf.groups()["t0"]) == list(range(100))


def test_replay_buffer_frac_budget():
    buf = ReplayBuffer(per_task_cap=None, seed=0, per_task_frac=0.02)
    buf.add("long", list(range(10000)))      # 2% of each task: long tasks keep more frames
    buf.add("short", list(range(5000)))
    buf.add("tiny", list(range(10)))         # rounds to 0 -> at least 1
    assert buf.memory_report() == {"long": 200, "short": 100, "tiny": 1}
    again = ReplayBuffer(per_task_cap=None, seed=0, per_task_frac=0.02)
    again.add("long", list(range(10000)))    # same seed and order -> same frames (replay_spec --keep)
    assert again.groups()["long"] == buf.groups()["long"]


def test_mixed_sampler_mix_and_epochs():
    current = list(range(1000, 1100))                     # 100 current-task frames
    replay = {"old_a": [1, 2, 3], "old_b": list(range(10, 60))}
    s = MixedSampler(current, replay, batch_size=8, mix=0.5, seed=0)
    assert (s.n_current, s.n_replay, s.batches_per_epoch) == (4, 4, 25)
    assert len(s) == 200
    e0 = list(s)
    assert len(e0) == 200
    batches = np.asarray(e0).reshape(-1, 8)
    assert (batches[:, :4] >= 1000).all() and (batches[:, 4:] < 1000).all()   # 1:1 in every batch
    assert sorted(batches[:, :4].ravel().tolist()) == current                  # one pass, no repeats
    rep = batches[:, 4:].ravel()
    share_a = np.isin(rep, replay["old_a"]).mean()
    assert 0.3 < share_a < 0.7          # uniform over tasks, not frames (frames: 3/53 = 0.06)
    e1 = list(s)
    assert e1 != e0                                       # the epoch advances
    assert list(MixedSampler(current, replay, 8, 0.5, seed=0)) == e0          # pure in (seed, epoch)


def test_mixed_sampler_wraps_and_sequential():
    s = MixedSampler(list(range(10)), {"a": [99]}, batch_size=8, mix=0.5)
    b = np.asarray(list(s)).reshape(-1, 8)
    assert b.shape == (3, 8) and set(b[:, :4].ravel()) == set(range(10))      # last batch wraps
    seq = MixedSampler(list(range(10)), {}, batch_size=4)
    assert seq.n_replay == 0 and sorted(list(seq)[:10]) == list(range(10))


def test_replay_frames_budget():
    cfg = Config()
    assert cfg.replay_frames(7000) == 140    # k frames per task, about 2% of ~7k
    cfg.replay_k = 14
    assert cfg.replay_frames(7000) == 14
    assert cfg.replay_frames(10) == 10       # capped at the task's frames
    cfg.replay_k = None
    assert cfg.replay_frames(7000) == 7000   # "all": joint-training upper bound


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"[ok] {name}")
    print("SPINE OK")
