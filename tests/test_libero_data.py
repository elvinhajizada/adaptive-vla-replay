"""CPU tests for the LIBERO-90 conversion helpers (NumPy only, no files)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_vla import libero_data as ld  # noqa: E402


def test_gripper_mapping():
    g = np.array([1.0, 0.0, 1.0], dtype=np.float32)        # nvidia: open, close, open
    out = ld.gripper_to_theta0(g)
    assert out.tolist() == [-1.0, 1.0, -1.0]                 # theta0: -1 open, +1 close
    assert out.dtype == np.float32


def test_gripper_rejects_unexpected_values():
    with pytest.raises(ValueError):
        ld.gripper_to_theta0(np.array([0.5]))
    with pytest.raises(ValueError):                         # already-converted data
        ld.gripper_to_theta0(np.array([-1.0, 1.0]))


def test_key_and_column_mapping():
    assert ld.map_key("observation.images.wrist_image") == "observation.images.image2"
    assert ld.map_key("observation.images.image") == "observation.images.image"
    assert ld.map_key("observation.states.joint_state") is None
    assert ld.map_key("action") == "action"
    assert ld.map_column("videos/observation.images.wrist_image/chunk_index") == \
        "videos/observation.images.image2/chunk_index"
    assert ld.map_column("stats/observation.states.ee_state/mean") is None
    assert ld.map_column("stats/action/std") == "stats/action/std"
    assert ld.map_column("episode_index") == "episode_index"


def test_affine_gripper_stats_matches_data():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1000, 7))
    a[:, 6] = rng.integers(0, 2, size=1000)
    st = {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(),
          "std": a.std(0).tolist(), "count": [1000]}
    b = a.copy()
    b[:, 6] = 1 - 2 * b[:, 6]
    out = ld.affine_gripper_stats(st)
    np.testing.assert_allclose(out["min"], b.min(0))
    np.testing.assert_allclose(out["max"], b.max(0))
    np.testing.assert_allclose(out["mean"], b.mean(0))
    np.testing.assert_allclose(out["std"], b.std(0))
    assert out["count"] == [1000]


def test_convert_stats_uses_theta0_mean_std():
    src = {
        "action": {"min": [0.0] * 7, "max": [1.0] * 7, "mean": [0.5] * 7, "std": [0.5] * 7, "count": [10]},
        "observation.state": {"min": [0.0] * 8, "max": [1.0] * 8, "mean": [0.1] * 8, "std": [0.2] * 8, "count": [10]},
        "observation.images.wrist_image": {"mean": [0.3]},
        "observation.states.ee_state": {"mean": [0.0] * 6},
    }
    theta0 = {"action.mean": np.arange(7.0), "action.std": np.ones(7),
              "observation.state.mean": np.zeros(8), "observation.state.std": np.full(8, 2.0)}
    out = ld.convert_stats(src, theta0)
    assert set(out) == {"action", "observation.state", "observation.images.image2"}
    assert out["action"]["mean"] == list(np.arange(7.0))
    assert out["observation.state"]["std"] == [2.0] * 8
    assert out["action"]["min"][6] == -1.0 and out["action"]["max"][6] == 1.0   # gripper flipped


def test_convert_features_order_and_drop():
    feats = {"observation.images.image": 1, "observation.images.wrist_image": 2,
             "observation.state": 3, "observation.states.gripper_state": 4, "action": 5}
    out = ld.convert_features(feats)
    assert list(out) == ["observation.images.image", "observation.images.image2", "observation.state", "action"]


def test_task_index_for_instruction(tmp_path):
    pd = pytest.importorskip("pandas")
    (tmp_path / "meta").mkdir()
    # LeRobot v3 layout: task strings as the index, task_index as the column
    pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(["open the drawer", "close the drawer"], name="task")) \
        .to_parquet(tmp_path / "meta" / "tasks.parquet")
    assert ld.task_index_for_instruction(str(tmp_path), "close the drawer") == 1
    with pytest.raises(KeyError):
        ld.task_index_for_instruction(str(tmp_path), "close the microwave")


def test_steps_for_passes():
    assert ld.steps_for_passes(8, 8081, 64) == 1011      # task 23: 8 passes
    assert ld.steps_for_passes(8, 5309, 64) == 664       # task 26
    assert ld.steps_for_passes(1, 64, 64) == 1
    with pytest.raises(ValueError):
        ld.steps_for_passes(0, 100, 64)
