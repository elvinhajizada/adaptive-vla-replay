"""Convert nvidia LIBERO-90 to theta0's conventions and verify it through LeRobotDataset.

Run in a Slurm CPU job (file I/O + a few video decodes): scripts/convert_libero90.sbatch.
Output: $AVLA_RUNS/../data/libero90_theta0 (DSS; rebuildable from the HF cache).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_vla import libero_data as ld  # noqa: E402
from adaptive_vla.config import Config  # noqa: E402

cfg = Config()
HUB = Path(os.environ["HF_HOME"]) / "hub"
SRC = HUB / "datasets--nvidia--LIBERO_LeRobot_v3" / "snapshots" / cfg.dataset_revision / "libero_90"
THETA0 = HUB / "models--HuggingFaceVLA--smolvla_libero" / "snapshots" / cfg.pretrained_revision
DST = Path(os.environ["AVLA_RUNS"]).parent / "data" / "libero90_theta0"


def verify(dst: Path) -> None:
    from lerobot.datasets import LeRobotDataset

    theta0 = ld.load_theta0_stats(str(THETA0))
    n_eps = json.loads((dst / "meta" / "info.json").read_text())["total_episodes"]
    probe = [0, n_eps // 2, n_eps - 1]           # spread over different parquet / video files
    ds = LeRobotDataset("local/libero90_theta0", root=dst, episodes=probe)
    keys = set(ds.meta.features)
    assert ld.DST_WRIST in keys and ld.SRC_WRIST not in keys, keys
    assert not any(k.startswith(ld.DROP_PREFIX) for k in keys), keys
    for k in ("action", "observation.state"):
        for f in ("mean", "std"):
            np.testing.assert_allclose(np.asarray(ds.meta.stats[k][f]).ravel(), theta0[f"{k}.{f}"], rtol=1e-6)

    grip = np.stack(ds.hf_dataset.with_format("numpy")["action"])[:, ld.GRIPPER]
    assert np.isin(grip, (-1.0, 1.0)).all(), np.unique(grip)
    for i in (0, len(ds) // 2, len(ds) - 1):
        item = ds[i]
        for cam in ("observation.images.image", ld.DST_WRIST):
            assert tuple(item[cam].shape) == (3, 256, 256), (cam, item[cam].shape)
    print(f"[verify] ok: {len(ds)} frames from episodes {probe}; gripper in {{-1,1}} "
          f"(close fraction {np.mean(grip > 0):.2f}); stats == theta0 normalizer; both cameras decode 3x256x256")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if not args.verify_only:
        counts = ld.convert(str(SRC), str(DST), str(THETA0), overwrite=args.overwrite)
        print(f"[convert] {SRC} -> {DST}: {counts}")
    verify(DST)
