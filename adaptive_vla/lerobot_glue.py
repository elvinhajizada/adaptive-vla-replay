"""Thin glue over the pinned LeRobot for SmolVLA + LIBERO, read off the installed commit
(config.LEROBOT_COMMIT), not newer docs.

Provides cache paths, `load_policy` (θ0 or a checkpoint plus θ0's eval-path preprocessor) and
`open_dataset`. Training and closed-loop eval run through the pinned CLIs (scripts/train.sh,
scripts/eval.sh); patches to the training loop are thin entry points like scripts/masked_train.py.
Replay mixing lives in scripts/replay_train.py and adaptive_vla/replay.py.
LeRobot is imported inside functions, so this module stays importable without torch or LIBERO.

Known gotchas: #2418 (SmolVLA+LIBERO tensor shapes), #2624 (LIBERO sim errors),
#3287 (replication configs), #3628 (v0.5.2 regression), and the nvidia LIBERO-90
conventions (wrist key `wrist_image`, gripper 1/0): see docs/LIBERO.md, section 5.
"""
import os
from typing import Any, List, Optional, Tuple

from .config import Config


def theta0_dir(cfg: Optional[Config] = None) -> str:
    """θ0 snapshot dir in the offline HF cache, at the pinned revision."""
    c = cfg or Config()
    return f"{os.environ['HF_HOME']}/hub/models--{c.pretrained.replace('/', '--')}/snapshots/{c.pretrained_revision}"


def nvidia_dir(cfg: Optional[Config] = None) -> str:
    """nvidia LIBERO dataset snapshot dir (one subdir per suite: libero_spatial, ..., libero_90)."""
    c = cfg or Config()
    return f"{os.environ['HF_HOME']}/hub/datasets--{c.dataset_repo.replace('/', '--')}/snapshots/{c.dataset_revision}"


def load_policy(path: Optional[str] = None, device: str = "cpu") -> Tuple[Any, Any, Any]:
    """(policy config, SmolVLA policy in eval mode, eval-path preprocessor) on `device`.

    path=None loads θ0; otherwise a full (not slim) checkpoint dir, built with θ0's config. The
    preprocessor is always θ0's: fine-tunes keep θ0's normalizer stats (checked per checkpoint by
    scripts/check_normalizer.py). Same calls as experiments/m1/eval_parity.py, which matched lerobot-eval.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    t0 = theta0_dir()
    pcfg = PreTrainedConfig.from_pretrained(t0)
    pcfg.device = device
    pre, _ = make_pre_post_processors(pcfg, pretrained_path=t0,
                                      preprocessor_overrides={"device_processor": {"device": device}})
    policy = SmolVLAPolicy.from_pretrained(str(path or t0), config=pcfg).to(device).eval()
    return pcfg, policy, pre


def open_dataset(root: str, episodes: Optional[List[int]] = None, delta_timestamps=None) -> Any:
    """A local LeRobot v3 dataset (nvidia suite dir or libero90_theta0), decoded with PyAV (dav1d)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset("local/x", root=root, episodes=episodes, delta_timestamps=delta_timestamps,
                          video_backend="pyav")
