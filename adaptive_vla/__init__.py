"""adaptive_vla: continual skill learning on a deployed SmolVLA (LIBERO).

Two layers:
  - torch-free spine (config, metrics, replay, provenance): pure Python / NumPy, unit-tested
    on CPU in tests/.
  - model wiring (lerobot_glue): thin wrappers over the pinned LeRobot commit, read off the
    installed version (policy and dataset loading; training and eval run through the pinned CLIs).
"""

__all__ = ["config", "metrics", "replay", "provenance", "lerobot_glue"]
