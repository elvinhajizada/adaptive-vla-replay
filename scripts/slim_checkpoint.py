"""Slim a fine-tuned policy dir in place: keep only the tensors that differ from θ0, drop training_state.

Lossless and verified before anything is deleted (adaptive_vla/checkpoints.py). Rebuild a loadable
full checkpoint with: python scripts/slim_checkpoint.py --materialize POLICY_DIR OUT_DIR

Usage: python scripts/slim_checkpoint.py POLICY_DIR [POLICY_DIR ...]
"""
import os
import sys

from adaptive_vla import checkpoints as ck
from adaptive_vla.config import Config

THETA0 = f"{os.environ['HF_HOME']}/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/{Config().pretrained_revision}"
TRAINABLE = 97_451_872          # num_learnable_params logged by lerobot-train for θ0 (expert only)

if sys.argv[1] == "--materialize":
    print("materialized ->", ck.materialize(sys.argv[2], sys.argv[3]))
    sys.exit()
for pdir in sys.argv[1:]:
    info = ck.slim_checkpoint(pdir, THETA0, drop_training_state=True, expected_params=TRAINABLE)
    print(f"slimmed {pdir}: {info['n_params'] / 1e6:.1f}M params in {info['n_tensors']} tensors, "
          f"{info['original_bytes'] / 1e9:.2f} GB -> {info['delta_bytes'] / 1e6:.0f} MB")
