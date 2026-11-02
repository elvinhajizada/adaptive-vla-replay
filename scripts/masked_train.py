"""lerobot-train with part of the action loss masked out (E26 counterfactual fine-tunes).

AVLA_LOSS_DIMS=arm   supervise only the arm (action dims 0-5); the gripper loss is zeroed
AVLA_LOSS_DIMS=grip  supervise only the gripper (dim 6)
Everything else is the pinned lerobot-train, arguments passed through. Called by scripts/train.sh
when AVLA_LOSS_DIMS is set. Zeroed dims still count in the loss mean; AdamW is invariant to that
constant scale, so the per-step update size matches an unmasked run's.
"""
import os

import torch
from lerobot.policies.smolvla import modeling_smolvla as ms
from lerobot.scripts.lerobot_train import main

KEEP = {"arm": slice(0, 6), "grip": slice(6, 7)}[os.environ["AVLA_LOSS_DIMS"]]
_forward = ms.VLAFlowMatching.forward


def forward(self, *args, **kwargs):
    losses = _forward(self, *args, **kwargs)          # (batch, chunk, max_action_dim)
    mask = torch.zeros(losses.shape[-1], device=losses.device, dtype=losses.dtype)
    mask[KEEP] = 1.0
    return losses * mask


ms.VLAFlowMatching.forward = forward
print(f"== masked_train: loss on action dims {KEEP} only", flush=True)

if __name__ == "__main__":
    main()
