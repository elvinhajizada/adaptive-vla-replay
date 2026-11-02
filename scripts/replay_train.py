"""lerobot-train with experience replay (M2: a buffer of k frames per task, 1:1 mixing).

AVLA_REPLAY_SPEC=<spec.json>   written by scripts/replay_spec.py: the current task's frames, the
                               replay buffer (k frames per task), mix, batch size and seed
AVLA_SAVE_STEPS=100,664,1328   optional: save checkpoints only at these steps (and the last one),
                               for the step-100 and mid-stage checkpoints

The pinned lerobot-train builds its sampler as `EpisodeAwareSampler(...)` inside
`make_dataloaders`; this swaps in `adaptive_vla.replay.MixedSampler`, so every batch holds the
spec's mix of current-task and replay frames. Everything else is the pinned lerobot-train, with
arguments passed through. The dataset must be the merged libero_all_theta0 without
--dataset.episodes: spec indices are absolute rows of that dataset. Called by scripts/train.sh
when AVLA_REPLAY_SPEC is set.
"""
import json
import os
import sys

import lerobot.scripts.lerobot_train as lt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_vla.replay import MixedSampler  # noqa: E402

SPEC = json.load(open(os.environ["AVLA_REPLAY_SPEC"]))

batch_args = [a for a in sys.argv if a.startswith("--batch_size=")]
assert batch_args and int(batch_args[-1].split("=")[1]) == SPEC["batch_size"], \
    f"lerobot-train --batch_size must equal the spec's {SPEC['batch_size']} (batches are sampler blocks)"
assert not any(a.startswith("--dataset.episodes") for a in sys.argv), \
    "spec indices are absolute rows: do not filter episodes"


class SpecSampler(MixedSampler):
    """Accepts EpisodeAwareSampler's arguments and ignores them in favour of the spec."""

    def __init__(self, *args, absolute_to_relative_idx=None, **kwargs):
        m = absolute_to_relative_idx.__getitem__ if absolute_to_relative_idx else int
        super().__init__([m(i) for i in SPEC["current"]],
                         {k: [m(i) for i in v] for k, v in SPEC["replay"].items()},
                         batch_size=SPEC["batch_size"], mix=SPEC["mix"], seed=SPEC["seed"])
        first = next(self.batches(0))
        print(f"== replay_train: {len(self.current)} current frames, {len(self.groups)} replay tasks "
              f"({sum(len(g) for g in self.groups)} frames); batch {self.n_current} current + "
              f"{self.n_replay} replay; epoch = {self.batches_per_epoch} batches; first batch "
              f"{first[:3].tolist()} ... {first[-3:].tolist()}", flush=True)

    def state_dict(self):
        return {"epoch": self._epoch}

    def load_state_dict(self, state):
        raise NotImplementedError("M2 stages never resume")


lt.EpisodeAwareSampler = SpecSampler

if os.environ.get("AVLA_SAVE_STEPS"):
    KEEP = {int(s) for s in os.environ["AVLA_SAVE_STEPS"].split(",")}
    lt.should_save_checkpoint = lambda step, save_freq, total_steps: step in KEEP or step == total_steps
    print(f"== replay_train: checkpoints at steps {sorted(KEEP)} and the last", flush=True)

if __name__ == "__main__":
    lt.main()
