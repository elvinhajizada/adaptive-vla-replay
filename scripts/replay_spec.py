"""Write the replay spec for one replay stage (read by scripts/replay_train.py).

  python scripts/replay_spec.py --current 26 --k 140 --seed 0 --out spec.json
  python scripts/replay_spec.py --current 23 --earlier 26,18 --old all --k 140 --seed 1 --out spec.json

--current   libero_90 env id of the task trained in this stage (mapped by instruction, never id)
--earlier   libero_90 env ids of earlier new tasks, replayed like θ0's tasks
--old       θ0 training suites in the buffer: all | none | a comma list, e.g. "libero_spatial,
            libero_object,libero_10" (the Goal hold-out run, E36)
--k         frames per task in the buffer; "all" keeps every frame; "p<percent>", e.g. "p2.05",
            keeps that share of each task's frames instead (E50: 2.05% of θ0's data, as k = 140, but
            split in proportion to task length)
--seed      buffer sample and sampler seed (varies with the run seed)
--keep      the previous stage's spec: assert every task buffered there has the same frames here
Random selection: each task's k frames are a uniform sample without replacement (the random-replay
condition). Steps = ceil(passes x current frames / current slots per batch): passes count
current-task samples only. Prints the steps and the buffer ledger; the spec carries both.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_vla.config import Config  # noqa: E402
from adaptive_vla.libero_data import frame_groups, steps_for_passes  # noqa: E402
from adaptive_vla.replay import MixedSampler, ReplayBuffer  # noqa: E402

THETA0_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
# Raw bytes of one replayed frame: two 256x256 RGB images, state (8 float32), 50-step action chunk
# (7 float32). Observations only: no action chunk.
BYTES_OBS = 2 * 256 * 256 * 3 + 8 * 4
BYTES_DEMO = BYTES_OBS + 50 * 7 * 4
# Rows of libero_all_theta0 that cannot be decoded: one corrupt AV1 packet in the upstream
# nvidia/LIBERO_LeRobot_v3 libero_goal wrist video (episode 4889, "push the plate to the front of the
# stove", frames 105 and 106 of 129; experiments/replay/video_integrity.py, E46). Loading either row
# raises, so no completed run ever trained on them. They are removed after the buffer draw, so every
# k-frame buffer drawn before this fix is unchanged; only k = "all" loses these 2 frames.
UNDECODABLE = {699256, 699257}


def instruction(env_id: int) -> str:
    from libero.libero.benchmark import get_benchmark_dict
    return get_benchmark_dict()["libero_90"]().get_task(env_id).language


def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", type=int, required=True)
    ap.add_argument("--earlier", default="")
    ap.add_argument("--old", default="all")
    ap.add_argument("--k", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--passes", type=float, default=cfg.passes_per_task)
    ap.add_argument("--batch", type=int, default=cfg.batch_size)
    ap.add_argument("--mix", type=float, default=cfg.replay_mix)
    ap.add_argument("--root", default=os.path.join(os.environ["AVLA_DSS"], "data", "libero_all_theta0"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", default="")
    a = ap.parse_args()

    groups = frame_groups(a.root)
    old = [] if a.old == "none" else THETA0_SUITES if a.old == "all" else a.old.split(",")
    assert set(old) <= set(THETA0_SUITES), old
    cur_key = f"libero_90:{instruction(a.current)}"
    earlier = [int(e) for e in a.earlier.split(",") if e]
    # θ0's tasks first (sorted, as in the single-stage pilot), then earlier new tasks in sequence
    # order. The buffer draws every task's k frames from one rng stream, so this order gives each
    # task the same sample at every later stage: a task's frames are stored once, and stage 1
    # of a sequence is the pilot's buffer.
    keys = sorted(k for k in groups if k.split(":")[0] in old) + [f"libero_90:{instruction(e)}" for e in earlier]
    for key in [cur_key] + keys:
        assert key in groups, f"no frames for {key!r}"
    assert cur_key not in keys, "the current task cannot also be replayed"

    frac = float(a.k[1:]) / 100 if a.k.startswith("p") else None
    k = None if a.k == "all" or frac is not None else int(a.k)
    buf = ReplayBuffer(per_task_cap=k, seed=a.seed, per_task_frac=frac)
    for key in keys:                                   # fixed order: the sample depends on the seed only
        buf.add(key, [i for r in groups[key] for i in r])
    replay = {key: sorted(i for i in v if i not in UNDECODABLE) for key, v in buf.groups().items()}
    current = [i for r in groups[cur_key] for i in r if i not in UNDECODABLE]
    if a.keep:
        prev = json.load(open(a.keep))
        assert set(prev["replay"]) <= set(replay), "a task buffered in the previous stage is missing"
        for key, v in prev["replay"].items():
            assert replay[key] == [i for i in v if i not in UNDECODABLE], f"buffer for {key!r} differs from {a.keep}"
        print(f"== --keep: all {len(prev['replay'])} buffered tasks of {a.keep} keep their frames")

    sampler = MixedSampler(current, replay, a.batch, a.mix if replay else 0.0, a.seed)
    steps = steps_for_passes(a.passes, len(current), sampler.n_current)
    n_frames = sum(len(v) for v in replay.values())
    ledger = {"replay_tasks": len(replay), "replay_frames": n_frames,
              "replay_mb_demo": round(n_frames * BYTES_DEMO / 1e6, 1),
              "replay_mb_obs": round(n_frames * BYTES_OBS / 1e6, 1),
              "current_frames": len(current)}
    spec = {"current_env_id": a.current, "current_key": cur_key, "earlier_env_ids": earlier,
            "old_suites": old, "k": a.k, "seed": a.seed, "passes": a.passes, "batch_size": a.batch,
            "mix": a.mix if replay else 0.0, "steps": steps, "ledger": ledger,
            "current": current, "replay": replay}
    with open(a.out, "w") as f:
        json.dump(spec, f)
    print(json.dumps({k: v for k, v in spec.items() if k not in ("current", "replay")}))
    print(steps)


if __name__ == "__main__":
    main()
