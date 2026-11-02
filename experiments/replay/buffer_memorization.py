"""E44 frame lists: buffered vs unbuffered θ0 frames of one replay run, plus its current task (CPU).

  python experiments/replay/buffer_memorization.py SPEC OUT_DIR [--per-task 5] [--gap 10] [--seed 0]

SPEC is a stage's replay spec (`$AVLA_RUNS/replay_specs/<job>-er-k<K>-s5.json`). For each of θ0's
40 tasks in its buffer, draws `per_task` buffered frames and as many unbuffered frames of the same
task, at least `gap` rows away from every buffered frame of the same episode (a neighbouring frame
is almost the same observation). Both are uniform draws, so their phase mix matches in
expectation. `current` takes 5 x 40 frames of the stage's own task. Writes buffered.json,
unbuffered.json, current.json in the format of `offline_action_error.py --frames`.
"""
import argparse
import bisect
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from adaptive_vla.libero_data import frame_groups  # noqa: E402

THETA0_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def rows(key, picks, ranges):
    """[index, episode end, instruction] for each picked row of a task."""
    starts = [r.start for r in ranges]
    out = []
    for i in picks:
        r = ranges[bisect.bisect_right(starts, i) - 1]
        assert i in r, (key, i)
        out.append([i, r.stop, key.split(":", 1)[1]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out_dir")
    ap.add_argument("--per-task", type=int, default=5)
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=os.path.join(os.environ["AVLA_DSS"], "data", "libero_all_theta0"))
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    assert spec["k"] != "all", "full replay has no unbuffered frames"
    groups = frame_groups(a.root)
    rng = random.Random(a.seed)
    sets = {"buffered": [], "unbuffered": [], "current": []}
    for key in sorted(spec["replay"]):
        if key.split(":")[0] not in THETA0_SUITES:
            continue                                   # θ0 never saw the earlier new tasks
        buf = spec["replay"][key]
        near = {i + d for i in buf for d in range(-a.gap, a.gap + 1)}
        pool = [i for r in groups[key] for i in r if i not in near]
        sets["buffered"] += rows(key, rng.sample(buf, a.per_task), groups[key])
        sets["unbuffered"] += rows(key, rng.sample(pool, a.per_task), groups[key])
    cur = spec["current_key"]
    sets["current"] = rows(cur, rng.sample(spec["current"], a.per_task * 40), groups[cur])
    os.makedirs(a.out_dir, exist_ok=True)
    for name, v in sets.items():
        json.dump(v, open(os.path.join(a.out_dir, f"{name}.json"), "w"))
        print(f"{name}: {len(v)} frames")


if __name__ == "__main__":
    main()
