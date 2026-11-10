"""Where do two closed-loop evals of the same checkpoint first diverge? (repeatability test)

Reads the step logs (steps.jsonl) and subgoal logs (subgoals.jsonl) of two eval dirs written with
AVLA_SUBGOALS=1 AVLA_STEPLOG=1, pairs episodes by (task_id, init_state, seed), and finds the first
step at which they differ. obs[t] is what the policy sees before choosing actions[t]; actions[t]
produces obs[t+1]. So per episode:
- identical:  every observation hash and every action is equal;
- policy:     the first difference is an action chosen from an identical observation history;
- simulator:  the first difference is an observation after identical actions (physics or render);
- reset:      the observations differ already at reset.
Torch-free. Usage: python experiments/setup/repeatability_compare.py DIR_A DIR_B [--out FILE]
"""
import argparse
import json
from pathlib import Path


def load(d: Path):
    def read(name):
        with open(d / name) as f:
            return [json.loads(line) for line in f if line.strip()]

    key = lambda r: (r["task_id"], r["init_state"], r["seed"])
    steps = {key(r): r for r in read("steps.jsonl")}
    success = {key(r): r["success"] for r in read("subgoals.jsonl")}
    return steps, success


def first_diff(a, b):
    n = min(len(a), len(b))
    return next((t for t in range(n) if a[t] != b[t]), None if len(a) == len(b) else n)


def compare_episode(a, b):
    t_obs, t_act = first_diff(a["obs"], b["obs"]), first_diff(a["actions"], b["actions"])
    if t_obs is None and t_act is None:
        return {"kind": "identical", "step": None, "max_abs_diff": 0.0}
    def what(t):   # at an observation divergence: does the simulator state differ, which parts do
        if "sim" not in a or t >= min(len(a["sim"]), len(b["sim"])):
            return {}
        pa, pb = a["parts"][t], b["parts"][t]
        return {"sim_state_differs": a["sim"][t] != b["sim"][t],
                "parts_differ": sorted(k for k in pa if pa[k] != pb.get(k)),
                "first_sim_diff": first_diff(a["sim"], b["sim"])}
    if t_obs == 0:
        return {"kind": "reset", "step": 0, "max_abs_diff": None, **what(0)}
    if t_act is not None and (t_obs is None or t_act < t_obs):   # same history up to obs[t_act]
        da = max(abs(x - y) for x, y in zip(a["actions"][t_act], b["actions"][t_act])) \
            if t_act < min(len(a["actions"]), len(b["actions"])) else None
        return {"kind": "policy", "step": t_act, "max_abs_diff": da}
    return {"kind": "simulator", "step": t_obs, "max_abs_diff": None, **what(t_obs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a", type=Path)
    ap.add_argument("dir_b", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    (sa, ua), (sb, ub) = load(args.dir_a), load(args.dir_b)
    keys = sorted(set(sa) & set(sb))
    if set(sa) != set(sb):
        print(f"warning: {len(set(sa) ^ set(sb))} episodes in only one run")
    eps = []
    for k in keys:
        r = compare_episode(sa[k], sb[k])
        r.update(task_id=k[0], init_state=k[1], seed=k[2], success_a=ua.get(k), success_b=ub.get(k),
                 steps_a=len(sa[k]["actions"]), steps_b=len(sb[k]["actions"]))
        eps.append(r)
    kinds = {k: sum(e["kind"] == k for e in eps) for k in ("identical", "policy", "simulator", "reset")}
    div = sorted(e["step"] for e in eps if e["step"] is not None)
    diffs = sorted(e["max_abs_diff"] for e in eps if e["kind"] == "policy" and e["max_abs_diff"] is not None)
    summary = {
        "dir_a": str(args.dir_a), "dir_b": str(args.dir_b), "episodes": len(eps), "kinds": kinds,
        "first_divergence_step": {"min": div[0], "median": div[len(div) // 2], "max": div[-1]} if div else None,
        "first_action_abs_diff": {"min": diffs[0], "median": diffs[len(diffs) // 2], "max": diffs[-1]} if diffs else None,
        "success_flips": sum(e["success_a"] != e["success_b"] for e in eps),
        "obs_divergence_with_sim_state_differing": sum(bool(e.get("sim_state_differs")) for e in eps),
        "obs_divergence_with_sim_state_equal": sum(e.get("sim_state_differs") is False for e in eps),
        "parts_differing": {p: sum(p in e.get("parts_differ", []) for e in eps)
                            for p in sorted({p for e in eps for p in e.get("parts_differ", [])})},
        "success": [sum(bool(e["success_a"]) for e in eps), sum(bool(e["success_b"]) for e in eps)],
        "episodes_detail": eps,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes_detail"}, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
