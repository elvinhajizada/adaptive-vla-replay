"""Subgoal-level eval diagnostics (secondary metric, added 2026-10-22).

LIBERO counts an episode as a success only if every goal predicate of the task's BDDL holds at
once (`_check_success` is a conjunction), and that binary success stays the primary metric.
Two-goal tasks (task 23, all LIBERO-10 probes) can still be partly done or partly forgotten,
which binary success hides. subgoal_env.SubgoalLiberoEnv writes one JSON line per eval episode:

  {"task_id", "init_state", "goals": ["Close white_cabinet_1_bottom_region", ...],
   "at_reset": [bool], "ever": [bool], "final": [bool], "success": bool, "steps": int}

`ever` = the predicate held at some step after reset, `final` = it held at the last step.
This module only aggregates those lines (pure Python, unit-tested).
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def read_log(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(records: List[dict]) -> Dict[int, dict]:
    """Per task: episodes, success, and per goal predicate the final / ever / at-reset rates.

    `partial` counts failed episodes that ended with at least one goal predicate holding that
    did not already hold at reset.
    """
    by_task = defaultdict(list)
    for r in records:
        by_task[int(r["task_id"])].append(r)
    out = {}
    for tid, rs in sorted(by_task.items()):
        inits = [r["init_state"] for r in rs]
        if len(set(inits)) != len(inits):
            raise ValueError(f"task {tid}: duplicate init states in subgoal log {sorted(inits)}")
        goals = rs[0]["goals"]
        if any(r["goals"] != goals for r in rs):
            raise ValueError(f"task {tid}: goal predicates differ between episodes")
        n = len(rs)
        rate = lambda key, k: sum(r[key][k] for r in rs) / n  # noqa: E731
        out[tid] = {
            "n": n,
            "success": sum(r["success"] for r in rs),
            "partial": sum((not r["success"]) and any(f and not a for f, a in zip(r["final"], r["at_reset"]))
                           for r in rs),
            "goals": [{"goal": g, "final": rate("final", k), "ever": rate("ever", k),
                       "at_reset": rate("at_reset", k)} for k, g in enumerate(goals)],
        }
    return out


def check_against_eval_info(summary: Dict[int, dict], eval_info: dict) -> None:
    """Fail loudly if the log disagrees with lerobot-eval's own per-task episode and success counts."""
    for t in eval_info["per_task"]:
        tid, s = int(t["task_id"]), summary.get(int(t["task_id"]))
        if s is None:
            raise ValueError(f"task {tid}: no subgoal records")
        if (s["n"], s["success"]) != (t["n_episodes"], t["n_success"]):
            raise ValueError(f"task {tid}: subgoal log has {s['success']}/{s['n']} successes, "
                             f"lerobot-eval {t['n_success']}/{t['n_episodes']}")


def format_summary(summary: Dict[int, dict]) -> str:
    lines = []
    for tid, s in summary.items():
        lines.append(f"task {tid}: success {s['success']}/{s['n']}, partial (failed, some goal reached) "
                     f"{s['partial']}/{s['n']}")
        for g in s["goals"]:
            lines.append(f"    {g['goal']:55s} final {g['final']:5.0%}  ever {g['ever']:5.0%}  "
                         f"at reset {g['at_reset']:5.0%}")
    return "\n".join(lines)


def write_summary(log_path: str, eval_info_path: str, out_path: str) -> Dict[int, dict]:
    summary = summarize(read_log(log_path))
    check_against_eval_info(summary, json.load(open(eval_info_path)))
    Path(out_path).write_text(json.dumps({str(k): v for k, v in summary.items()}, indent=2))
    return summary
