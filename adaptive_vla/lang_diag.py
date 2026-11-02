"""Language check: does θ0 use the language instruction? (docs/LIBERO.md, sections 9 and 11).

Protocol. LIBERO-Goal, a few tasks, N episodes each, three conditions:
  true     task A's initial state, A's instruction, scored by A's goal (baseline)
  blank    task A's initial state, a blank instruction, scored by A's goal
  swapped  task A's initial state, B's instruction, scored by B's goal
Every condition runs in task A's env from A's init states with the same seeds, so episode i
starts from the same state in all three conditions (paired). Everything else is the
scripts/eval.sh protocol: init_states=true, hard resets, seed 1000, batch = episodes per
task, offline, MUJOCO_GL=egl, render size from AVLA_RENDER.

How "swapped" is built (read off the pinned LeRobot 9a6bb61 and hf-libero 0.1.4):
- The policy reads the instruction from the env attribute `task_description`
  (`lerobot_eval.rollout`: observation["task"] = env.call("task_description")), and the
  score is `info["is_success"]` from `OffScreenRenderEnv.check_success()`, which evaluates
  `parsed_problem["goal_state"]` of the env's own BDDL (all predicates must hold).
- All 10 LIBERO-Goal BDDL files are identical except `:language`, `:obj_of_interest` and
  `:goal`: same fixtures, objects, regions and `:init` placement regions, so they compile to
  the same MuJoCo model (init-state files are all (50, 79): separate samples from the same
  regions). `obj_of_interest` only feeds LIBERO's segmentation helpers, not obs or success.
- So we keep A's env (A's BDDL, A's init states: the physical setup is exactly the "true"
  one) and replace two things: `task_description` (B's language) and the goal predicates
  that `check_success()` evaluates (B's `goal_state`). The alternative, B's env loaded with
  A's init state, gives the same model here but relies on the state layouts matching; ours
  never loads a state into a foreign model.
- Fail-loud checks (adaptive_vla/lang_diag_env.py): every Goal BDDL must parse to the same
  scene as A's (else RuntimeError), every goal predicate must name objects/regions that exist
  in A's sim, the init-state length must equal the sim state length, the scored goal must not
  already hold at reset, and at every step the env's own success flag must equal our
  evaluation of the scored goal. `--selftest` runs the env-side checks without the policy.

What is logged per episode. The env's reward channel carries a bitmask over the suite's
goals: bit k is set once goal k (LIBERO-Goal task k's goal) holds at some step of the
episode, counting only goals that did not already hold at reset. LIBERO's reward is just
the success indicator, so nothing is lost. `max_reward` in eval_info.json is that mask;
`sum_reward` is meaningless here. This gives, for free: A-goal success under "swapped"
(did it do A instead of B?), and A- and B-goal rates under "blank" and "true".

Blank. SmolVLA's preprocessor appends "\\n" to the task and tokenizes without special
tokens, so "" becomes the single token "\\n" (id 198), the terminator every training
instruction ends with. `--blank` sets another text (e.g. a neutral placeholder).

This file is torch-free (tested in tests/test_lang_diag.py). The LeRobot/LIBERO side is
adaptive_vla/lang_diag_env.py. Entry point: scripts/lang_diag.sh.
"""
import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

CONDITIONS = ("true", "blank", "swapped")
# Parsed-BDDL keys that define the physical scene; they must match across the suite.
SCENE_KEYS = ("problem_name", "fixtures", "regions", "objects", "scene_properties", "initial_state")


@dataclass(frozen=True)
class EnvSpec:
    """One (condition, task A) cell: A's env and init states, this instruction, this goal."""
    condition: str
    task_id: int                 # A: env, BDDL scene and init states
    instruction: str             # what the policy reads
    instructed_task_id: Optional[int]   # whose instruction it is (None for blank)
    scored_task_id: int          # whose goal decides success
    partner_task_id: int         # B from the pair, tracked in every condition


# ---------- tasks and pairs ----------

def parse_tasks(s: str) -> List[int]:
    """'8,9,7' or '[8, 9, 7]' -> [8, 9, 7]."""
    ids = [int(t) for t in re.split(r"[,\s]+", s.strip().strip("[]")) if t]
    if not ids:
        raise ValueError("no tasks given")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate task ids: {ids}")
    return ids


def cyclic_pairs(tasks: Sequence[int]) -> Dict[int, int]:
    """A -> B pairing that shifts the task list by one: every task is A once and B once."""
    if len(tasks) < 2:
        raise ValueError("need at least 2 tasks to pair")
    return {a: tasks[(i + 1) % len(tasks)] for i, a in enumerate(tasks)}


def parse_pairs(s: str) -> Dict[int, int]:
    """'8:9,9:7' -> {8: 9, 9: 7}."""
    pairs: Dict[int, int] = {}
    for item in s.replace(" ", "").split(","):
        if not item:
            continue
        a, b = (int(x) for x in item.split(":"))
        if a in pairs:
            raise ValueError(f"task {a} paired twice")
        pairs[a] = b
    return pairs


def validate_pairs(tasks: Sequence[int], pairs: Dict[int, int], n_suite_tasks: int) -> None:
    for t in tasks:
        if not 0 <= t < n_suite_tasks:
            raise ValueError(f"task {t} out of range [0, {n_suite_tasks - 1}]")
    if set(pairs) != set(tasks):
        raise ValueError(f"pairs must give one B for every task A: tasks {sorted(tasks)}, "
                         f"pairs for {sorted(pairs)}")
    for a, b in pairs.items():
        if a == b:
            raise ValueError(f"task {a} paired with itself")
        if not 0 <= b < n_suite_tasks:
            raise ValueError(f"pair {a}:{b} out of range [0, {n_suite_tasks - 1}]")


def build_specs(tasks: Sequence[int], pairs: Dict[int, int], languages: Sequence[str],
                conditions: Sequence[str] = CONDITIONS, blank: str = "") -> List[EnvSpec]:
    """All (condition, A) cells, grouped by condition in CONDITIONS order, tasks in given order."""
    validate_pairs(tasks, pairs, len(languages))
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions {sorted(unknown)}; choose from {CONDITIONS}")
    specs = []
    for cond in (c for c in CONDITIONS if c in conditions):
        for a in tasks:
            b = pairs[a]
            if cond == "true":
                specs.append(EnvSpec(cond, a, languages[a], a, a, b))
            elif cond == "blank":
                specs.append(EnvSpec(cond, a, blank, None, a, b))
            else:
                specs.append(EnvSpec(cond, a, languages[b], b, b, b))
    return specs


# ---------- BDDL text checks (pure; the env re-checks with LIBERO's own parser) ----------

def _strip_block(text: str, head: str) -> str:
    """Remove the balanced s-expression that starts with '(' + head."""
    i = text.find("(" + head)
    if i < 0:
        return text
    depth = 0
    for j in range(i, len(text)):
        depth += {"(": 1, ")": -1}.get(text[j], 0)
        if depth == 0:
            return text[:i] + text[j + 1:]
    raise ValueError(f"unbalanced {head} block")


def bddl_scene_text(text: str) -> str:
    """BDDL text without language, obj_of_interest and goal, whitespace-normalized."""
    for head in (":language", ":obj_of_interest", ":goal"):
        text = _strip_block(text, head)
    return " ".join(text.split())


def bddl_goal_text(text: str) -> str:
    """The :goal block, whitespace-normalized."""
    i = text.find("(:goal")
    if i < 0:
        raise ValueError("no :goal block")
    rest = text[i:]
    block_len = len(rest) - len(_strip_block(rest, ":goal"))
    return " ".join(rest[:block_len].split())


def check_shared_scene(bddl_texts: Dict[str, str]) -> None:
    """Raise if the BDDL files do not describe one scene with distinct goals."""
    names = list(bddl_texts)
    ref = bddl_scene_text(bddl_texts[names[0]])
    for n in names[1:]:
        if bddl_scene_text(bddl_texts[n]) != ref:
            raise ValueError(f"{n} has a different scene than {names[0]}")
    goals = [bddl_goal_text(bddl_texts[n]) for n in names]
    if len(set(goals)) != len(goals):
        raise ValueError("two tasks share a goal")


# ---------- aggregation ----------

def mask_bits(mask: float) -> List[int]:
    """Goal bitmask (as stored in max_reward) -> sorted goal ids."""
    m = int(round(mask))
    if m < 0 or abs(m - mask) > 1e-6:
        raise ValueError(f"not a goal bitmask: {mask}")
    return [k for k in range(m.bit_length()) if m >> k & 1]


def wilson_ci95(k: int, n: int) -> List[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    z, p = 1.959963984540054, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * max(0.0, c - h), 1), round(100 * min(1.0, c + h), 1)]


def _rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "pc": round(100 * k / n, 1) if n else float("nan"), "ci95": wilson_ci95(k, n)}


def summarize(eval_info: dict, specs: Sequence[EnvSpec]) -> dict:
    """Per-episode goal outcomes and per-condition rates from lerobot-eval's eval_info.json.

    Raises if any episode's success flag disagrees with its scored goal's bit: that would mean
    the env scored a different goal than the spec says.
    """
    by_cell = {(s.condition, s.task_id): s for s in specs}
    episodes = []
    for t in eval_info["per_task"]:
        spec = by_cell[(t["task_group"], t["task_id"])]
        m = t["metrics"]
        for i, (success, mask) in enumerate(zip(m["successes"], m["max_rewards"], strict=True)):
            goals = mask_bits(mask)
            if bool(success) != (spec.scored_task_id in goals):
                raise ValueError(f"{spec.condition} task {spec.task_id} episode {i}: success={success} "
                                 f"but goals reached {goals} (scored goal {spec.scored_task_id})")
            episodes.append({
                "condition": spec.condition, "task_id": spec.task_id, "episode": i,
                "partner_task_id": spec.partner_task_id, "success": bool(success),
                "own_goal": spec.task_id in goals, "partner_goal": spec.partner_task_id in goals,
                "goals_reached": goals,
            })
    conditions = {}
    for cond in CONDITIONS:
        eps = [e for e in episodes if e["condition"] == cond]
        if not eps:
            continue
        n = len(eps)
        row = {
            "success": _rate(sum(e["success"] for e in eps), n),
            "own_goal": _rate(sum(e["own_goal"] for e in eps), n),
            "partner_goal": _rate(sum(e["partner_goal"] for e in eps), n),
            "own_not_partner": _rate(sum(e["own_goal"] and not e["partner_goal"] for e in eps), n),
            "no_goal": _rate(sum(not e["goals_reached"] for e in eps), n),
            "per_task": {},
        }
        for tid in dict.fromkeys(e["task_id"] for e in eps):
            te = [e for e in eps if e["task_id"] == tid]
            row["per_task"][tid] = {
                "success": sum(e["success"] for e in te),
                "own_goal": sum(e["own_goal"] for e in te),
                "partner_goal": sum(e["partner_goal"] for e in te),
                "n": len(te),
            }
        conditions[cond] = row
    return {"conditions": conditions, "episodes": episodes, "specs": [asdict(s) for s in specs]}


def format_summary(summary: dict) -> str:
    """Plain-text table: one row per condition, then per task."""
    lines = ["condition  n    success          A-goal           B-goal           A-not-B   no-goal",
             "  (success = scored goal: A for true/blank, B for swapped; A = init-state task, B = pair)"]
    for cond, r in summary["conditions"].items():
        def f(key):
            x = r[key]
            return f"{x['k']:3d}/{x['n']:<3d} {x['pc']:5.1f}%"
        lines.append(f"{cond:9s} {r['success']['n']:3d}  {f('success')}  {f('own_goal')}  "
                     f"{f('partner_goal')}  {r['own_not_partner']['k']:3d}       {r['no_goal']['k']:3d}")
        for tid, pt in r["per_task"].items():
            spec = next(s for s in summary["specs"] if s["condition"] == cond and s["task_id"] == int(tid))
            lines.append(f"    A={int(tid)} B={spec['partner_task_id']}: success {pt['success']}/{pt['n']}, "
                         f"A-goal {pt['own_goal']}, B-goal {pt['partner_goal']}, "
                         f"instruction {spec['instruction']!r}")
    return "\n".join(lines)


# ---------- CLI: lerobot-eval with the env factory swapped ----------

class _SelftestDone(Exception):
    pass


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="theta0 language diagnostic. Unknown flags go to lerobot-eval unchanged.",
        allow_abbrev=False)
    ap.add_argument("--tasks", required=True, help="LIBERO-Goal task ids (A), e.g. 8,9,7")
    ap.add_argument("--pairs", default="cycle", help="A:B list, e.g. 8:9,9:7,7:8 (default: shift tasks by one)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--blank", default="", help="instruction text for the blank condition")
    ap.add_argument("--selftest", action="store_true",
                    help="build and reset every env, check goals and instructions, step once, exit (no policy)")
    args, rest = ap.parse_known_args(argv if argv is not None else sys.argv[1:])
    out_dir = next((a.split("=", 1)[1] for a in rest if a.startswith("--output_dir=")), None)
    if out_dir is None and not args.selftest:
        raise SystemExit("--output_dir=... is required")

    tasks = parse_tasks(args.tasks)
    pairs = cyclic_pairs(tasks) if args.pairs == "cycle" else parse_pairs(args.pairs)
    conditions = [c for c in args.conditions.split(",") if c]

    # Heavy imports only here: tests import this module without LeRobot.
    from lerobot.scripts import lerobot_eval
    from . import lang_diag_env

    state = {}

    def make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False, **_):
        languages = lang_diag_env.suite_languages(env_cfg.task)
        specs = build_specs(tasks, pairs, languages, conditions, args.blank)
        state["specs"] = specs
        for s in specs:
            print(f"[lang_diag] {s.condition:8s} A={s.task_id} scored={s.scored_task_id} "
                  f"partner={s.partner_task_id} instruction={s.instruction!r}", flush=True)
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            json.dump({"suite": env_cfg.task, "tasks": tasks, "pairs": pairs, "blank": args.blank,
                       "languages": list(languages), "specs": [asdict(s) for s in specs]},
                      open(Path(out_dir) / "lang_diag_spec.json", "w"), indent=2)
        envs = lang_diag_env.make_diag_envs(env_cfg, specs, n_envs, use_async_envs)
        if args.selftest:
            lang_diag_env.selftest(envs, specs)
            raise _SelftestDone
        return envs

    lerobot_eval.make_env = make_env   # eval_main calls it once, in this process
    sys.argv = [sys.argv[0]] + rest
    try:
        lerobot_eval.main()
    except _SelftestDone:
        print("[lang_diag] SELFTEST PASSED", flush=True)
        return

    info = json.load(open(Path(out_dir) / "eval_info.json"))
    summary = summarize(info, state["specs"])
    json.dump(summary, open(Path(out_dir) / "lang_diag_summary.json", "w"), indent=2)
    print(format_summary(summary), flush=True)


if __name__ == "__main__":
    main()
