"""Run the pinned lerobot-eval with goal-predicate logging (secondary metric, see subgoals.py).

Same pattern as the language diagnostic (lang_diag.py / lang_diag_env.py): lerobot_eval.make_env
is swapped for one that builds SubgoalLiberoEnv, a LiberoEnv that only *reads* the BDDL goal
predicates after each step. Observations, rewards, termination and success are untouched, so the
primary metric is computed exactly as by `lerobot-eval`; afterwards the log is summarized and
checked against eval_info.json (episode and success counts must match).

Logged episodes are those started by the rollout's explicit, seeded reset. Vector-env autoresets
(NEXT_STEP mode, after a success) call reset() without a seed and are ignored. LiberoEnv never
truncates (the rollout loop stops at the step limit), so a failed episode's record is written on
the next reset or on close().

Imports LeRobot and LIBERO at module level: use it only after `source scripts/env.sh`.
Usage (eval.sh does this when AVLA_SUBGOALS=1): python -m adaptive_vla.subgoal_env <lerobot-eval args>
       python -m adaptive_vla.subgoal_env --selftest --env.task=libero_10 --env.task_ids=[3]
"""
import functools
import json
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from lerobot.envs.libero import LiberoEnv, _get_suite, get_libero_dummy_action
from lerobot.envs.utils import _LazyAsyncVectorEnv, parse_camera_names

from . import subgoals

LOG_NAME, SUMMARY_NAME = "subgoals.jsonl", "subgoals_summary.json"


class SubgoalLiberoEnv(LiberoEnv):
    def __init__(self, *, subgoal_log: str, **kwargs):
        super().__init__(**kwargs)
        self._subgoal_log = subgoal_log
        self._ep = None
        self._ep_open = False

    def _holds(self):
        dom = self._env.env
        return [bool(dom._eval_predicate(p)) for p in dom.parsed_problem["goal_state"]]

    def _flush(self):
        if self._ep is not None:
            with open(self._subgoal_log, "a") as f:
                f.write(json.dumps(self._ep) + "\n")
        self._ep, self._ep_open = None, False

    def reset(self, seed=None, **kwargs):
        self._flush()
        init = None if self._init_states is None else int(self.init_state_id % len(self._init_states))
        obs, info = super().reset(seed=seed, **kwargs)
        if seed is not None:
            h = self._holds()
            self._ep = {"task_id": self.task_id, "init_state": init, "seed": int(np.asarray(seed).item()),
                        "goals": [" ".join(p) for p in self._env.env.parsed_problem["goal_state"]],
                        "at_reset": h, "ever": [False] * len(h), "final": h, "success": False, "steps": 0}
            self._ep_open = True
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self._ep_open:
            h = self._holds()
            ep = self._ep
            ep["final"], ep["ever"] = h, [a or b for a, b in zip(ep["ever"], h)]
            ep["steps"] += 1
            ep["success"] = bool(info["is_success"])
            if terminated or truncated:
                self._ep_open = False          # later steps belong to an autoreset episode
        return obs, reward, terminated, truncated, info

    def close(self):
        self._flush()
        super().close()


def make_subgoal_envs(env_cfg, n_envs: int, use_async_envs: bool, log_path: str):
    """{suite: {task_id: vec env}}, built like lerobot.envs.libero.create_libero_envs
    (same LiberoEnv arguments from env_cfg, same vector-env classes and autoreset mode)."""
    if "," in env_cfg.task:
        raise ValueError("subgoal logging supports one suite per eval")
    suite = _get_suite(env_cfg.task)
    gym_kwargs = dict(env_cfg.gym_kwargs)
    task_ids = gym_kwargs.pop("task_ids", None) or list(range(len(suite.tasks)))
    common = dict(task_suite=suite, task_suite_name=env_cfg.task,
                  camera_name=parse_camera_names(env_cfg.camera_name), init_states=env_cfg.init_states,
                  episode_length=env_cfg.episode_length, n_envs=n_envs, control_mode=env_cfg.control_mode,
                  camera_name_mapping=env_cfg.camera_name_mapping, is_libero_plus=env_cfg.is_libero_plus,
                  subgoal_log=log_path, **gym_kwargs)
    is_async = use_async_envs and n_envs > 1   # lerobot.envs.configs._make_vec_env_cls
    out, cached = defaultdict(dict), None
    for tid in task_ids:
        fns = [functools.partial(SubgoalLiberoEnv, task_id=tid, episode_index=i, **common) for i in range(n_envs)]
        if is_async:
            vec = _LazyAsyncVectorEnv(fns, *(cached or (None, None, None)))
            cached = cached or (vec.observation_space, vec.action_space, vec.metadata)
        else:
            vec = gym.vector.SyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
        out[env_cfg.task][tid] = vec
        print(f"Built subgoal-logging vec env | suite={env_cfg.task} | task_id={tid} | n_envs={n_envs}")
    return {k: dict(v) for k, v in out.items()}


def main() -> None:
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.scripts import lerobot_eval

    args = sys.argv[1:]
    if "--selftest" in args:      # build envs, run 2 seeded episodes of 5 no-op steps each, print the log
        kv = dict(a.lstrip("-").split("=", 1) for a in args if "=" in a)
        cfg = LiberoEnvConfig(task=kv["env.task"], task_ids=json.loads(kv["env.task_ids"]),
                              observation_height=256, observation_width=256)
        log = Path(kv.get("output_dir", "/tmp")) / LOG_NAME
        log.parent.mkdir(parents=True, exist_ok=True)
        log.unlink(missing_ok=True)
        for tid, vec in make_subgoal_envs(cfg, 2, use_async_envs=True, log_path=str(log))[cfg.task].items():
            vec.reset(seed=[1000, 1001])
            for _ in range(5):
                vec.step(np.stack([get_libero_dummy_action()] * 2))
            vec.close()
        print(subgoals.format_summary(subgoals.summarize(subgoals.read_log(str(log)))))
        print("SUBGOAL SELFTEST PASSED")
        return

    out_dir = next((a.split("=", 1)[1] for a in args if a.startswith("--output_dir=")), None)
    if out_dir is None:
        raise SystemExit("--output_dir=... is required")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log = Path(out_dir) / LOG_NAME
    log.unlink(missing_ok=True)

    def make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False, **_):
        return make_subgoal_envs(env_cfg, n_envs, use_async_envs, str(log))

    lerobot_eval.make_env = make_env   # eval_main calls it once, in this process
    sys.argv = [sys.argv[0]] + args
    lerobot_eval.main()
    summary = subgoals.write_summary(str(log), str(Path(out_dir) / "eval_info.json"),
                                     str(Path(out_dir) / SUMMARY_NAME))
    print(subgoals.format_summary(summary), flush=True)


if __name__ == "__main__":
    main()
