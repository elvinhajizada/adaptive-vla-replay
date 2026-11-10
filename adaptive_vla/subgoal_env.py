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

Two opt-in diagnostics for eval repeatability (experiments/setup/repeatability.sbatch):
- AVLA_STEPLOG=1 also writes steps.jsonl: per logged episode, a hash of every observation (the one
  from reset, then one per step), of each observation part (camera images, state) and of the
  simulator state (qpos, qvel), and every action exactly as the env receives it. Comparing two
  runs (experiments/setup/repeatability_compare.py) finds where their rollouts first differ.
- AVLA_DETERMINISTIC=1 turns off cuDNN autotuning (the pinned eval_main sets benchmark=True) and turns
  on torch's deterministic algorithms; CUBLAS_WORKSPACE_CONFIG must be set before CUDA starts.
- AVLA_ACTION_NOISE=σ adds N(0, (σ x θ0's action std)^2) to the six arm dimensions of every action
  before the simulator executes it (gripper untouched), drawn from a generator seeded by the episode
  seed (noise-tolerance test, E58). With AVLA_STEPLOG the logged actions are the executed ones.

Imports LeRobot and LIBERO at module level: use it only after `source scripts/env.sh`.
Usage (eval.sh does this when AVLA_SUBGOALS=1): python -m adaptive_vla.subgoal_env <lerobot-eval args>
       python -m adaptive_vla.subgoal_env --selftest --env.task=libero_10 --env.task_ids=[3]
"""
import functools
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from lerobot.envs.libero import LiberoEnv, _get_suite, get_libero_dummy_action
from lerobot.envs.utils import _LazyAsyncVectorEnv, parse_camera_names

from . import subgoals

LOG_NAME, SUMMARY_NAME, STEP_LOG_NAME = "subgoals.jsonl", "subgoals_summary.json", "steps.jsonl"


def obs_hash(obs) -> str:
    """Hash of every array in an observation dict (keys sorted), exact to the byte."""
    h = hashlib.blake2b(digest_size=8)

    def feed(x):
        if isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode())
                feed(x[k])
        else:
            a = np.ascontiguousarray(x)
            h.update(str(a.dtype).encode() + str(a.shape).encode())
            h.update(a.tobytes())

    feed(obs)
    return h.hexdigest()


class SubgoalLiberoEnv(LiberoEnv):
    def __init__(self, *, subgoal_log: str, step_log: str | None = None,
                 action_noise: float = 0.0, action_std=None, **kwargs):
        super().__init__(**kwargs)
        self._subgoal_log = subgoal_log
        self._step_log = step_log
        self._noise = float(action_noise)
        self._noise_std = None if action_std is None else np.asarray(action_std, dtype=np.float64)[:6]
        self._noise_rng = None
        self._steps = None
        self._ep = None
        self._ep_open = False

    def _step_record(self, obs):
        """Hashes of the observation, of each of its parts, and of the simulator state (qpos, qvel)."""
        d = self._env.env.sim.data
        parts = {}

        def walk(x, prefix=""):
            for k in sorted(x):
                if isinstance(x[k], dict):
                    walk(x[k], f"{prefix}{k}.")
                else:
                    parts[prefix + k] = obs_hash(x[k])

        walk(obs)
        return obs_hash(obs), parts, obs_hash({"qpos": d.qpos, "qvel": d.qvel})

    def _log_step(self, obs):
        h, parts, sim = self._step_record(obs)
        self._steps["obs"].append(h)
        self._steps["parts"].append(parts)
        self._steps["sim"].append(sim)

    def _holds(self):
        dom = self._env.env
        return [bool(dom._eval_predicate(p)) for p in dom.parsed_problem["goal_state"]]

    def _flush(self):
        if self._ep is not None:
            with open(self._subgoal_log, "a") as f:
                f.write(json.dumps(self._ep) + "\n")
            if self._steps is not None:
                with open(self._step_log, "a") as f:
                    f.write(json.dumps(self._steps) + "\n")
        self._ep, self._steps, self._ep_open = None, None, False

    def reset(self, seed=None, **kwargs):
        self._flush()
        init = None if self._init_states is None else int(self.init_state_id % len(self._init_states))
        obs, info = super().reset(seed=seed, **kwargs)
        if seed is not None:
            if self._noise > 0:   # noise draws seeded by the episode, so noise levels share draws up to scale
                self._noise_rng = np.random.default_rng(int(np.asarray(seed).item()) + 7919)
            h = self._holds()
            self._ep = {"task_id": self.task_id, "init_state": init, "seed": int(np.asarray(seed).item()),
                        "goals": [" ".join(p) for p in self._env.env.parsed_problem["goal_state"]],
                        "at_reset": h, "ever": [False] * len(h), "final": h, "success": False, "steps": 0}
            if self._step_log:
                self._steps = {k: self._ep[k] for k in ("task_id", "init_state", "seed")}
                self._steps.update(obs=[], parts=[], sim=[], actions=[])
                self._log_step(obs)
            self._ep_open = True
        return obs, info

    def step(self, action):
        if self._noise > 0 and self._noise_rng is not None:
            action = np.array(action, dtype=np.float64)
            action[:6] += self._noise_rng.normal(0.0, 1.0, 6) * self._noise * self._noise_std
        obs, reward, terminated, truncated, info = super().step(action)
        if self._ep_open:
            if self._steps is not None:   # float32 -> float64 is exact, and json round-trips float64
                self._steps["actions"].append(np.asarray(action, dtype=np.float64).tolist())
                self._log_step(obs)
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


def make_subgoal_envs(env_cfg, n_envs: int, use_async_envs: bool, log_path: str,
                      step_log_path: str | None = None, action_noise: float = 0.0, action_std=None):
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
                  subgoal_log=log_path, step_log=step_log_path, action_noise=action_noise,
                  action_std=None if action_std is None else list(map(float, action_std)), **gym_kwargs)
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
        step_log = log.parent / STEP_LOG_NAME
        step_log.unlink(missing_ok=True)
        envs = make_subgoal_envs(cfg, 2, use_async_envs=True, log_path=str(log), step_log_path=str(step_log))
        for tid, vec in envs[cfg.task].items():
            vec.reset(seed=[1000, 1001])
            for _ in range(5):
                vec.step(np.stack([get_libero_dummy_action()] * 2))
            vec.close()
        print(subgoals.format_summary(subgoals.summarize(subgoals.read_log(str(log)))))
        steps = subgoals.read_log(str(step_log))
        assert steps and all(len(s["obs"]) == len(s["actions"]) + 1 == 6 for s in steps), steps
        print(f"step log: {len(steps)} episodes, first obs hash {steps[0]['obs'][0]}")
        print("SUBGOAL SELFTEST PASSED")
        return

    out_dir = next((a.split("=", 1)[1] for a in args if a.startswith("--output_dir=")), None)
    if out_dir is None:
        raise SystemExit("--output_dir=... is required")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log = Path(out_dir) / LOG_NAME
    log.unlink(missing_ok=True)
    step_log = None
    if os.environ.get("AVLA_STEPLOG") == "1":
        step_log = Path(out_dir) / STEP_LOG_NAME
        step_log.unlink(missing_ok=True)

    noise, a_std = float(os.environ.get("AVLA_ACTION_NOISE") or 0), None
    if noise > 0:
        from .lerobot_glue import theta0_dir
        from .libero_data import load_theta0_stats
        a_std = load_theta0_stats(theta0_dir())["action.std"]
        print(f"action noise: N(0, ({noise} x θ0 action std)^2) on the 6 arm dims, std {a_std[:6].round(4).tolist()}",
              flush=True)

    def make_env(env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False, **_):
        return make_subgoal_envs(env_cfg, n_envs, use_async_envs, str(log), step_log and str(step_log),
                                 action_noise=noise, action_std=a_std)

    lerobot_eval.make_env = make_env   # eval_main calls it once, in this process
    if os.environ.get("AVLA_DETERMINISTIC") == "1":
        import torch
        seed_fn = lerobot_eval.set_seed

        def deterministic_set_seed(seed, *a, **kw):   # eval_main calls it right after benchmark=True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)
            print(f"deterministic: cudnn.benchmark=False, cudnn.deterministic=True, "
                  f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}", flush=True)
            return seed_fn(seed, *a, **kw)

        lerobot_eval.set_seed = deterministic_set_seed
    flags = {k: os.environ.get(k) for k in ("AVLA_STEPLOG", "AVLA_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG",
                                            "AVLA_ACTION_NOISE")}
    if any(flags[k] not in (None, "", "0") for k in ("AVLA_STEPLOG", "AVLA_DETERMINISTIC", "AVLA_ACTION_NOISE")):
        (Path(out_dir) / "diagnostic_flags.json").write_text(json.dumps(flags))   # provenance for the diagnostics
    sys.argv = [sys.argv[0]] + args
    lerobot_eval.main()
    summary = subgoals.write_summary(str(log), str(Path(out_dir) / "eval_info.json"),
                                     str(Path(out_dir) / SUMMARY_NAME))
    print(subgoals.format_summary(summary), flush=True)


if __name__ == "__main__":
    main()
