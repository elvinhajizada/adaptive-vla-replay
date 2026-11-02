"""LeRobot/LIBERO side of the language diagnostic. Protocol and rationale: lang_diag.py.

Imports the pinned LeRobot and LIBERO at module level, so import it only in a shell or job
that sourced scripts/env.sh (LIBERO_CONFIG_PATH, MUJOCO_GL). It sits apart from
lerobot_glue.py so the replay glue stays importable without LIBERO.

Async eval workers start with `forkserver` and rebuild envs from pickled factories, so the
env class lives here (importable by reference) and every override travels as a constructor
argument. Nothing is monkeypatched inside LeRobot or LIBERO.
"""
import functools
import os
from collections import defaultdict
from typing import Dict, List, Sequence

import gymnasium as gym
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs.bddl_utils import robosuite_parse_problem

from lerobot.envs.libero import LiberoEnv, _get_suite, get_libero_dummy_action
from lerobot.envs.utils import NEW_ROLLOUT_OPTION, _LazyAsyncVectorEnv, parse_camera_names

from .lang_diag import SCENE_KEYS, EnvSpec


def suite_languages(suite_name: str) -> List[str]:
    """Instruction of every task in the suite, as LiberoEnv sets task_description."""
    suite = _get_suite(suite_name)
    return [suite.get_task(i).language for i in range(suite.n_tasks)]


def _parse_suite(suite) -> List[dict]:
    return [robosuite_parse_problem(os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file))
            for t in (suite.get_task(i) for i in range(suite.n_tasks))]


class DiagLiberoEnv(LiberoEnv):
    """LiberoEnv for task A (BDDL scene and init states) with a replaced instruction and
    scored goal. The reward is a bitmask of the suite's goals reached so far this episode.
    """

    def __init__(self, *, instruction: str, scored_task_id: int, **kwargs):
        super().__init__(**kwargs)
        self.task_description = instruction      # read by lerobot_eval.rollout via env.call
        self.scored_task_id = scored_task_id
        problems = _parse_suite(kwargs["task_suite"])
        own = problems[self.task_id]
        for k, p in enumerate(problems):
            for key in SCENE_KEYS:
                if p[key] != own[key]:
                    raise RuntimeError(f"task {k} BDDL '{key}' differs from task {self.task_id}: "
                                       "goals of one task cannot be scored in the other's scene")
        self._goals = [p["goal_state"] for p in problems]
        self._mask_at_reset = 0
        self._reached = 0

    def _ensure_env(self) -> None:
        fresh = self._env is None
        super()._ensure_env()
        if fresh:
            self._install_goal()

    def _install_goal(self) -> None:
        dom = self._env.env   # the BDDL problem domain; check_success() evaluates its goal_state
        names = set(dom.object_states_dict)
        for k, goal in enumerate(self._goals):
            missing = {n for pred in goal for n in pred[1:]} - names
            if missing:
                raise RuntimeError(f"goal of task {k} names {sorted(missing)}, absent from task "
                                   f"{self.task_id}'s sim")
        if self._init_states is not None:
            n_sim, n_init = self._env.get_sim_state().shape[0], self._init_states.shape[1]
            if n_sim != n_init:
                raise RuntimeError(f"init state length {n_init} != sim state length {n_sim}")
        dom.parsed_problem["goal_state"] = self._goals[self.scored_task_id]

    def _goal_mask(self) -> int:
        dom = self._env.env
        mask = 0
        for k, goal in enumerate(self._goals):
            if all(dom._eval_predicate(pred) for pred in goal):
                mask |= 1 << k
        return mask

    def reset(self, seed=None, **kwargs):
        obs, info = super().reset(seed=seed, **kwargs)
        self._mask_at_reset = self._goal_mask()
        if self._mask_at_reset >> self.scored_task_id & 1:
            raise RuntimeError(f"scored goal {self.scored_task_id} already holds at reset "
                               f"(task {self.task_id}, init state {self.init_state_id - self._reset_stride})")
        self._reached = 0
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        now = self._goal_mask()
        if bool(info["is_success"]) != bool(now >> self.scored_task_id & 1):
            raise RuntimeError(f"env success {info['is_success']} != goal {self.scored_task_id} "
                               f"(goals holding: mask {now})")
        self._reached |= now & ~self._mask_at_reset
        return obs, float(self._reached), terminated, truncated, info

    def diag_info(self) -> dict:
        self._ensure_env()
        return {
            "task_id": self.task_id,
            "scored_task_id": self.scored_task_id,
            "instruction": self.task_description,
            "installed_goal": self._env.env.parsed_problem["goal_state"],
            "scored_goal": self._goals[self.scored_task_id],
            "own_goal": self._goals[self.task_id],
            "sim_state_len": int(self._env.get_sim_state().shape[0]),
            "init_state_len": None if self._init_states is None else int(self._init_states.shape[1]),
            "mask_at_reset": self._mask_at_reset,
        }


def make_diag_envs(env_cfg, specs: Sequence[EnvSpec], n_envs: int,
                   use_async_envs: bool) -> Dict[str, Dict[int, object]]:
    """{condition: {task A: vec env}}, built like lerobot.envs.libero.create_libero_envs
    (same LiberoEnv arguments from env_cfg, same vector-env classes and autoreset mode)."""
    suite = _get_suite(env_cfg.task)
    gym_kwargs = dict(env_cfg.gym_kwargs)
    gym_kwargs.pop("task_ids", None)
    common = dict(
        task_suite=suite,
        task_suite_name=env_cfg.task,
        camera_name=parse_camera_names(env_cfg.camera_name),
        init_states=env_cfg.init_states,
        episode_length=env_cfg.episode_length,
        n_envs=n_envs,
        control_mode=env_cfg.control_mode,
        camera_name_mapping=env_cfg.camera_name_mapping,
        is_libero_plus=env_cfg.is_libero_plus,
        **gym_kwargs,
    )
    is_async = use_async_envs and n_envs > 1   # lerobot.envs.configs._make_vec_env_cls
    out: Dict[str, Dict[int, object]] = defaultdict(dict)
    cached = None
    for s in specs:
        fns = [functools.partial(DiagLiberoEnv, task_id=s.task_id, episode_index=i,
                                 instruction=s.instruction, scored_task_id=s.scored_task_id, **common)
               for i in range(n_envs)]
        if is_async:
            vec = _LazyAsyncVectorEnv(fns, *(cached or (None, None, None)))
            cached = cached or (vec.observation_space, vec.action_space, vec.metadata)
        else:
            vec = gym.vector.SyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
        out[s.condition][s.task_id] = vec
    return {c: dict(m) for c, m in out.items()}


def selftest(envs: Dict[str, Dict[int, object]], specs: Sequence[EnvSpec]) -> None:
    """Reset every env, check instruction, installed goal and state sizes, step once. No policy."""
    for s in specs:
        vec = envs[s.condition][s.task_id]
        n = vec.num_envs
        vec.reset(seed=list(range(1000, 1000 + n)), options={NEW_ROLLOUT_OPTION: True})
        for info in vec.call("diag_info"):
            assert info["instruction"] == s.instruction, info
            assert info["task_id"] == s.task_id and info["scored_task_id"] == s.scored_task_id, info
            assert info["installed_goal"] == info["scored_goal"], info
            assert info["init_state_len"] in (None, info["sim_state_len"]), info
            if s.condition == "swapped":
                assert info["scored_goal"] != info["own_goal"], info
        _, reward, terminated, _, step_info = vec.step(
            np.tile(np.asarray(get_libero_dummy_action(), dtype=np.float32), (n, 1)))
        assert not np.any(terminated), step_info
        info = vec.call("diag_info")[0]
        print(f"[selftest] {s.condition:8s} A={s.task_id} n_envs={n} ({type(vec).__name__}) "
              f"instruction={s.instruction!r} scored goal={info['installed_goal']} "
              f"own goal={info['own_goal']} sim/init state {info['sim_state_len']}/{info['init_state_len']} "
              f"goals at reset mask={info['mask_at_reset']} reward after 1 no-op step={reward.tolist()}",
              flush=True)
        vec.close()
