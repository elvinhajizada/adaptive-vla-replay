"""Central configuration for Adaptive VLA V1.

The M1 values below were fixed by M1 measurements (2026-10-21 to 23; the experiments are in
docs/report.md). Everything else is a project default.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Recorded pin; requirements.txt installs exactly this commit.
LEROBOT_COMMIT = "9a6bb61043bac8c14353fcb6ea513b7473c118e3"  # LeRobot 0.6.2


@dataclass
class Config:
    # --- start checkpoint (theta0) ---
    policy: str = "smolvla"
    # Released SmolVLA LIBERO fine-tune, trained on lerobot/libero (Spatial/Object/Goal/10).
    # Verified 2026-10-19 (normalizer stats match lerobot/libero), see docs/LIBERO.md section 11.
    pretrained: str = "HuggingFaceVLA/smolvla_libero"
    pretrained_revision: str = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
    # theta0's vlm_model_name is HuggingFaceTB/SmolVLM2-500M-Instruct (load_vlm_weights=True);
    # the Hub redirects it to SmolVLM2-500M-Video-Instruct. Revision pinned 2026-10-19.
    vlm_revision: str = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    # LIBERO sim assets (meshes, textures); hf-libero fetches them lazily, we prefetch.
    libero_assets_revision: str = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"

    # --- new tasks, learned in sequence ---
    new_suite: str = "libero_90"
    dataset_repo: str = "nvidia/LIBERO_LeRobot_v3"
    dataset_revision: str = "e5907374380b8f96511957e6ba5582be52a1e179"
    # libero_90 *env* task ids (LIBERO task_maps order), kitchen scenes 3/4/4/4/6. Map to the
    # dataset by instruction string (libero_data.task_index_for_instruction), never by id:
    # dataset task_index 22, 13, 5, 8, 20. Chosen 2026-10-22 from the 11 single-scene
    # kitchen candidates (zero-shot 0 to 10%); all pass the headroom rule (zero-shot < 30%, and the
    # 8-pass fine-tune's 95% CI lower bound about 20 pts or more above it): 76, 54, 88, 72, 88%.
    # Listed in M2 sequence order (fixed 2026-10-25): 26, 18, 23, 27, 34 (scenes 4, 3, 4, 4, 6).
    new_task_ids: List[int] = field(default_factory=lambda: [26, 18, 23, 27, 34])

    # --- old-skill retention probes (schedule fixed before any replay result) ---
    # Per-stage subsets, fixed by rule before any replay result: the 5 tasks per suite with the
    # highest θ0 success (ties to the lower id), plus Goal task 9 (near-collision with task 27).
    # The final checkpoint runs all 10 Spatial and Goal tasks.
    # Far probe: Spatial tasks, scenes the new tasks don't use (θ0 8, 10, 8, 10, 9 of 10, E06).
    probe_suite: str = "libero_spatial"
    probe_task_ids: List[int] = field(default_factory=lambda: [1, 2, 3, 7, 8])
    # Near-skill probe: Goal tasks, shared objects in another scene (θ0 18, 15, 19, 20, 16, 13 of 20, E12).
    goal_probe_suite: str = "libero_goal"
    goal_probe_task_ids: List[int] = field(default_factory=lambda: [1, 2, 4, 7, 8, 9])
    # Near probe (2026-10-19): the LIBERO-10 tasks sharing a scene with the new
    # tasks: env ids 2 (KITCHEN_SCENE3), 3 (KITCHEN_SCENE4), 9 (KITCHEN_SCENE6); θ0 at 50, 84,
    # 56% (job 113038). KITCHEN_SCENE8's task (8, 16%) is left out: no new task uses scene 8
    # and it sits at the floor. All three at the final checkpoint; the scene-4 task every stage.
    near_probe_suite: str = "libero_10"
    near_probe_task_ids: List[int] = field(default_factory=lambda: [2, 3, 9])
    near_probe_stage_task_ids: List[int] = field(default_factory=lambda: [3])

    # --- training (SmolVLA default recipe: action expert only) ---
    # Fixed passes over each task's demos, not fixed steps (2026-10-22): steps =
    # libero_data.steps_for_passes(passes_per_task, task frames, batch_size). Task 23 learned
    # 54/64/48% at 8/16/24 passes (no detectable difference), forgetting was total at all three.
    passes_per_task: float = 8.0
    batch_size: int = 64                       # LeRobot LIBERO docs recipe (train.sh default)
    lr: float = 1e-4                           # θ0's peak LR (SmolVLA default); cosine to lr / 40,
                                               # warmup and decay auto-scaled to the run length.
                                               # 1e-5 learned less and forgot as much (job 113823)
    update_scope: str = "expert"               # expert | lora_expert | full_vlm (only expert is used)

    # --- replay / selection ---
    # The buffer holds earlier new tasks and θ0's 40 training tasks. Budget: k frames per
    # task, the same for old and new tasks; None = all of a task's frames. Inside replay,
    # sampling is uniform over tasks, then over frames.
    selector: str = "none"                     # none (Sequential) | random | coverage
    replay_k: Optional[int] = 140              # about 2% of a task; picked after the single-stage pilot
    replay_k_pilot: Tuple[Optional[int], ...] = (14, 140, 700, None)   # single-stage pilot sweep
    replay_mix: float = 0.5                    # fraction of each batch from replay (1:1, Liu et al.)

    # --- evaluation (sim rollouts, MUJOCO_GL=egl) ---
    eval_episodes_learned: int = 50            # the task just learned, after its stage (S[i][i])
    eval_episodes_stage: int = 20              # earlier new tasks and the scene-4 probe, per stage
    eval_episodes_probe_stage: int = 10        # Spatial and Goal subsets, per stage
    eval_episodes_final: int = 50              # new tasks and LIBERO-10 probes, final checkpoint
    eval_episodes_final_probe: int = 20        # all 10 Spatial and Goal tasks, final checkpoint
    eval_batch_size: int = 0                   # 0 = one batch per task (batch = episodes per task),
                                               # the protocol: runs stay paired across conditions
    render_size: int = 256                     # θ0's training-frame size (2026-10-22);
                                               # the env default 360 cost 12 pts on Spatial

    # --- seeds ---
    seeds: Tuple[int, ...] = (0, 1, 2)
    seed: int = 0

    # --- io ---
    out_dir: str = "runs"                      # point at DSS for real runs, not $HOME
    device: str = "cuda"

    def replay_frames(self, n_task_frames: int) -> int:
        """Per-task replay buffer size in frames: k, capped at the task's frames."""
        if self.replay_k is None:
            return n_task_frames
        return min(self.replay_k, n_task_frames)
