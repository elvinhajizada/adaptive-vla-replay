# LIBERO: dataset and benchmark reference

Compiled 2026-10-19, revised 2026-10-26. Items marked **(verified)** were checked
directly against Hub metadata, parquet shards, decoded video frames, the pinned LeRobot
0.6.2 install or the cited paper's own tables. Items marked **(unverified)** come from
papers, docs or secondary summaries and were not re-checked.

## 1. What LIBERO is

LIBERO (Liu et al., *Benchmarking Knowledge Transfer for Lifelong Robot Learning*,
NeurIPS 2023 Datasets & Benchmarks) is a simulated manipulation benchmark built on
robosuite/MuJoCo with a Franka Panda arm.

- **Task definitions.** Each task is a BDDL file that defines the scene, the objects,
  where they may be placed initially, a goal predicate and one language instruction.
- **Success.** Success is checked by the goal predicate, so there is no learned or
  hand-tuned reward.
- **Purpose.** It was built for lifelong (continual) learning. Today it is mainly the
  standard benchmark for fine-tuning VLAs.

## 2. Where the data comes from

All LIBERO data is **human teleoperation inside the simulator**. A person drove the
simulated arm with a 3D SpaceMouse until the task succeeded. One such recorded attempt
is a **demo** (also called an episode or trajectory). There is no real-robot data and
no scripted data.

The simulator plays two more roles:

1. **Re-rendering.** Raw demos store the full simulator state and the actions, so they
   can be replayed to render new images. OpenVLA re-rendered everything at 256×256,
   dropped replays that failed and removed idle (zero-motion) steps. Most VLA work, and
   every LeRobot version below, descends from this filtered set, which is why suites
   have fewer than 50 demos per task.
2. **Evaluation.** The policy drives the simulated robot from fixed initial states, and
   the goal predicate scores success.

## 3. Hierarchy

```
LIBERO (130 tasks)
└── Suite: Spatial | Object | Goal | 10 (= Long) | 90
    └── Task = one BDDL file: scene + objects + goal predicate + one instruction
        ├── Training: 50 human demos (34 to 50 after filtering)
        │   └── Demo: ~100 to 500 steps at 20 Hz (LIBERO-90 mean ≈ 145, Long up to 505)
        │       └── Step (= frame = transition): 2 RGB images, 8-D state, 7-D action
        └── Evaluation: 50 fixed initial states (object placements), no demos
```

| Suite | LeRobot name | Tasks | What varies between tasks | Max steps | Longest demo |
|---|---|---|---|---|---|
| LIBERO-Spatial | `libero_spatial` | 10 | Same objects, different layout | 280 | 193 |
| LIBERO-Object | `libero_object` | 10 | Same layout, different object | 280 | 254 |
| LIBERO-Goal | `libero_goal` | 10 | Same scene, different goal (only the instruction tells tasks apart) | 300 | 270 |
| LIBERO-Long | `libero_10` | 10 | Two-step, long-horizon tasks | 520 | 505 |
| LIBERO-90 | `libero_90` | 90 | Short-horizon tasks across 20 scenes | 400 | 373 |

**Max steps (verified).** Both step columns are from `TASK_SUITE_MAX_STEPS` in the pinned
LeRobot 0.6.2 (`lerobot/envs/libero.py`). Suites not in that table get 500, and
`--env.episode_length` overrides it. Other eval scripts use other limits (OpenVLA's uses
220 for Spatial, unverified), which is one more source of protocol drift.

**Scenes.** LIBERO-90 and LIBERO-10 together are "LIBERO-100". They are built on 20
scenes (10 kitchen, 6 living room, 4 study), and several tasks share a scene.
Spatial, Object and Goal have their own setups.

## 4. Data format

### Step contents (verified)

| Field | Shape | Meaning |
|---|---|---|
| Main image | 256×256×3 uint8 | Third-person "agentview" camera, upright |
| Wrist image | 256×256×3 uint8 | `eye_in_hand` camera |
| `observation.state` | 8 | EEF xyz (m, world frame; table height varies by scene), EEF orientation as axis-angle (≈ [π, 0, 0] with the gripper pointing down; components can exceed π), 2 finger positions (±0.039 when open) |
| `action` | 7 | Delta EEF pose from robosuite's OSC controller (3 translation, 3 rotation), clipped at ±0.9375 in the data, plus 1 gripper value |
| task | via `task_index` | One instruction string per task |

There are no rewards or success labels in the LeRobot versions, because every stored
demo succeeded.

### Raw HDF5 (unverified)

- One file per task.
- `data/demo_i/` holds `actions`, `states` (full MuJoCo state), `robot_states`,
  `rewards`, `dones`, and `obs/` with `agentview_rgb` and `eye_in_hand_rgb` (128×128),
  `ee_pos`, `ee_ori`, `ee_states`, `gripper_states`, `joint_states`.
- Raw renders are upside down.

### LeRobot v3 layout (verified)

```
<suite>/
  data/chunk-000/file-*.parquet    # one row per step, many episodes per file
  videos/<camera>/chunk-000/*.mp4  # AV1 video, many episodes per file
  meta/info.json                   # features, shapes, fps
  meta/stats.json                  # mean/std/min/max per feature
  meta/tasks.parquet               # instruction <-> task_index
  meta/episodes/*.parquet          # per-episode length, file index, video time window
```

## 5. Dataset versions

| Source | Format | Content |
|---|---|---|
| LIBERO GitHub download | HDF5, 128×128 | Raw demos, all 130 tasks |
| `openvla/modified_libero_rlds` | RLDS, 256×256 | Re-rendered, filtered, 4 standard suites |
| `physical-intelligence/libero` | LeRobot | openpi's version |
| `lerobot/libero` (1.9 GB, AV1 video) and `HuggingFaceVLA/libero` (70 GB, PNG) | LeRobot v3 | 4 suites: 40 tasks, 1,693 demos, 273,465 frames |
| `nvidia/LIBERO_LeRobot_v3` (4.5 GB, AV1 video) | LeRobot v3 | All 5 suites: 113 task ids, 5,614 demos, 842,714 frames |

### nvidia vs lerobot conventions (verified)

The four standard suites in the nvidia set are **the same demos** as in
`lerobot/libero`. Episode lengths match, state and arm actions are identical, and
decoded frames differ by about 1/255. Only the conventions differ:

| | `lerobot/libero` | nvidia (incl. LIBERO-90) |
|---|---|---|
| Wrist camera key | `observation.images.image2` | `observation.images.wrist_image` |
| **Gripper action** | **-1 open, +1 close** | **1 open, 0 close** |
| fps label | 10 | 20 (the true control rate is 20 Hz) |
| Image orientation | upright | upright |
| Extra keys | none | `observation.states.{ee_state, gripper_state, joint_state}` (libero_90 only; they duplicate `observation.state`) |

Converting nvidia data to lerobot conventions:
1. Rename `wrist_image` to `image2`.
2. Convert the gripper: `g = 1 - 2 * g_nvidia`, and assert every value is in {-1, 1}.
3. Drop the `observation.states.*` keys.
4. Write θ0's normalizer stats (action and state mean and std) into the converted
   dataset's `meta/stats.json`; never use nvidia's (wrong gripper convention). When
   fine-tuning from a pretrained path (not resuming), the pinned `lerobot-train`
   overwrites the policy's normalizer and unnormalizer stats with `dataset.meta.stats`
   (`lerobot_train.py` lines 533 and 556 to 558 at `9a6bb61`), which is read from
   `meta/stats.json` and not recomputed for an episode subset. Carrying θ0's stats
   makes that overwrite a no-op; drifting stats would move old skills too and show up
   as fake forgetting.

If you skip step 2, a policy trained on lerobot data learns to drive the gripper
backwards, and nothing raises an error. This repo implements steps 1 to 4 in
`adaptive_vla/libero_data.py` (`scripts/convert_libero90.py`).

The fps labels differ but the data does not: both sets step at 20 Hz, so a 50-step
action chunk spans the same 50 consecutive control steps in either (checked on LIBERO-90:
image orientation, fps labels and chunk timing against the simulator).

### LIBERO-90 in the nvidia set (verified)

- **Tasks.** The 90 BDDL tasks have only 74 unique instructions. The dataset keys
  `task_index` by instruction, which gives 73 ids: "pick up the butter and put it in
  the basket" has no demos.
- **Size.** 3,921 demos, about 145 frames each.
- **Merged scenes.** 12 instructions exist in 2 or 3 scenes and are merged under one
  `task_index`, with 76 to 146 demos and no scene label. Examples: "open the top
  drawer of the cabinet" (KITCHEN_SCENE1 and 2), "turn on the stove" (KITCHEN_SCENE3
  and 9), and the three "pick up the book and place it in the
  front/left/right compartment of the caddy" (STUDY_SCENE1, 2, 3).
- **Single-scene tasks.** The other 61 instructions have 34 to 50 demos each.
- **Mapping.** Map dataset tasks to simulator tasks by instruction string plus scene
  name, never by integer id.

## 6. How LIBERO is used: two protocols

| | Lifelong protocol (LIBERO paper) | VLA protocol (most papers since 2024) |
|---|---|---|
| Model | Small policies from scratch (ResNet-RNN, ResNet-T, ViT-T) | Large pretrained VLA |
| Training | **Sequential**: tasks T1…T10 in order, only Tk's demos at stage k (50 epochs), plus whatever the algorithm keeps (ER memory, EWC penalty, PackNet masks) | **Joint**: one fine-tune on all demos of a suite, or all 4 suites pooled |
| Evaluation | Every 5 epochs on the current task; after each stage, the best checkpoint gets 20 rollouts (max 600 steps) on every task seen so far, giving a success matrix | Once at the end: 50 rollouts per task (LeRobot recommends 10 per task × 3 seeds) |
| Metrics | FWT, NBT (forgetting, lower is better), AUC (see below) | Success rate per suite |
| Question | How much is forgotten, and how much transfers? | How well does the model imitate these tasks? |

The VLA protocol is the LIBERO paper's multitask (MTL) upper-bound condition, so it
cannot measure forgetting. In the paper, LIBERO-90 is used for pretraining and
LIBERO-Long for the downstream lifelong run.

### Lifelong metrics (verified, LIBERO paper Eq. 3)

Notation: K tasks learned in order; c(i, j, e) is the success on task j after e epochs
of training on task i; c(k, k) is the best success on task k during its own training.
Each metric is the mean of its per-task value over the K tasks.

- **FWT_k** = mean of c(k, k, e) over e ∈ {0, 5, …, 50}: the area under task k's own
  learning curve. Despite the name it is not measured against a from-scratch baseline;
  it rewards learning the new task fast and well.
- **NBT_k** = (1 / (K − k)) · Σ over τ > k of [c(k, k) − c(τ, k)]: the mean drop on task
  k over all later stages. Lower is better; it can be negative (later stages helped).
- **AUC_k** = (1 / (K − k + 1)) · [FWT_k + Σ over τ > k of c(τ, k)]: learning and
  retention in one number.

A low NBT on its own can mean "learned nothing, so lost nothing": check it next to FWT.
This repo's `adaptive_vla/metrics.py` uses the stage-end success for c(k, k) instead of
the best one seen in training, and reports normalized forgetting, (c(k, k) − c(K, k)) /
c(k, k), next to NBT.

## 7. Published numbers

### VLA protocol: joint training, not continual learning

| Model | Training regime | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| Diffusion Policy (scratch) | per suite | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| Octo | per suite | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA (7B) | per suite (LoRA) | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π0-FAST | 4 suites pooled | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| SmolVLA (0.45B) | multi-task | 90 | 96 | 92 | 71 | 87.3 |
| π0 | 4 suites pooled | 96.8 | 98.8 | 95.8 | 85.2 | 94.2 |
| π0.5 (openpi) | 4 suites pooled | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| OpenVLA-OFT | per suite | 97.6 | 98.4 | 97.9 | 94.5 | 97.1 |
| AtomVLA (2026) | SFT + post-training | 96.4 | 99.6 | 97.6 | 94.4 | 97.0 |
| HarnessPAI (π0.5, 2026) | harness on π0.5 | 97.2 | 100.0 | 97.2 | n/a | 98.1 (3 suites) |
| **θ0, our eval (render 256)** | 4 suites pooled | **76** | not run | **76** | **51.5** (4 tasks) | |

- **Sources (checked 2026-10-26).** Diffusion Policy, Octo, OpenVLA, π0-FAST, π0 and
  OpenVLA-OFT match OpenVLA-OFT's LIBERO table; OFT's paper states it fine-tunes "on each
  task suite independently" (the variant with wrist camera, proprio and the filtered
  dataset). SmolVLA matches its own Table 2, which says "multi-task" without saying the 4
  suites were pooled. The π0.5 row was checked earlier against openpi. HarnessPAI and
  AtomVLA's Spatial/Object/Goal are from HarnessPAI's Table 4; AtomVLA's Long and average
  are from a summary of its paper (unverified).
- **Mixed averages.** HarnessPAI averages only Spatial, Object and Goal, and lists AtomVLA
  at 97.9 on the same 3-suite basis, against 97.0 over 4 suites in AtomVLA's own table.
  LaMem-VLA's 97.6% average comes from press coverage only (unverified). Read "2026 SOTA
  about 97 to 98%" with that in mind.
- **Same model, different numbers.** π0 is 94.2 in openpi's own numbers but 86.0 in the
  SmolVLA paper's table, and π0.5 is 96.9 in openpi's but 95.6 / 100.0 / 95.2 in
  HarnessPAI's. Compare only within one paper's protocol.
- **θ0 reproduction.** θ0 is the released SmolVLA LIBERO checkpoint; its model card gives
  no LIBERO numbers. Our eval: Spatial 76/100, Goal 152/200, the 4 LIBERO-10 kitchen tasks
  103/200 (render 256, seed 1000). On the final probes of the report (20 episodes per
  Spatial and Goal task, LIBERO-10 tasks 2, 3 and 9) θ0 scores 73.5, 77 and 61% (report, section 2.6).
  Community runs of the checkpoint reach about 80% (unverified), and
  the SmolVLA paper reports 90% for Spatial.
- **LeRobot's π0.5 reproduction:** 97.5% (unverified).

### Lifelong protocol

- **LIBERO paper** (verified, Table 8: LIBERO-Long, ResNet-T, mean ± standard error over
  3 seeds):

  | Method | FWT | NBT | AUC |
  |---|---|---|---|
  | SeqL | 0.54 ± 0.01 | 0.63 ± 0.01 | 0.15 ± 0.00 |
  | ER | 0.48 ± 0.02 | 0.32 ± 0.04 | 0.32 ± 0.01 |
  | EWC | 0.13 ± 0.02 | 0.22 ± 0.03 | 0.02 ± 0.00 |
  | PackNet | 0.22 ± 0.01 | 0.08 ± 0.01 | 0.25 ± 0.00 |

  Sequential fine-tuning has the highest FWT: every CL method costs forward learning.
  EWC's lower NBT comes with FWT 0.13, so it mostly learned less. The paper also reports
  that pretraining sometimes hurt (not re-checked).
- **Liu H. et al. 2026** (arXiv 2603.03818, "Pretrained VLA models are surprisingly
  resistant to forgetting"; checked 2026-10-26 against v2). The closest published
  comparison to our replay experiments.
  - *Setup.* π0 (3B) and GR00T N1.5 (3B) against three small policies trained from
    scratch (BC-Diffusion Policy, BC-Transformer, BC-ViT). Each of the 4 suites is its
    own 10-task sequence in a fixed order, one model per suite. 10k steps per task,
    training data from OpenVLA's filtered LIBERO set. GR00T keeps vision and language
    frozen and fully fine-tunes the action head, the same update scope as θ0's expert-only
    fine-tune. π0's scope is stated inconsistently (Table 8). Seed and episode counts are
    not given.
  - *Metric.* NBT = success on task k right after learning it minus success on k after
    each later task, averaged. It is an absolute success drop, so a task the model
    never learned well cannot show much forgetting. The authors note this and add a
    normalized variant (appendix only).
  - *Main result.* ER with 1000 transitions per task (about 15 to 20% of a task's data),
    sampled 1:1 with current data, gives almost no forgetting: average NBT -0.016 (π0)
    and 0.027 (GR00T), against 0.13 to 0.25 for the scratch policies. Without replay,
    both VLAs forget heavily: sequential NBT 0.70 (π0) and 0.75 (GR00T) on Object, 0.56
    and 0.76 on LIBERO-10. EWC barely helps (GR00T Object: EWC 0.766, ER 0.004). So the
    "resistant" in the title holds only with replay.
  - *Buffer ablation.* Average NBT over the suites at 100 transitions per task (about
    2%): π0 0.12, GR00T 0.18, scratch policies 0.46 to 0.52. At 10 per task (about
    0.2%): π0 0.58, GR00T 0.75, scratch 0.60 to 0.78. Pretraining helps most at the
    middle budget; at 10 transitions nothing holds. Selection can only matter where
    random replay leaves forgetting, which is this 0.2 to 2% range.
  - *Why pretrained VLAs resist.* π0 regains its peak success on a forgotten task in 6
    to 11% of the original training steps (BC-Transformer: 33 to 187%), so much of the
    lost behaviour is still recoverable. Swapping components between checkpoints puts
    most of π0's forgetting in the vision-language backbone, not the action head (in
    θ0, only the action expert is trained). Low NBT in scratch policies partly
    reflects that they learned new tasks poorly, not that they kept old ones.
  - *Difference from our setup.* Their old tasks are learned inside the sequence, and
    ER replays them. Our old skills are θ0's pretraining tasks, probed after new
    LIBERO-90 tasks, and replay draws from θ0's own training data.
- **PHASER** (arXiv 2606.03598, checked 2026-10-26 against v2). The closest published
  replay-*selection* method, and the reference point for a replay selector.
  - *Problem.* Uniform frame-level replay fills the buffer in proportion to phase
    length, so short, critical phases (grasp, contact transitions) get few frames and
    are forgotten first ("phase starvation"). Causal check: emptying the buffer slots of
    the first phase drops success to 16%, below plain ER (31%) and full PHASER (49%).
  - *Method.* Split each demo into 2 to 5 sub-skill phases (a 5-minute human annotation
    per task, or Auto-PC: a gripper and velocity change-point detector plus GPT-4o).
    Give every phase the same frame budget (the "phase floor"). Then weight replay
    toward past phases that look similar to the current task in language and vision but
    need different actions (tri-modal prototypes, interference score).
  - *Setup.* OpenVLA-OFT-7B and two Qwen2.5-VL-3B VLAs, rank-32 LoRA, 10k steps per
    task, LIBERO-Goal and LIBERO-Long as 10-task sequences, 50 rollouts per task.
    Buffers: 110 (Goal) or 180 (Long) frames per task for the 7B model, about one demo's
    worth; 1000 per task for the 3B models. The total buffer matches ER within 10%.
  - *Results.* Success after the sequence beats uniform ER in all 6 model x suite cells,
    by +10 to +40 points. The abstract's "up to 31" is the 7B on Long (85.8 vs 54.6). The
    largest gap in the table is a 3B model on Goal (79.0 vs 39.4). MIR and iCaRL fall in
    between, and NBT is not lowest for PHASER in every cell.
  - *Ablation.* The phase floor carries the gain. Interference routing alone, without
    the floor, drops the 7B on Long to 23.4 (from ER's 54.6); adding the floor restores
    85.8. A coverage-style selector is close to the floor: it buys even coverage of the
    trajectory instead of frames in proportion to time spent.
  - *Limits.* LIBERO only, uneven seed reporting, Auto-PC fails badly once (Goal: 63.6
    vs 87.8 with human phases). Overhead about 3 to 5% of training time.

## 8. Transitions and replay budgets

A **transition** is one timestep: the (observation, action) pair at time t, i.e. one
parquet row. "Frame", "step" and "sample" are used for the same thing.

| | Transitions |
|---|---|
| One LIBERO-90 demo | ~145 on average |
| One LIBERO-90 task (34 to 50 demos) | ~5k to 7k |
| Liu et al.'s 1000 per task | ≈ 7 demos, ~15 to 20% of a task |
| 2% budget | ≈ 140, about one demo's worth |
| 0.2% budget | ≈ 14 |

Chunked policies such as SmolVLA (`chunk_size=50`) train on the observation at t plus
actions t…t+49. A replay buffer of frame indices into the dataset gets the chunk for
free; a buffer of copied single frames does not.

The same frame count can also cover a task very differently: 140 frames scattered
across demos cover far more than one contiguous 140-frame demo. That gap is where a
coverage-based selector can beat random replay. Check whether a paper's budget counts
frames or chunk starts before matching it.

## 9. Caveats

- **Saturation.** For the strongest models, every suite but Long is at about 97 to 100%,
  so small differences between them are noise. θ0 (about 76%) is far from that ceiling.
- **Memorization.** Evaluation uses the training tasks; only the initial object
  placements shift, and only slightly.
  - **LIBERO-PRO** (arXiv 2510.03827) reruns the LIBERO suites with controlled
    perturbations along five axes: object appearance (colour, texture, size), initial
    object positions, instruction wording (paraphrases), environment (background,
    lighting, table texture) and the task itself (new target object or action). It tests
    OpenVLA (one checkpoint per suite), π0 and π0.5, all above 90% on standard LIBERO, at
    50 episodes per task.
    - Position shifts are the sharpest test. OpenVLA and π0 reach 0% once objects move
      more than about 0.2 units from their trained spots; π0.5 holds out to about 0.4
      before it also reaches 0. On Goal with moved objects: π0.5 38%, the other two 0%.
    - The failure modes show what was learned. Told to pick up the salad dressing, with
      the dressing swapped for alphabet soup, the policy grasps the soup; with the
      dressing removed, it grasps at the empty spot where it usually stands. A nonsense
      instruction ("xxxx", "fdsgfdsgsd") yields almost the same trajectory as the real
      one. A policy that solves "pick up the bowl and place it on the stove" and "turn on
      the stove" separately fails the two combined.
    - The authors' reading: LIBERO success mostly measures a memorized mapping from a
      familiar scene layout to a trajectory, not instruction understanding or visually
      guided control.
  - **LIBERO-Plus** (arXiv 2510.13626, CVPR 2026) is broader: 10,030 perturbed task
    variants over 7 dimensions (object layout, camera viewpoint, robot initial state,
    language, lighting, background texture, sensor noise) and 10 models, among them
    OpenVLA-OFT, π0, π0-FAST, UniVLA and WorldVLA.
    - Camera and robot-start perturbations hurt most: π0 falls from 94.2% to 15.8%
      (camera) and 6.6% (robot start), OpenVLA-OFT from 97.1% to 59.7% and 37.2%.
      Background and lighting hurt least, which the authors credit to the wrist camera.
    - Language perturbations cost about 25 points on average, one of the smaller drops,
      and the paper reads this as indifference, not robustness. Blanking the instruction
      leaves OpenVLA-OFT's Object-suite success largely unchanged. When the instruction
      names a different target object, success falls to near 0 because the policy fetches
      the original one. Only LIBERO-Long degrades clearly without language; Goal is not
      analysed separately.
    - Fine-tuning OpenVLA-OFT on about 20k perturbed demos raises its average from 67.9%
      to 79.5% (camera +37 points), but robot-start success stays at 30%, and the
      training perturbations come from the same families as the test ones. That shows
      the value of coverage, not generalization to unseen shifts.
  - **For this project.** Neither paper tests SmolVLA. Our checks agree with them on
    memorization (θ0 does whole LIBERO-10 tasks but not their halves, section 11). On language
    they only seem to disagree: on 3 Goal tasks θ0 follows a swapped instruction and does
    nothing under a blank one (section 11), but Goal's tasks share one scene, so language is
    required there. That is LIBERO-Plus' own exception (language matters where the scene
    does not determine the task: Long there, Goal here), and the swap selects among
    trained instructions rather than testing new ones. Our probes reuse
    the training initial states, so a forgetting number here is damage to memorized
    scene-to-trajectory mappings (section 11).
- **Protocol drift.** Papers differ in idle-step filtering, image flips, wrist-camera
  use, joint vs per-suite training, episode count, max episode steps, render size,
  `n_action_steps` and action parameterization (relative vs absolute). Compare only
  against your own runs under the same protocol.
- **Statistics.** At 10 episodes per task, the standard error is about ±15 points per
  task. LeRobot recommends averaging over 3 seeds.
- **Episodes do not repeat exactly (measured).** Identical evals (same
  checkpoint, seed and init states) do not give identical rollouts. Once: 42/50 vs 41/50
  on the scene-4 probe, with 9 of 50 episodes flipping outcome. Twice, two identical runs
  of the per-stage probe protocol: 122 of 130 episodes agree (8 flips, 5 one way and 3 the
  other), and no probe subset moves by more than 3 episodes. The cause (GPU kernels or
  the simulator) has not been tested. A matching total hides episode-level noise, so
  compare conditions as paired runs on the same seeds and treat per-episode pairing as
  approximate.
- **Data revisions.** Hub datasets get re-uploaded, so pin `--dataset.revision` when
  reporting.

## 10. Practical notes (LeRobot 0.6.2)

- **Install:** LeRobot's LIBERO extra (`pip install -e ".[libero]"` in a LeRobot checkout, Linux only).
  This repository installs pinned versions instead (README, step 1).
- **Headless rendering:** `MUJOCO_GL=egl` on a GPU machine, `osmesa` on a CPU-only one.
- **Render size: 256×256.** Pass `--env.observation_height=256
  --env.observation_width=256` to match θ0's training frames. The env default of 360×360
  cost θ0 12 points on Spatial (64% vs 76%, same seeds and init states, paired McNemar
  p = 0.029), although the 360 and 256 frames differ by only 0.005 pixel MAE.
  Treat every eval setting as result-changing.
- **Eval flags:** for reported numbers, use `--env.init_states=true` with hard resets
  (the default). Soft resets are faster but not bit-identical. Use the same `--seed`
  across conditions so comparisons are paired.
- **Control mode:** `--env.control_mode` must match the checkpoint (`relative` for
  delta actions).
- **Env outputs:** `observation.images.image`, `observation.images.image2`,
  `observation.state` (8), and actions in `Box(-1, 1, (7,))`.
- **Video decoding:** the videos are AV1, so the decoder needs dav1d (PyAV wheels
  include it).
- **Offline compute nodes:** download checkpoints and datasets once on a machine with
  internet, then run jobs with `HF_HUB_OFFLINE=1`. LIBERO also needs its
  `.libero/config.yaml`: without it, `import libero.libero` calls `input()` and a batch
  job hangs.
- **CPU-bound rollouts:** simulation is the bottleneck, so request enough cores for
  parallel envs.

## 11. Implications for this project

State as of 2026-10-26. The evidence for each point is in `docs/report.md`.

### θ0

- **Identity.** θ0 is `HuggingFaceVLA/smolvla_libero` @ `6721902b`. Its normalizer
  stats equal `lerobot/libero`'s, so it was trained jointly on the 4 suites (verified).
- **Inputs.** Two camera slots (`image`, `image2`) and `empty_cameras=0`, so no
  slot mapping is needed (the three-slot layout belongs to `smolvla_base`).
- **Replanning.** `n_action_steps=1`, so θ0 replans every step. Keep this for
  reproduction and log it.
- **Reference numbers.** θ0's own evaluation, not the paper's numbers, is the baseline for
  every forgetting claim: on the final probes Spatial 73.5%, Goal 77%, LIBERO-10 tasks 2, 3
  and 9 61%, at render 256 (report, section 2.6; the first evaluations are in section 7).
- **Training data for new tasks.** The new tasks come from `nvidia/LIBERO_LeRobot_v3`
  @ `e5907374`, so apply the section 5 conversion. Keep θ0's normalizer stats fixed
  for every stage: drifting stats would shift old skills too and look like forgetting.
  When fine-tuning from a pretrained path, `lerobot-train` overwrites the policy's stats
  with the dataset's `meta/stats.json`, so the converted dataset carries θ0's stats.

### Task pool and chosen tasks

- **Excluded:** the 12 multi-scene instructions, plus "pick up the book and place it
  in the back compartment of the caddy" (STUDY_SCENE2), which is word-for-word a θ0
  LIBERO-10 instruction.
- **Remaining pool:** 60 single-scene instructions.
- **Familiar-scene confound.** θ0 has already seen 9 LIBERO-90 scenes through
  LIBERO-10: KITCHEN_SCENE3, 4, 6 and 8, LIVING_ROOM_SCENE1, 2, 5 and 6, and
  STUDY_SCENE1.
- **Chosen.** LIBERO-90 env tasks 18, 23, 26, 27 and 34 (frying pan on the stove;
  close the bottom drawer and open the top one; wine bottle in the bottom drawer; wine
  bottle on the wine rack; mug in front of the white mug), trained in the order 26, 18,
  23, 27, 34. Their scenes are 4, 3, 4, 4 and 6: all familiar, so every new task is a
  new skill in a scene θ0 knows. Tasks 23 and 26 share sub-steps with the scene-4 probe
  (stated confound).

### Memorization and language

- **Retention claims.** The retention probes measure retention of memorized,
  in-distribution behaviour on fixed initial states. Word results that way.
- **θ0 reproduces whole tasks, not sub-skills.** θ0 does the LIBERO-10 task "put
  the black bowl in the bottom drawer of the cabinet and close it" 84% of the time, but
  its two halves (LIBERO-90 tasks in the same scene) only 0/10 and 1/10.
- **θ0 does use the instruction.** On 3 LIBERO-Goal tasks, 30
  episodes per condition: true instruction 83%, blank 0%, swapped (task A's initial
  state, task B's instruction) 87% reaching B's goal and 0% A's. Same-scene new tasks are
  therefore distinct skills to θ0, and interference between them is not just an artefact
  of ignored language. Scope: three tasks of one suite.

## Sources

- LIBERO paper: https://arxiv.org/abs/2306.03310
- LIBERO code and BDDL files: https://github.com/Lifelong-Robot-Learning/LIBERO
- LeRobot LIBERO docs: https://huggingface.co/docs/lerobot/libero
- Datasets: https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3 ,
  https://huggingface.co/datasets/lerobot/libero
- θ0: https://huggingface.co/HuggingFaceVLA/smolvla_libero
- OpenVLA: https://arxiv.org/abs/2406.09246 ; OpenVLA-OFT: https://arxiv.org/abs/2502.19645 ;
  openpi: https://github.com/Physical-Intelligence/openpi
- SmolVLA: https://arxiv.org/abs/2506.01844
- LIBERO-PRO: https://arxiv.org/abs/2510.03827 ; LIBERO-Plus: https://arxiv.org/abs/2510.13626
- Continual learning: Liu et al. arXiv 2603.03818 ; PHASER arXiv 2606.03598
- 2026 SOTA: AtomVLA arXiv 2603.08519 ; HarnessPAI arXiv 2609.29166 ; LaMem-VLA arXiv
  2607.07608
