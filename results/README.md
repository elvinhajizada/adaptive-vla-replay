# results/

Generated tables: every evaluation and training run of the project, collected from the run
outputs by `scripts/collect_results.py` (rerun it after new jobs; do not edit by hand).
`docs/report.md` cites runs by the `run` column below. The experiment ids (E01 to E52) in run
labels, launchers and code comments are listed under "Experiment ids" at the end.

**`runs.csv`**: one row per (eval run, task).

| Column | Meaning |
|---|---|
| run | eval output directory (job id plus a label, `-r256` = render 256) |
| date | date the eval finished (UTC) |
| suite, task_id, task | LIBERO suite, env task id (LIBERO task order), instruction |
| policy | `theta0`, or the training checkpoint evaluated (`train/<run>/checkpoints/<step or last>`) |
| n_episodes, n_success, success_pct | episodes, successes (all BDDL goal predicates hold at once) |
| ci95_lo, ci95_hi | Wilson 95% interval of the success rate (per task), from lerobot-eval |
| render, seed, batch_size | eval settings (protocol: render 256, seed 1000, batch = episodes per task) |
| subgoal_partial | failed episodes that ended with at least one goal predicate newly holding (subgoal-logged runs only) |
| subgoals_final | per goal predicate, the share of episodes where it held at the end |
| repo_commit | code version that ran the eval |

**`train_runs.csv`**: one row per fine-tune (action expert only, batch 64), from θ0 or a previous stage.
CPU plumbing runs are left out.

| Column | Meaning |
|---|---|
| run | training output directory (job id) |
| dataset_task_index, task, task_frames | converted LIBERO-90 task(s) and their frame count |
| steps, passes | training steps and passes over the task's demos (steps x batch x (1 - mix) / frames) |
| k, mix | replay runs (M2): demo frames kept per old task (`all` = every frame) and the replay share of each batch; blank k and mix 0 = no replay |
| replay_frames, replay_mb_obs | replay buffer size: frames, and MB of stored observations |
| replay_seed | seed of the replay-buffer draw (separate from the training seed) |
| seed | training seed (1000 = the lerobot-train default, used by every run before M2) |
| init | start policy: `theta0`, or the checkpoint it continues from (the Sequential chain) |
| peak_lr | peak learning rate (θ0's recipe: 1e-4, cosine to peak / 40) |
| loss_dims | action dims supervised: `all`, or `arm` / `grip` for E26's loss-masked counterfactuals |
| steps, schedule_steps | steps trained, and the length of the LR schedule; they differ only for E35's runs stopped early inside a 1k-step schedule (`AVLA_STOP_STEP`) |
| optimizer | `adamw` (θ0's recipe), `sgd` (E35; peak_lr then gives SGD's LR), or AdamW warmed by zero-LR updates on θ0's data (E35) |
| checkpoints | saved checkpoints (kept slim: only the tensors that differ from θ0) |
| wall_clock_s, repo_commit | training time including start-up, code version |

**`train_curves.csv`**: per training run, loss, grad norm and LR every 50 steps (from the logs).
**`offline_action_error.csv`**: per checkpoint and frame set (spatial, goal, l10_s4 = the scene-4
LIBERO-10 task, task23), mean |error| of the first 10 predicted actions vs the demo and vs θ0, in
θ0 action-std units (translation, rotation, gripper).
**`weight_drift.csv`**: per checkpoint and module group, changed parameters and relative L2 drift.
**`lang_diag.csv`**: language diagnostic, per condition (true, blank, swapped) and outcome.
**`coverage_offline.csv`** (E40): per replayed task (`suite:instruction`), budget k and metric, the
mean, 5th and 95th percentile over 50 random k-frame draws and the greedy k-center value. Metrics on
state + action-chunk features: `radius` (largest distance from a task frame to its nearest selected
frame), `mean_nn` (mean of that distance), `phase_cov` (share of the task's gripper phases with a
selected frame), `demos_cov`, `trans_share` (share of selected frames within 0.25 s of a gripper
toggle; `trans_base` is the task's base rate).
**`coverage_buffers.csv`** (E40): the same metrics for the actual stage-5 replay buffers of E37
(`k14-b0`, `k14-b1`, `k14-b2`) and E39 (`k140-b0`).

**`buffer_memorization.csv`** (E44): per replay run (`memo-k14-b0` ... `memo-k140-b0`), model (`theta0` or
the stage-5 checkpoint) and frame set (`buffered`: θ0 frames in that run's buffer; `unbuffered`: other
frames of the same θ0 tasks, at least 10 rows from any buffered frame; `current`: the stage's own task),
mean |error| of the first 10 predicted actions vs the demo and vs θ0 (std units), as in
`offline_action_error.csv`.
**`failure_anatomy.csv`** (E45, written by `experiments/m2/m2_task_analysis.py`, not by the
collector): per condition (`theta0-E41`, `k14-b0`, ...) and two-step task, final-checkpoint success
and the share of episodes where each goal predicate holds at the end (`step1`, `step2`), in demo order
(`order` = demo) or, for E47's far-scene LIBERO-10 tasks 0, 1, 4, 6, 7, in the order the instruction names
them (`order` = instruction); θ0 rows `theta0-E41`, `theta0-113801` (task 3), `theta0-E47`.
**`skill_margin.csv`** (E45, same script): per Spatial and Goal task, θ0's success in the M1 run
(`theta0_margin`), θ0's reference in E41 (`theta0_ref`), and retention = final success / reference per
replay run and averaged.
**`m2_metrics_ci.csv`** (written by `experiments/m2/m2_metrics_ci.py`, not by the collector): per condition
(`ER k14`, ...), seed (`b0`, ...; `mean` = seed mean) and metric, the estimate and a 95% percentile bootstrap
interval over episodes (episode noise only): `nbt`, normalized forgetting (`_stage5_x20`, `_final_x50`),
`final_<suite>` (final probes), `retention_<suite>` (final / θ0 on the final protocol, θ0's episodes
resampled too; θ0 = every run of the final protocol pooled: `115785`, the M1 runs `113038-goal` and
`113038-l10kitchen`, and `113801` for LIBERO-10 task 3), `learned_task<id>` and
`learning_cost_task<id>` / `learning_cost_mean` (just-learned success minus the mean no-replay reference:
Sequential and the M1 single-task fine-tunes, E48a).
**`phase_coverage.csv`** (E48b, `experiments/m2/phase_coverage.py`): per LIBERO-10 task 2, 3, 9 and k = 140
stage-5 buffer, the step-2 frames in the demos and in the buffer, their shares, and the demos whose step 2
the buffer covers.
**`step_conditional.csv`** (E52, `experiments/m2/step_conditional.py`): per group (`kitchen` = LIBERO-10
tasks 2, 3, 9 in demo order; `far` = E47's tasks 0, 1, 4, 6, 7 in instruction order), task (`pooled` or the
task id) and condition (seeds pooled), the end-of-episode shares `step1`, `step2` and `step2_given_step1` from
each final probe's `subgoals.jsonl`, and for replay conditions retention vs θ0 of step 1 and of step 2 given
step 1 with 95% bootstrap intervals (episodes resampled within run and task, θ0's too) and the interval of
their difference (`diff_lo`, `diff_hi`). θ0: the runs with sub-goal logs (`115785`, `113801`, E47).

**`gripper_*.csv`** (E26, E33, E34, written by `experiments/m1/gripper_forensics.py report`, not by the
collector): teacher-forced on 210 Spatial and Goal frames at four grasp phases (approach, pre-grasp,
hold, pre-release), fixed noise. Model labels as in `offline_action_error.csv`, plus `cf_arm` /
`cf_grip` (train runs `115546-arm` / `115546-grip`).
- `gripper_behaviour.csv`: per model and phase, share of "close" commands where the demo closes, of
  "open" where it opens, mean first gripper command, and change vs θ0 (std units).
- `gripper_patching.csv`: the 1k task-23 checkpoint with one parameter group reverted to θ0, or θ0
  with one group of its change injected; `share_of_ft_grip_change` is the share of the gripper
  change removed (revert) or reproduced (inject).
- `gripper_probes.csv`: ridge-probe AUC for "demo gripper closed" per expert depth, with a probe
  fit on the model's own features and with θ0's probe.
- `gripper_hidden_drift.csv`: relative change and cosine of the expert's residual stream vs θ0.
- `gripper_probes_leakfree.csv` (E33): ridge-probe AUC (own probe, and θ0's probe transferred) per
  model, contrast (`approach_vs_pregrasp`, `near_vs_pregrasp`), denoising step (`t1` = first step,
  input = fixed noise, no leak; `last`) and depth, on 525 frames from 10 demos per task; rows
  `state_only` (the 8-dim robot state as features) and `predicted_gripper` (the model's own gripper
  output as the score) are baselines.
- `gripper_drift_direction.csv` (E34): per model, rms change of the first 10 predicted arm actions vs
  θ0 (std units), the share of that change shared by all frames, the top singular value's share, and
  the mean cosine between each frame's change and (a task's mean action minus θ0's prediction), for
  the tasks the measured fine-tunes trained on (dataset task index 5 = env 26, 8 = 27, 13 = 23,
  22 = 18).
- `gripper_attention.csv`: per layer, attention share of the first 10 action tokens on agentview,
  wrist, language, state and action tokens.
- `gripper_gradients.csv`, `gripper_gradients_by_group.csv`: at θ0, first-order change of the old
  frames' arm and gripper flow loss under one step on the arm part or the gripper part of a new
  task's loss gradient (SGD and Adam-preconditioned), in total and per parameter group.

Caveat: episodes do not repeat bit for bit across runs (GPU nondeterminism), so comparisons
between runs share seeds and initial states, not identical rollouts.

## Report sections, launchers and run labels

For each section of `docs/report.md`: the experiment ids, the launcher or analysis script under `scripts/` or
`experiments/`, and the run labels as they appear in the `run` column of `results/runs.csv` (a
label starts with the job that ran it; `*` stands for the stage or seed parts). Offline analyses
have no run label and point to their own CSV.

| Section | Experiments | Launcher or analysis | Run labels or result file | Planned or added |
|---|---|---|---|---|
| 2.1 Data conversion | E02, E08 | `scripts/convert_libero90.sbatch`, `scripts/train_smoke.sbatch` | none (data checks) | planned |
| 2.2 New tasks | E10, E11, E12, E19 | `experiments/m1/candidate_demos.py`, `experiments/m1/zeroshot_kitchen.sbatch`, `experiments/m1/probe_baselines.sbatch`, `scripts/finetune_eval.sh` | `113037-r256` (zero-shot); `113038-goal-r256`, `113038-l10kitchen-r256` (θ0); `113795-r256`, `113803-r256` to `113806-r256` (single-task fine-tunes) | planned |
| 2.3 Language use | E07 | `scripts/lang_diag.sbatch` | `results/lang_diag.csv` | planned (required check) |
| 2.4 Training length | E13 | `scripts/finetune_eval.sbatch` | `113795-r256`, `113796-r256`, `113797-r256` | planned |
| 2.5 Render size, repeatability | E03 to E06, E17, E29 | `experiments/m1/eval_theta0_spatial.sbatch`, `experiments/m1/eval_theta0_spatial_r256.sbatch`, `experiments/m1/subgoal_check.sbatch`, `scripts/probe_eval.sbatch` | `113009` (360x360), `113024-r256`, `113801-r256`, `115594-theta0-refA-r256`, `115595-theta0-refB-r256` | E06 agreed in advance after E03; E29 planned |
| 2.6 θ0 reference, every run of the final protocol pooled | E41, E12, E17 | `experiments/m2/m2_metrics_ci.py` | `115785-theta0-final-r256`, `113038-goal-r256`, `113038-l10kitchen-r256`, `113801-r256` | changed after review |
| 3 One fine-tune | E14, E18, E21 to E25 | `experiments/m1/forgetting_vs_steps.sbatch`, `experiments/m1/far_probe_check.sbatch`, `experiments/m1/forgetting_onset.sbatch`, `experiments/m1/lr_test.sbatch`, `scripts/probe_eval.sbatch`, `scripts/offline_action_error.py` | `113800-ft23-*`, `113802-ft23-1k-r256`, `114663-ft26-r256`, `114664-ft27-r256`, `113822-s000*`, `113823-*`; drift in `results/offline_action_error.csv` | E14 planned, its result a surprise; the others added after it |
| 3 Sequence without replay | E32 | `experiments/m2/seq_stage.sbatch` | `113804-r256` (stage 1), `115603-seq-*` to `115606-seq-*`, `115615-seq-s5-final-r256` | planned |
| 4 Single-stage pilot | E30 | `experiments/m2/pilot_stage.sbatch` | `115608-t26-k14-r256` to `115613-t23-kall-r256` | planned |
| 4 Budget and retention | E37 (0.2%), E39 (2%, seed 1), E41 (θ0), E42 (2%, seed 2), E43 (every frame), E46 (every frame, seed 2) | `experiments/m2/er_stage.sbatch`, `experiments/m2/er_seed_chain.sh`, `scripts/probe_eval.sbatch` | 0.2%: `115608-*`, `115620-*` to `115641-*` (`k14`, `er14`); 2%: `115609-*`, `115758-*` to `115762-*`, `115786-*` to `115791-*` (`k140`); every frame: `115611-*`, `115792-*` to `115796-*`, `115815-*` to `115820-*` (`kall`); θ0: `115785-theta0-final-r256` | E37 planned; E39 added after E37; E41 and E42 planned; E43 added after E39; E46 added after E43 |
| 5 Failure anatomy, margin | E45 | `experiments/m2/m2_task_analysis.py` | the `*-s5-final-*` labels above; `results/failure_anatomy.csv`, `results/skill_margin.csv` | planned |
| 5 Second sub-goal given the first | E52 | `experiments/m2/step_conditional.py` | `results/step_conditional.csv` (from the final probes' per-episode logs) | added after review, no prediction |
| 5 Far-scene control | E47 | `scripts/probe_eval.sbatch` | `115841-e47-farscene-*` to `115845-e47-farscene-*` | added after E45, rule fixed first |
| 5 Phase coverage | E48 (b) | `experiments/m2/phase_coverage.py` | `results/phase_coverage.csv` | added after E45, prediction first |
| 5 Task 9 at stage 4 | E49 | `scripts/probe_eval.sbatch` | `115846-e49-task9-*`, `115847-e49-task9-*` | added after E46, rule fixed first |
| 5 Proportional split | E50 (E51, a second seed, stopped after two stages) | `experiments/m2/e50_prop_chain.sh` | `115873-*` to `115879-*` (`kp2.05`) | added after E47, rule fixed first |
| 6 Learning cost | E37, E39, E42, E43, E46; E48 (a) | `experiments/m2/m2_metrics_ci.py` | the `libero_90` rows of the sequences above; `113795-r256`, `113803-r256` to `113806-r256`; `results/m2_metrics_ci.csv` | E48 (a) post hoc, no prediction |
| 6 Memorization | E44 | `experiments/m2/buffer_memorization.sbatch` | `results/buffer_memorization.csv` | planned |
| 7 Noise and drift | E29, E37; offline drift in E32, E37, E39, E43, E46 | `scripts/offline_action_error.py` | `115594-theta0-refA-r256`, `115595-theta0-refB-r256`; `results/offline_action_error.csv` | planned |

## Experiment ids

Run labels, launcher comments and analysis scripts name experiments by id. Each one, with the
`docs/report.md` section that uses it ("not reported": the launcher and its outputs are here, but
the report does not discuss the result).

| Id | Question | Report |
|---|---|---|
| E01 | Does the pinned stack run end to end, offline, on GPU and CPU? | not reported |
| E02 | Can the nvidia LIBERO-90 data be converted to θ0's conventions? | 2.1 |
| E03 | How good is θ0 on LIBERO-Spatial with the documented eval settings? | 2.5 |
| E04 | Do the eval's observations match θ0's training frames? | 2.5 |
| E05 | Does `env.step` execute actions like the demos' controller? | 2.5 |
| E06 | Does rendering at θ0's training-frame size (256) change θ0's success? | 2.5 |
| E07 | Does θ0 use the instruction? (language diagnostic) | 2.3 |
| E08 | Does fine-tuning run, and does it keep θ0's normalizer? | 2.1 |
| E09 | How fast is evaluation, and what limits it? | not reported |
| E10 | How many demos does each kitchen candidate have? | 2.2 |
| E11 | Which kitchen candidates can θ0 already do? (zero-shot screen) | 2.2 |
| E12 | θ0 on the candidate retention probes | 2.2, 2.6 |
| E13 | How many training passes does a new task need? | 2.4 |
| E14 | Does a longer fine-tune cost more same-scene retention? | 3 |
| E15 | How far does fine-tuning move θ0's weights? | not reported |
| E16 | Is the slow start of training a data-loader worker-count effect? | not reported |
| E17 | Is the subgoal-logging eval path faithful? Are episodes reproducible? | 2.5, 2.6 |
| E18 | Is the forgetting local to the shared scene? | 3 |
| E19 | Can the other new tasks be learned at 8 passes? | 2.2 |
| E20 | What changes in the policy's actions? (offline action error) | 3 |
| E21 | Does a fine-tune on a task that grasps forget less? | 3 |
| E22 | How fast does the forgetting set in? (onset) | 3 |
| E23 | Does a 10x smaller learning rate forget less? | 3 |
| E24 | Does learning a near-duplicate instruction hurt the matching old skill most? | 3 |
| E25 | Offline action change during onset and at the lower learning rate | 3 |
| E26 | How does a task-23 fine-tune erase grasping? (gripper forensics, `gripper_*.csv`) | not reported |
| E27 | Do the image, fps and chunk-timing checks hold on LIBERO-90? | Appendix A |
| E28 | Does the gripper-masked fine-tune still learn task 23 and keep any old skill? | not reported |
| E29 | How repeatable is a closed-loop eval? θ0 on the per-stage probes | 2.5, 7 |
| E30 | Single-stage pilot: how much replay keeps θ0's skills through one new task? | 4 |
| E31 | How much weight drift can θ0's old skills tolerate? (interpolation) | not reported |
| E32 | The five-task sequence without replay | 3 |
| E33 | Is the old grasp decision still readable in the fine-tuned expert? | not reported |
| E34 | Does the change on old frames point toward the new task's mean action? | not reported |
| E35 | Is the early drift an artefact of a fresh Adam? (optimizer control) | not reported |
| E36 | Does replay protect a suite whose data is not in the buffer? (Goal hold-out) | not reported |
| E37 | Random replay at k = 14 (0.2%), three seeds | 4, 6, 7 |
| E38 | θ0 on LIBERO-Object (reference only) | not reported |
| E39 | Random replay at k = 140 (2%), seed 1 | 4, 6 |
| E40 | Is there a within-task frame-selection lever? (offline buffer coverage) | not reported |
| E41 | θ0 on the final-probe protocol | 2.6, 4 |
| E42 | Random replay at k = 140 (2%), seed 2 | 4, 6 |
| E43 | Replay of every frame, seed 1 | 4, 6 |
| E44 | Does a small buffer get memorized? | 6 |
| E45 | Where does replay fail? Failure anatomy and skill margin | 5 |
| E46 | Replay of every frame, seed 2 | 4, 6 |
| E47 | Is what 2% loses a scene effect or a horizon effect? (far-scene two-step tasks) | 5 |
| E48 | A second no-replay reference (a), and phase coverage of the buffers (b) | 6 (a), 5 (b) |
| E49 | Is full replay's loss on LIBERO-10 task 9 a recency effect? | 5 |
| E50 | Does splitting the 2% budget by task length keep more of LIBERO-10? | 5 |
| E51 | E50 in a second seed (stopped after two stages) | 5 |
| E52 | Is the second step lost beyond what losing the first implies? | 5 |

## Provenance

Every run records the commit of the code that produced it (`repo_commit` in `runs.csv` and
`train_runs.csv`, and `run_info.json` next to each raw run). Those commits belong to the private
repository the project was developed in; they do not exist here. This repository starts from one
commit exported from that repository at the commit named in `SOURCE`. The export copies a fixed
list of files and changes them in two mechanical ways only:
- Slurm `#SBATCH` headers drop the cluster, node and absolute log-path lines and get a placeholder
  partition (`<your-partition>`);
- blocks in `scripts/env.sh` and `scripts/build_env.sh` marked as site-specific (module paths,
  storage location) are removed.

No code logic changes in the export. Before it, the private repository was prepared for release
with changes that do not alter what a run computes: comments, launchers that no longer assume a
fixed checkout path, an environment build that installs the lock file, provenance that tolerates
a missing `.git`, and the figure script reduced to the report's figures.

The CSVs here are the output of `scripts/collect_results.py` and the analysis scripts named above.
Those read the raw run directories (checkpoints, per-episode eval outputs and logs), which are not
published, so they cannot be rerun from this repository alone; `scripts/make_figures.py` needs only
the CSVs.
