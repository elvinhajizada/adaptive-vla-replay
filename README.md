# Forgetting and replay in a deployed SmolVLA

How much does a deployed vision-language-action model forget when it is taught new manipulation
skills, and how much of its own training data has to be replayed to keep its old ones?

![θ0 succeeds; after learning a new task in the same kitchen, the same skill fails; the new skill works](docs/figures/m1_forgetting.gif)

*One fine-tune erases an old skill. Left: θ0 does the LIBERO-10 task "put the black bowl in the
bottom drawer of the cabinet and close it". Middle: after a single fine-tune (1,000 steps) on new
task 23 in the same kitchen, the same task from the same start fails. Right: the fine-tuned policy
does task 23 itself (close the bottom drawer, open the top one). Played at about 1.5x real time.*

We start from the released SmolVLA LIBERO checkpoint (θ0, `HuggingFaceVLA/smolvla_libero`, 605M
parameters, trained on LIBERO-Spatial, Object, Goal and LIBERO-10), teach it five new LIBERO-90
kitchen tasks one after another with its own fine-tuning recipe (action expert only), and re-test
its old skills: LIBERO-Spatial (other scene), LIBERO-Goal (other scene, shared objects) and the
LIBERO-10 tasks in and outside the new tasks' kitchens. Fine-tuning without replay is compared with
random replay of θ0's own data and the earlier new tasks at budgets from 14 frames per task to
every frame.

The full write-up, with methods, per-run tables, predictions and limitations, is
**[docs/report.md](docs/report.md)**.


## Findings

1. **One fine-tune erases θ0's skills.** After learning a single new task, θ0's
   old skills drop to 0–27% (from 74–84%); on the same-scene task, the only one tested during
   training, the skill is gone by step 100, before the new task is learned. A 10x lower learning rate
   does not prevent it.
2. **Without replay, a sequence keeps only the last task.** Every earlier task, and every old θ0
   skill, is at 0% after five stages.
3. **Replaying θ0's own data protects it, and the budget sets how much.** After five new tasks:

   | Replay budget (frames per task, share of the data) | Seeds | LIBERO-Spatial | LIBERO-Goal | LIBERO-10, same kitchens |
   |---|---|---|---|---|
   | θ0 (reference) | | 73.5% | 77% | 61% |
   | none | 1 | 0% | 0% | 0% |
   | 14 (0.2%) | 3 | 14–37% | 13–31% | 0–3% |
   | 140 (2%) | 2 | 59–63.5% | 54.5–64% | 15% |
   | every frame | 2 | 71–75.5% | 69.5–74% | 52–53% |

   The budget is a share of θ0's whole training set, split equally per task (at 2%, 6,160 frames for
   44 tasks). 2% keeps about four fifths of θ0's Spatial and Goal success, 0.2% about a third.
   Replaying every frame keeps Spatial and Goal at θ0's level (within 5.5 pts), so the model can hold
   these skills while it learns; at small budgets it lacks data, not capacity. LIBERO-10 is the
   exception: even with every frame it ends 8–9 pts below θ0, and its task 9 below half of θ0's level.
4. **A tiny replay buffer costs new-task learning.** At 0.2% the five new tasks learn about 10 pts
   less on average than without replay; from 2% up the average cost is inside the noise, though single
   tasks can still lose up to 16 pts. The 14-frame buffer is memorized (its frames fit about 4.5x
   tighter than θ0 itself fits them).
5. **What 2% replay loses most is θ0's long, two-step (LIBERO-10) skills, in any scene.** They
   keep a quarter of θ0's success, both in the new tasks' kitchens and in rooms they never touch.
   No single step is to blame: the first step succeeds about half as often as under θ0, and after a
   successful first step the second also succeeds about half as often, which leaves a quarter.
   Replaying every frame brings both steps back on tasks 2 and 3, and about half of task 9.
6. **How a budget is split matters for long tasks (one seed).** An equal split gives every task the
   same number of frames, so a LIBERO-10 task, whose demos are twice as long, gets the thinnest slice
   (1.4% of its frames, against 2.6–2.8% for Spatial and Goal). Splitting the same 2% by task
   length about doubles LIBERO-10 retention (from 0.24 to 0.47) with no detectable cost on Spatial or
   Goal. In the far scenes the second steps still lag, so coverage explains part of the loss, and
   horizon or θ0's weaker hold on these tasks the rest. A second seed was stopped early, so this
   rests on one sequence.
7. **One new task is not kept by replay.** Task 23 (close the bottom drawer, open the top one), whose
   demos never close the gripper, is lost at 0.2% and 2% and only partly kept with every frame (8
   and 32% in two seeds).
8. **Small details move the numbers a lot.** θ0 scores 76% on LIBERO-Spatial when the simulator
   renders at 256x256, the size of its training images, and 64% at the default 360x360. The random
   seed matters too: three runs at the 0.2% budget, differing only in which frames went into the
   buffer and in training randomness, ended at 14, 19.5 and 37% on LIBERO-Spatial. Evaluating the
   same model twice moves its score by only a few points, so the spread comes from training, not
   from the evaluation. At that spread, even three seeds per budget detect only differences of about
   30 pts, and the 2% and every-frame budgets have two.

![Retention vs replay budget](docs/figures/fig11_m2_retention.png)

*Retention (final success divided by θ0's) after five new tasks, for each replay budget and
each suite of θ0's old skills. Dots are seeds, bars their mean.*

Caveats that apply throughout: one checkpoint and one recipe, five new tasks in one order,
LIBERO's fixed initial states (so retention of in-distribution behaviour, not general skill), and
few seeds per condition. Every number in the report traces to a row of [results/](results/);
[results/README.md](results/README.md) explains the columns, run labels and experiment ids.

**Open questions.** At 2%, random frames already sample each step of the long tasks at its share of
the data, so choosing frames within a task looks like a weak lever. What is left is how a fixed
budget is split across tasks, the one new task that replay cannot keep, and methods that need no
stored data. Fine-tuned checkpoints are not published yet.

## Reproducing

### What you need

- Linux, an NVIDIA GPU with CUDA 12.6, Python 3.13 and cmake. All results come from one A100
  (80 GB) per job; we did not test smaller GPUs. A CPU-only machine runs the tests but is far too
  slow for evaluation.
- About 2.4 GB of RAM per simulated episode: a 50-episode evaluation batch needs about 120 GB, and
  the launchers request 160 GB.
- About 30 GB of disk for models, data and the Python environment.
- Compute: [docs/report.md](docs/report.md), appendix A, gives it per experiment. The cheapest check
  of the main effect is θ0's reference and the single fine-tune in step 5.
- Slurm for the launchers (`*.sbatch`), or one machine with `scripts/run_local.sh` in place of
  `sbatch` (see "Running without Slurm").

### 1. Install

On a machine with internet access:

```bash
git clone https://github.com/elvinhajizada/adaptive-vla-replay.git && cd adaptive-vla-replay
bash scripts/build_env.sh        # .venv with the exact versions in requirements-lock.txt (CUDA 12.6);
                                 # needs python3.13 on PATH, or AVLA_PYTHON=/path/to/python3.13
# CPU-only machine instead (tests and plumbing checks; needs uv): bash scripts/build_env_cpu.sh
```

### 2. Download the models and data

```bash
AVLA_ONLINE=1 source scripts/env.sh   # allow downloads in this shell
python scripts/download_assets.py     # θ0, SmolVLM2, the nvidia dataset, LIBERO assets, at pinned revisions
```

Every later shell and job runs `source scripts/env.sh` without `AVLA_ONLINE`. That activates the
environment and switches to offline mode, so runs load only the pinned files from the local cache.
Data lives under `$AVLA_DSS` (default `~/avla-data`; set it before sourcing to move it). Rendering
uses EGL where an NVIDIA GPU is visible and OSMesa elsewhere (`AVLA_GL=egl|osmesa` overrides).

### 3. Check the setup

```bash
source scripts/env.sh
python -m pytest tests/              # unit tests, CPU
sbatch scripts/smoke_gpu.sbatch      # GPU node: LIBERO env reset and a 2-episode θ0 eval, ends with "SMOKE DONE"
```

Without Slurm, run `bash scripts/smoke.sh cuda` on the GPU machine instead.

### 4. Convert the data (once, CPU)

```bash
j=$(sbatch --parsable scripts/convert_libero90.sbatch)               # LIBERO-90 -> θ0's conventions, verified
sbatch --dependency=afterok:${j%%;*} scripts/convert_theta0_suites.sbatch   # θ0's own four suites, merged with it (replay data)
```

### 5. Reproduce the results

Each experiment is a chain of Slurm jobs: one job per training stage, then a job for the final
probes: the evaluations that measure what was kept (all LIBERO-Spatial and LIBERO-Goal tasks,
LIBERO-10 tasks 2, 3 and 9, and the earlier new tasks). A chain script submits the whole chain at
once from wherever you call `sbatch`; each job starts when the previous one has finished (without
Slurm, see "Running without Slurm" below).
Training runs are written to `$AVLA_RUNS/train/<job>/`, evaluations to
`$AVLA_RUNS/eval/<suite>/<job>-<label>-r256/` (r256: the render size).

**Replay.** One run teaches θ0 the five new tasks in order, replaying stored frames alongside each
one, and ends with the final probes:

```bash
bash experiments/replay/er_seed_chain.sh K BUFFER_SEED TRAINING_SEED
```

- `K`: frames kept per task in the replay buffer: `14` (0.2% of θ0's data), `140` (2%) or `all`.
- `BUFFER_SEED`: decides which frames are drawn into the buffer.
- `TRAINING_SEED`: everything else that is random in training, such as the batch order and the
  noise drawn for the training loss.

Runs differ for two reasons: which frames were stored, and the randomness of training itself. The
two seeds are separate so that each can be varied with the other held fixed. The report's runs
change both together (0 and 1000, 1 and 1001, 2 and 1002; seeds 1, 2 and 3 in the report), so their
spread mixes the two:

```bash
for s in "0 1000" "1 1001" "2 1002"; do bash experiments/replay/er_seed_chain.sh 14 $s; done
for s in "0 1000" "1 1001"; do bash experiments/replay/er_seed_chain.sh 140 $s; done
for s in "0 1000" "1 1001"; do bash experiments/replay/er_seed_chain.sh all $s; done
bash experiments/replay/e50_prop_chain.sh      # 2%, split by task length instead of equally
```

**No replay**: the same five tasks with plain fine-tuning, training seed 1000.

```bash
bash experiments/replay/seq_chain.sh
```

**θ0's reference**: θ0 itself on the same final probes, the baseline for every retention number.

```bash
sbatch scripts/probe_eval.sbatch \
    "$HF_HOME/hub/models--HuggingFaceVLA--smolvla_libero/snapshots/6721902bc4d61e50a3bfdb11dfb4cb626f05d102" \
    theta0-final libero_spatial:all:20 libero_goal:all:20 "libero_10:[2,3,9]:50"
```

**One fine-tune**: train θ0 on new task 23 for 1,000 steps and evaluate it on that task, then
measure what it forgot with the probes of the report's Table 2 (θ0's row there is the reference).

```bash
sbatch scripts/finetune_eval.sbatch 23 1000
JOB=...    # the job id the line above printed; start this once that job has finished
sbatch scripts/probe_eval.sbatch "$AVLA_RUNS/train/$JOB/checkpoints/last/pretrained_model" \
    ft23 "libero_10:[3]:50" libero_spatial:all:10 libero_goal:all:10
```

**Everything else**: [results/README.md](results/README.md) lists, for each section of the report,
the launcher and the run labels in `results/runs.csv`.

**Tables and figures.** The figure script draws the report's figures from the CSVs shipped in
`results/`; it selects our runs by their job ids. The collector gathers your runs into the same
tables, in a directory of your choice, to compare with ours:

```bash
python scripts/make_figures.py                       # docs/figures/ from results/*.csv
python scripts/collect_results.py --out my_results   # your runs under $AVLA_RUNS as CSV tables
```

### Running on Slurm

The launchers (`*.sbatch`) carry the resource requests the runs used (GPU, CPUs, memory, time) and
a placeholder partition. Set yours once per shell, and the chain scripts pass it on to every job
they submit (Slurm's environment variables override the launchers' `#SBATCH` lines):

```bash
export SBATCH_PARTITION=<your-partition>   # also SBATCH_ACCOUNT, SBATCH_QOS if your site needs them
```

Submit from the repository root, or set `AVLA_REPO`. Every batch job starts
`scripts/memguard.py`, which kills the job at 90% of its `--mem` request, for sites that do not
enforce memory limits.

### Running without Slurm

On a single machine with a GPU, `scripts/run_local.sh` stands in for `sbatch`: it runs a launcher
right away and returns when it has finished, so the jobs of a chain run one after another.

```bash
export AVLA_SBATCH=scripts/run_local.sh       # the chain scripts in step 5 now run here
bash scripts/run_local.sh scripts/convert_libero90.sbatch    # any single launcher: replace "sbatch"
```

Logs go to `slurm-<job>.out` in the current directory. Evaluation is the limit on one machine: the
protocol runs all episodes of a task at once, one simulator process each, so a 50-episode task
needs about 120 GB of RAM. Evaluating in smaller batches changes the protocol the reported numbers
used. The replay stages also evaluate several probes at once along the way (up to 60 processes,
about 145 GB). `export AVLA_STAGE_PROBES=0` skips those: the final probes still run, but the
measurements along the sequence (retention after each stage, and how each new task is forgotten as
later ones are learned) are then missing. No memory watchdog runs here. We have tested this path's plumbing, not a full run on a local GPU.

## Pitfalls this repo handles

- **The nvidia LIBERO-90 data needs a conversion** before training θ0 on it: another wrist-camera
  key and a flipped gripper convention (1 open / 0 close instead of -1 open / +1 close). Trained on
  it unchanged, θ0 learns to drive the gripper backwards and nothing errors.
- **`lerobot-train` replaces a pretrained policy's normalizer stats with the dataset's** when
  fine-tuning. Left alone, each stage would re-normalize θ0 and show forgetting that is not real;
  the converted datasets carry θ0's own stats.
- **Render at 256x256**, θ0's training size: the environment default (360x360) costs 12 pts on
  LIBERO-Spatial.
- **θ0 has 605M parameters**, not the paper's 450M: it keeps all 32 SmolVLM2 layers (the 450M refers
  to `smolvla_base`, which keeps 16). 97.5M of them, the action expert, are trained.
- SmolVLA loads its VLM by name without a revision; `download_assets.py` pins that name's offline
  revision, so a cached run cannot silently pick up a newer model.

Details: [docs/LIBERO.md](docs/LIBERO.md) (data formats, conventions, protocols, published numbers).

## Layout

```
adaptive_vla/              # the package (CPU-testable parts in tests/)
  config.py                #   every setting in one place: pinned revisions, the five tasks, probes,
                           #     training recipe, replay budgets, evaluation protocol
  libero_data.py           #   nvidia LIBERO-90 -> θ0's conventions (camera key, gripper sign,
                           #     normalizer stats); task lookup by instruction; passes -> steps
  replay.py                #   replay buffer of frame handles and the sampler that mixes replay
                           #     into each batch (torch-free)
  metrics.py               #   success-matrix metrics: forgetting, NBT, normalized forgetting,
                           #     gap recovered, bootstrap intervals (NumPy only)
  checkpoints.py           #   slim checkpoints (only tensors that differ from θ0, lossless),
                           #     rebuilding them, weight interpolation
  provenance.py            #   run_info.json next to every run: commit, pinned revisions, command
  lerobot_glue.py          #   loading θ0, checkpoints and datasets through the pinned LeRobot
  subgoals.py, subgoal_env.py      # per-episode sub-goal logging during evaluation, and its summary
  lang_diag.py, lang_diag_env.py   # the language check: true, blank and swapped instructions
scripts/                   # the reusable pipeline
  env.sh, build_env*.sh, download_assets.py   # environment, offline cache, pinned assets
  convert_*.py             #   data conversion and verification
  train.sh, replay_train.py, replay_spec.py   # fine-tuning, with or without replay
  eval.sh, probe_eval.sbatch                  # evaluation at the protocol settings
  run_local.sh, memguard.py                   # running without Slurm; per-job memory watchdog
  collect_results.py, make_figures.py         # results/*.csv and the report's figures
experiments/               # one launcher or analysis per experiment (which is which: results/README.md)
  setup/                   #   data and evaluation checks, θ0 baselines (report section 2)
  single_finetune/         #   one fine-tune and how it forgets (section 3)
  replay/                  #   replay sequences and their analyses (sections 4 to 7)
  coverage/                #   offline buffer coverage
tests/                     # CPU unit tests
results/                   # every evaluation and training run as CSV, with a column guide
docs/report.md             # the write-up
docs/LIBERO.md             # LIBERO background: data, conventions, protocols, published numbers
docs/figures/              # the report's figures and the rollout GIF
SOURCE                     # the private development commit this repository was exported from
```

## Credits and licences

Code in this repository: Apache-2.0 ([LICENSE](LICENSE)). It ships no datasets or checkpoints; the
environment build and `scripts/download_assets.py` fetch them from their sources at pinned revisions,
under their own terms:

- [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0), used as a library at commit
  `9a6bb61` (version 0.6.2). The training patches in `scripts/` import and wrap its functions; no
  LeRobot source is copied here.
- [`HuggingFaceVLA/smolvla_libero`](https://huggingface.co/HuggingFaceVLA/smolvla_libero) (θ0) and
  [`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct),
  both Apache-2.0.
- [`nvidia/LIBERO_LeRobot_v3`](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3),
  OpenMDW-1.1.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (MIT, Copyright (c) 2023 Lifelong Robot
  Learning), via the `hf-libero` package (MIT) and the `lerobot/libero-assets` copy of its
  simulation assets. The rollout frames in the GIF and figures are rendered from these assets.

Related work and references: [docs/report.md](docs/report.md), sections 8 and References.

## How to cite

Elvin Hajizada. *Forgetting and replay in a deployed SmolVLA.* 2026. Citation metadata:
[CITATION.cff](CITATION.cff).

```bibtex
@misc{hajizada2026forgetting,
  author = {Hajizada, Elvin},
  title  = {Forgetting and replay in a deployed {SmolVLA}},
  year   = {2026},
  url    = {https://github.com/elvinhajizada/adaptive-vla-replay}
}
```
