# Forgetting and replay in a deployed SmolVLA

How much does a deployed vision-language-action model forget when it is taught new manipulation
skills, and how much of its own training data has to be replayed to keep its old ones?

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
   old skills drop to 0 to 27% (from 74 to 84%); on the same-scene task, the only one tested during
   training, the skill is gone by step 100, before the new task is learned. A 10x lower learning rate
   does not prevent it.
2. **Without replay, a sequence keeps only the last task.** Every earlier task, and every old θ0
   skill, is at 0% after five stages.
3. **Replaying θ0's own data protects it, and the budget sets how much.** After five new tasks:

   | Replay budget (frames per task, share of the data) | Seeds | LIBERO-Spatial | LIBERO-Goal | LIBERO-10, same kitchens |
   |---|---|---|---|---|
   | θ0 (reference) | | 73.5% | 77% | 61% |
   | none | 1 | 0% | 0% | 0% |
   | 14 (0.2%) | 3 | 14 to 37% | 13 to 31% | 0 to 3% |
   | 140 (2%; 1.4 to 2.8% of a task by suite) | 2 | 59 to 63.5% | 54.5 to 64% | 15% |
   | every frame | 2 | 71 to 75.5% | 69.5 to 74% | 52 to 53% |

   The budget is a share of θ0's whole training set, split equally per task: 140 frames per task is 2%
   of the data (6,160 frames for 44 tasks), but 2.6 to 2.8% of a Spatial or Goal task and 1.4% of a
   LIBERO-10 task, whose demos are twice as long. It keeps four fifths of θ0's Spatial and Goal
   success; a tenth of it keeps a third. Replaying everything keeps θ0's level (within 5.5 pts), so the
   model has the capacity to keep θ0's skills; their loss at small budgets is a data question.
4. **A tiny replay buffer costs new-task learning.** At 0.2% the five new tasks learn about 10 pts
   less on average than without replay; from 2% up the average cost is inside the noise, though single
   tasks can still lose up to 16 pts. The 14-frame buffer is memorized (its frames fit about 4.5x
   tighter than θ0 itself fits them).
5. **What 2% replay loses most is θ0's long, two-step (LIBERO-10) skills, in any scene.** They
   keep a quarter of θ0's success, both in the new tasks' kitchens and in rooms they never touch.
   The policy finishes the second step less often than it starts the first, but the second needs the
   first: per episode, at 2% the second step is kept about as well as the first once the first is
   done, so the loss is spread over both steps and compounds. Replaying every frame brings both back. These tasks also got the thinnest slice of the equal-split
   buffer (1.4%). Splitting the same 2% by task length about doubles their retention in one seed
   (0.24 to 0.47) at no detectable cost to the short tasks; in the far scenes the second steps still lag, so coverage
   explains part of the loss and horizon or θ0's weaker hold the rest. One new task, whose demos never close the gripper, is lost at 0.2% and 2% and only partly kept with every
   frame (8 and 32% in two seeds).
6. **Measurement matters as much as method.** Rendering at the env's default size instead of θ0's
   training size costs 12 pts. At 0.2%, seeds (buffer draw and training together), not evaluation noise, dominate the variance (seed SD
   10 to 12 pts against about ±5 pts of episode noise), so with 3 seeds per arm only differences of
   about 30 pts are detectable. Offline action drift ranks conditions but does not predict which run
   succeeds.

![Retention vs replay budget](docs/figures/fig11_m2_retention.png)

![θ0 succeeds; after learning a new task in the same kitchen, the same skill fails; the new skill works](docs/figures/m1_forgetting.gif)

Caveats that apply throughout: one checkpoint and one recipe, five new tasks in one order,
LIBERO's fixed initial states (so retention of in-distribution behaviour, not general skill), and
few seeds per condition. Every number in the report traces to a row of [results/](results/);
[results/README.md](results/README.md) explains the columns, run labels and experiment ids.

**Open questions.** Within-task frame selection looks like a weak lever at 2% (random frames already
cover every gripper phase), so what is left is how a fixed budget is split across tasks, the one new
task that replay cannot keep, and methods that need no stored data. Fine-tuned checkpoints are not
published yet.

## Reproducing

### What you need

- Linux with an NVIDIA GPU of the A100 80 GB class and CUDA 12.6 drivers; Python 3.13 and cmake.
  Every reported number comes from such a GPU. A CPU-only machine runs the tests and plumbing
  checks, but θ0 replans every control step, so a CPU episode takes about half an hour.
- Memory for evaluation: about 2.4 GB of RAM per simulator process, one process per episode in a
  batch, so a 50-episode batch needs about 120 GB. The evaluation launchers request 160 GB.
- Disk: about 13 GB of Hub assets, 4 GB of converted data, 8 GB for the environment, and about
  0.2 GB per slimmed checkpoint (a full one takes a few GB until it is slimmed).
- A Slurm cluster for the launchers (`*.sbatch`); the `scripts/*.sh` entry points also run directly.

### Steps

```bash
git clone https://github.com/elvinhajizada/adaptive-vla-replay.git && cd adaptive-vla-replay
bash scripts/build_env.sh                # .venv with the exact versions in requirements-lock.txt
AVLA_ONLINE=1 source scripts/env.sh
python scripts/download_assets.py        # θ0, SmolVLM2, the nvidia dataset, LIBERO assets, at pinned revisions
source scripts/env.sh && python -m pytest tests/   # offline from here on
bash scripts/smoke.sh cuda               # device, LIBERO env reset, a 2-episode θ0 eval
```

`scripts/env.sh` keeps all data under `$AVLA_DSS` (default `~/avla-data`; set it before sourcing to
move it), renders with EGL where an NVIDIA GPU is visible and with OSMesa elsewhere (`AVLA_GL`
overrides), and sets offline mode unless `AVLA_ONLINE=1`. On a cluster, run the build and the
download on a node with internet; compute nodes are assumed offline.

Then convert the data once and run the experiments:

```bash
sbatch scripts/convert_libero90.sbatch         # nvidia LIBERO-90 -> θ0's conventions, verified
sbatch scripts/convert_theta0_suites.sbatch    # θ0's four suites, merged with LIBERO-90 (replay data)
bash scripts/eval.sh libero_spatial all 20     # θ0 on a suite at the protocol settings
sbatch scripts/finetune_eval.sbatch 23 8p      # one new task, 8 passes, evaluated and slimmed
bash experiments/m2/er_seed_chain.sh 140 1 1001   # a five-stage replay sequence (k, buffer seed, training seed) and its final probes
python scripts/make_figures.py                 # the report's figures, from results/*.csv only
```

**Slurm headers are a template.** The launchers carry the resource requests the runs used (GPU,
CPUs, memory, time) and a placeholder partition: set `--partition` (and an account or QOS if your
site needs them) before submitting. Submit from the repository root, or set `AVLA_REPO`. Every
batch job starts `scripts/memguard.py`, which kills the job at 90% of its `--mem` request, for sites
that do not enforce memory limits. For each report section, [results/README.md](results/README.md)
names the launcher and the run labels; `scripts/collect_results.py` rebuilds the tables from your
own run directories.

### Pitfalls this repo handles

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
adaptive_vla/        # the package: config (pinned revisions, tasks, probes, recipe), metrics (NBT,
                     #   forgetting, bootstrap CIs), replay buffer, LIBERO data conversion, slim
                     #   checkpoints, provenance, sub-goal and language-diagnostic env wrappers
scripts/             # the reusable pipeline: environment, assets, conversion, train, eval, probes,
                     #   replay training, offline action error, results collection, figures
experiments/         # one launcher or analysis per experiment, named by its id (results/README.md):
  m1/                #   setup checks and single-task fine-tunes
  m2/                #   replay sequences and their analyses
  m3/                #   offline buffer coverage
tests/               # CPU unit tests
results/             # every evaluation and training run as CSV, with a column guide
docs/report.md       # the write-up
docs/LIBERO.md       # LIBERO background: data, conventions, protocols, published numbers
docs/figures/        # the report's figures and the rollout GIF
SOURCE               # the private development commit this repository was exported from
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
