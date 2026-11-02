# Forgetting and replay in a deployed SmolVLA

_Version 1.0, 2026-11-02._

## Abstract

A deployed robot policy will be fine-tuned on new tasks, usually by whoever holds its training
data. We teach the released SmolVLA checkpoint for LIBERO five new LIBERO-90 tasks in sequence, with
its own fine-tuning recipe, and measure how much of its own training data has to be replayed to keep
its skills. Without replay, one fine-tune takes its success on its old suites from about 75% to
between 0 and 27%, and a five-task sequence keeps only the last task. Replaying a fixed random sample
of its training data alongside each new task keeps the skills, more of them the larger the sample:
after five tasks, 2% of the data keeps about four fifths of its success on suites in other scenes,
0.2% about a third, and every frame all of it. Its long LIBERO-10 tasks keep least, a quarter at 2%
whether or not they share a scene with the new tasks; near the new tasks the loss is spread over
both sub-goals of each task. Splitting the same 2% by task length instead of equally about doubles their
retention in one seed, at no detectable cost to the short tasks. At 0.2% the buffer is
memorized, the new tasks learn about 10 pts less, and the seed moves results by up to 23 pts, so
three seeds detect only differences of about 30 pts. All numbers come from closed-loop simulation on
LIBERO's fixed initial layouts, with one model and one task order.

## 1. Introduction

A robot policy that is already deployed will be asked to learn new tasks, and whoever deploys it
usually holds the data it was trained on. We take a released vision-language-action (VLA) model,
the SmolVLA checkpoint for the LIBERO benchmark, which we call θ0 [1], teach it new LIBERO-90
tasks one after another with its own fine-tuning recipe, and ask two questions: how many of its own
skills does it lose, and how much of its own training data has to be replayed to keep them?

The first answer is that it loses nearly all of them, and fast. After one fine-tune on a single new
task, θ0's success on LIBERO-Spatial and LIBERO-Goal falls from about 75% to between 0 and 27%, and
on a LIBERO-10 task in the new task's scene from 84% to 0. On that one old task, the only one we
tracked during training, the skill was gone after 100 steps, before the new task was learned, and a
10x lower learning rate did not prevent it (section 3).

Prior work suggests where to look for a fix. Pretrained VLAs such as π0 and GR00T N1.5 forget little
in LIBERO task sequences when trained with experience replay, and much without it [2]; their
buffers hold the earlier tasks of the sequence. Fine-tuning π0 on a new LIBERO task erodes its
pretraining tasks, and mixing pretraining data back into the fine-tune reduces the loss [3].
A deployed model needs the second setting carried through a sequence of tasks: how much of its own
data to replay, and what a small budget fails to keep. We therefore teach θ0 five new tasks in a
row, one fine-tuning stage per task, and replay a fixed random sample of θ0's own training data
alongside each: 0.2% or 2% of that data, or every frame of it. The question becomes one of data: how
much of its old data a deployed policy has to keep.

Replay of θ0's own data keeps its skills, and the budget sets how much. We report retention: the
final success on a probe after the new tasks, divided by θ0's own success on it. After five new tasks, 2%
keeps about four fifths of θ0's success on two LIBERO suites set in other scenes, 0.2% about a
third, and every frame all of it (section 4). What 2% keeps least are θ0's long LIBERO-10 tasks, in
any scene, and one new task whose demos never close the gripper (section 5). Part of the LIBERO-10
loss is how the budget is split: splitting the same 2% by task length, so that long tasks get more
frames, about doubles their retention in one seed. At 0.2% the new tasks also learn about 10 pts
less on average (section 6), and the seed moves results by up to 23 pts, so with three seeds per
condition only differences of about 30 pts are detectable (section 7).

The study is small and controlled: one checkpoint, one recipe, five new tasks in one fixed order,
LIBERO's fixed initial layouts, and one to three seeds per condition. Retention here means that θ0
still reproduces trained behaviour from known layouts, which is most of what LIBERO measures
[4]. Every number comes from closed-loop simulation, and the code and per-run results are
released with this report; raw evaluation outputs and checkpoints are not.

## 2. Setup

Figure 1 shows the protocol; the subsections below give its parts.

![The five-task sequence with replay](figures/fig16_m2_protocol.png)

*Figure 1. The sequence. Each stage trains one new task for 8 passes over its demos, with half of each
batch drawn from a buffer that holds k frames of each of θ0's 40 tasks and of every new task already
learned. Probes run after every stage and, more widely, after the last. Without replay there is no
buffer and the probes are the same. "10 tasks x 20" means 20 episodes of each of 10 tasks.*

### 2.1 θ0 and its training data

θ0 is the released SmolVLA checkpoint for LIBERO, `HuggingFaceVLA/smolvla_libero` [5].
SmolVLA pairs a SmolVLM2 vision-language backbone with a separate action expert, a transformer
trained by flow matching that outputs chunks of 50 actions [1]. The SmolVLA paper keeps the
first 16 layers of the language model and reports about 450M parameters; θ0 keeps all 32 and has
605M. θ0 executes one action of each chunk and then replans. Fine-tuning follows θ0's own recipe,
which trains the action expert and the state projection (97.5M parameters) and leaves the vision
encoder and the language model frozen.

LIBERO is a simulated manipulation benchmark with a Franka arm [6]. Each task is defined by a
scene, a language instruction and a set of goal predicates (for example "the bowl is in the bottom
drawer" and "the bottom drawer is closed"); an episode succeeds when all predicates hold at once.
Training data are human teleoperated demos, and evaluation starts from 50 fixed initial object
layouts per task. θ0's model card names no training data. Its normalizer statistics are identical
to those of `lerobot/libero` [7], which holds the four suites LIBERO-Spatial,
LIBERO-Object, LIBERO-Goal and LIBERO-10 (40 tasks, 1,693 demos), so we take that set to be θ0's
training data. We call these 40 tasks θ0's skills.

New-task demos come from a LeRobot copy of LIBERO that also contains LIBERO-90
[8], and its conventions differ from θ0's in two ways that would each show up as
forgetting. Its gripper sign is flipped (1 for open where θ0 uses -1); trained on it unchanged, θ0
learns to drive the gripper backwards and nothing raises an error. And when fine-tuning starts from a
pretrained checkpoint, the pinned `lerobot-train` [9] replaces the policy's normalizer
statistics with the dataset's, so every training stage would re-normalize θ0's inputs and outputs.
We convert the data to θ0's conventions and write θ0's own statistics into the converted dataset,
which turns the replacement into a no-op; every checkpoint in this report carries θ0's statistics
unchanged. θ0's four suites, converted from the same copy, supply the replay buffer; their episodes,
states and arm actions match `lerobot/libero` exactly.

### 2.2 Five new tasks

The new tasks had to be ones θ0 cannot already do. We screened the 11 LIBERO-90 kitchen tasks that
have demos from a single scene, and θ0 solved 2 of 110 episodes. Five candidates are sub-steps of
θ0's own LIBERO-10 tasks in the same scene, and θ0 fails those too: it completes "put the black bowl
in the bottom drawer of the cabinet and close it" in 42 of 50 episodes, but "close the bottom
drawer" in none of 10. We left out such sub-steps, since a probe change after training on one would
be ambiguous, and one scene whose LIBERO-10 task θ0 solves only 16% of the time, and chose the five
tasks of Table 1 from the rest, all at 0 of 10 zero-shot. Tasks 23 and 26 still use the bottom drawer of LIBERO-10
task 3, a confound for that probe. The screen also suggests what θ0 is: it reproduces trained
combinations of instruction and scene rather than reusable sub-skills. This matches reports that
LIBERO success measures reproduction of trained layouts more than general skill [4], so retention in
this report means retention of in-distribution behaviour from fixed initial states.

*Table 1. The five new tasks, in training order. Sub-goals: number of goal predicates. Single-task
fine-tune: 8 passes from θ0 (task 23: 1,000 steps, 7.9 passes), 50 episodes.*

| Order | Task (LIBERO-90 id) | Kitchen scene | Sub-goals | Demos | θ0 zero-shot | Single-task fine-tune |
|---|---|---|---|---|---|---|
| 1 | put the wine bottle in the bottom drawer of the cabinet (26) | 4 | 1 | 43 | 0/10 | 88% |
| 2 | put the frying pan on the stove (18) | 3 | 1 | 47 | 0/10 | 76% |
| 3 | close the bottom drawer of the cabinet and open the top drawer (23) | 4 | 2 | 34 | 0/10 | 54% |
| 4 | put the wine bottle on the wine rack (27) | 4 | 1 | 39 | 0/10 | 72% |
| 5 | put the yellow and white mug to the front of the white mug (34) | 6 | 1 | 44 | 0/10 | 88% |

Task 23 stands apart: its demos never close the gripper, while θ0's own demos grasp in all 40 of its
tasks. The order is the same for every condition. It opens with task 26, which we had already used
for a single-stage pilot, and keeps task 23 and task 27 (whose instruction nearly repeats a
LIBERO-Goal one) off the last stage. With one order, order effects are not measured.

### 2.3 Probes

Old skills are re-tested on three probes. LIBERO-Spatial and LIBERO-Goal (10 tasks each) are set in
scenes that no new task uses; Goal shares objects with the new tasks (bowls, plates, a wine bottle
and rack), and its tasks share one scene and differ only in instruction. The third probe is the
three LIBERO-10 tasks set in the new tasks' kitchens (ids 2, 3 and 9). Each has two sub-goals, goal
predicates that the demos reach one after the other (for task 3, "the bowl is in the drawer", then
"the drawer is closed"); the second can only hold once the first has been done. After the last
stage, the final probes run every Spatial and Goal task at 20 episodes and the LIBERO-10 tasks at 50.
After every stage, smaller per-stage subsets (5 Spatial and 6 Goal tasks at 10 episodes) track the
sequence; they were fixed by rule before any replay result existed. A control added later probes
five more LIBERO-10 tasks with two sub-goals in living-room scenes (section 5). LIBERO-Object, θ0's
fourth suite, is not probed, so the probes cover 28 of θ0's 40 tasks.

A policy that ignored language would fail on Goal, where only the instruction tells tasks apart.
θ0 does not: on three Goal tasks it succeeds in 25 of 30 episodes with the true instruction, in none
with a blank one, and with another task's instruction it reaches that other task's goal in 26 of
30. On Goal, then, θ0 selects among trained tasks by instruction, so interference between tasks
that share a scene cannot be put down to a policy that ignores language. The test covers three
tasks of one suite.

### 2.4 Training and replay

Every new task is trained for a fixed 8 passes over its demos with θ0's recipe (AdamW, peak learning
rate 1e-4 with warmup and cosine decay, batch 64). Each stage starts from the previous stage's
weights with a fresh optimizer and learning-rate schedule. We fixed passes rather than steps
because longer training lowered the loss without improving the policy: on task 23, tripling the steps cut the final
loss from 0.32 to 0.14 while success stayed between 48 and 64% of 50 episodes, with no detectable
difference between 1k, 2k and 3k steps.

Replay keeps a buffer of k random frames from each of θ0's 40 tasks and from every new task already
learned, drawn once and kept for the whole sequence. Each batch holds 32 frames of the current task
and 32 from the buffer, sampled uniformly over tasks and then over a task's frames, as in the 1:1
mixing of [2]. We run k = 14, k = 140 and every frame, and name a budget by the share of θ0's
data it holds: θ0's 40 tasks have 273,465 frames, so 140 frames per task (5,600) is 2% and 14 per task
is 0.2%. Every task gets the same k, so long tasks are covered more thinly: at 2% a Spatial or Goal
task keeps about one demo's worth of frames and a LIBERO-10 task, whose demos are twice as long,
about half. At the last stage the buffer holds 616 frames at 0.2%, 6,160 at 2% and 305,311 with every
frame, counting the earlier new tasks. A stage runs 8 passes over the new task's frames whether or not
it replays, and with half of each batch given to replay that takes twice the optimizer steps (a
confound we return to in section 6). One seed value sets both the buffer draw and the training
randomness; runs at different budgets share seed values but not buffers, so they are not paired.

### 2.5 Evaluation

Every number comes from closed-loop episodes in LeRobot's LIBERO evaluation, with LIBERO's fixed
initial states, a full reset before every episode and one evaluation seed, rendered at 256x256. The
render size is not cosmetic. θ0 trained on 256x256 frames [7], while the simulator renders 360x360 by
default, and at the default θ0 solved 64 of 100 Spatial episodes against 76 at 256 (a paired test
over the same episodes gives p = 0.029), although open-loop checks had found no mismatch at either
size. Repeating an evaluation does not reproduce it exactly (two runs of θ0 agreed on 122 of 130
episodes), so runs share initial states but not rollouts.

θ0 scores below the SmolVLA paper's LIBERO results: on our final probes 73.5% on Spatial and 77% on
Goal, against 90% and 92% for the paper's 0.45B model, which was trained on a copy of the same 1,693
demos [1]. θ0 has none of the paper's three model sizes and its card reports no LIBERO results, so
no published number exists for this checkpoint. We have not found the cause of the gap. Every
retention divides by θ0 as evaluated here, on the same protocol as the run, so the gap moves the
reference, not the comparison; a θ0 handicapped by our pipeline could still forget differently.

### 2.6 Metrics and uncertainty

The headline metric is retention, a run's final success on a probe divided by θ0's success on the
same protocol:

$$R = \frac{s_\text{run}}{s_{\theta_0}}.$$

On the final probes θ0 succeeds in 73.5% of Spatial episodes, 77% of Goal and 61% of the three
LIBERO-10 tasks, pooling every θ0 run on that protocol (one for Spatial, two for Goal and for each
LIBERO-10 task; appendix B). The per-stage subsets have their own θ0 reference.

Retention measures θ0's skills. Forgetting of the new tasks themselves uses LIBERO's negative
backward transfer (NBT) [6], computed from the success matrix $S$, where $S_{ij}$ is the success on
new task $j$ after stage $i$ of $T = 5$:

$$\mathrm{NBT} = \frac{1}{T-1}\sum_{j=1}^{T-1} \frac{1}{T-j}\sum_{i=j+1}^{T}\left(S_{jj} - S_{ij}\right).$$

It is the average drop of each new task, except the last, from its success right after it was
learned to its success in the later stages. LIBERO takes $S_{jj}$ as the best success seen while
training task $j$; we evaluate only at the end of each stage and use that value, so our NBT is not
comparable as a number with published ones.

Tables give the seed mean, with the standard deviation across seeds where there are three; with two
seeds we give the mean, and both seeds are in appendix B. Episode noise on a 200-episode probe is
about ±5 pts (bootstrap); intervals for every run are in appendix B and in the text only where a
comparison rests on them. These intervals cover episode noise only; at 0.2% the seeds vary far more
(section 7). We call differences inside 6 to 10 pts no detectable difference. Each experiment's text
says whether it was planned or added after a result; from the first replay experiment on, a
prediction was written down before every experiment except two post-hoc analyses, marked where they
appear (appendix C lists the predictions).

## 3. One fine-tune erases θ0's skills

After one fine-tune on a new task, θ0 fails on its old tasks, in scenes the new task never visits as
well as in its own (Table 2). After the task-23 fine-tune, no Spatial or Goal episode succeeds. A
fine-tune on task 26, which grasps like θ0's own tasks, does barely better.
Task 27's fine-tune keeps a third of θ0's Goal success, most of it on three Goal tasks that stay near
θ0's level. The checkpoints are not broken: each solves its own new task, keeps θ0's normalizer
statistics, and turns all of LIBERO's instructions into exactly the tokens θ0 does. We found the
collapse while checking how training length trades against forgetting; the runs after it were added
to understand it.

*Table 2. θ0's skills after one fine-tune on one new task (one run each), success (%). Same-scene
task: LIBERO-10 task 3, in kitchen scene 4 with tasks 23, 26 and 27, 50 episodes. Spatial and Goal:
10 tasks x 10 episodes from the same initial states as θ0's row. "Not run": these two runs were
evaluated only for the question each was trained for: whether a 10x lower learning rate spares the
same-scene task and Spatial, and whether learning task 27 hurts Goal's near-identical "put the wine
bottle on the rack".*

| Fine-tuned on | Same-scene task | Spatial | Goal |
|---|---|---|---|
| (θ0, no fine-tune) | 84 | 76 | 74 |
| task 23 | 0 | 0 | 0 |
| task 23, learning rate 1e-5 | 0 | 3 | not run |
| task 26 | 0 | 12 | 2 |
| task 27 | not run | not run | 27 |

On the same-scene task, the skill is gone 100 steps into the fine-tune, before the new task is
learned (Figure 2). With a checkpoint every 100 steps of the task-23 run, θ0's same-scene skill fails
in every episode from step 100 on, while task 23 first succeeds at step 400 and reaches 75% at step
700. Step 100 is less than one pass over the new demos. This was measured on one task, in one run, at
20 episodes per checkpoint; Spatial and Goal were not evaluated in closed loop during training.
Offline, the change in θ0's predicted arm motion (translation and rotation) on Spatial demo frames is
already at or above its final size at step 100; the gripper change keeps growing until about step
700.

![Forgetting comes before learning](figures/fig5_onset.png)

*Figure 2. Left: success on the same-scene LIBERO-10 task (old skill) and on new task 23 during task
23's fine-tune; checkpoints of one run over 20 episodes, θ0 and the 1k-step end point (a separate
run with the same recipe) over 50. Right: change of the predicted actions from θ0's on fixed Spatial
demo frames, in units of θ0's action standard deviation; diamonds, right of the divider: the end of
a separate run at a 10x lower learning rate. The old skill is gone before the new one appears.*

A 10x lower learning rate learns less of the new task and forgets as much. At a peak of 1e-5 the
policy learns only the first sub-goal of task 23 (it closes the bottom drawer but never opens the
top one), still fails the same-scene task in every episode, and solves 3 of 100 Spatial
episodes. Its arm change on Spatial frames is nearly as large as at 1e-4; only the gripper changes less. One run.

Without replay, a five-task sequence keeps only the task learned last. Each new task is learned at
roughly its single-task level (tasks 27 and 34 within 2 pts, task 18 12 pts higher and task 23 18
pts lower; task 26's single-task run is the sequence's first stage), and each is lost entirely one
stage later: every earlier task is at 0 from the next stage on, and at the end every probe of θ0's
skills is at 0. NBT is 0.72, the mean just-learned success of the first four tasks, since all of it
is lost. The sequence ran with one seed; at the floor, more seeds would re-measure zero. The
success matrices of all conditions are in appendix B.

## 4. Replay of θ0's own data: the budget sets retention

After five new tasks, replaying 2% of θ0's data keeps about four fifths of its Spatial and Goal
success (retention 0.83 and 0.77), 0.2% keeps about a third (0.32 and 0.31), and every frame keeps
all of it (1.00 and 0.93; Table 3, Figure 3). The budgets are well separated: every 0.2% seed ends
more than 20 pts below both 2% seeds on both probes, four times the ±5 pt episode noise. With three
budgets we can rank them but cannot say where between 0.2% and 2% retention rises fastest. In the
five-task sequence the 0.2% budget was planned; 2% and every frame were added after its result.

*Table 3. Success (%) on the final probes after five new tasks: seed mean, ± SD across seeds where
there are three. Retention divides by θ0 on the same protocol. No replay was probed on a smaller
protocol and sits at 0. The split by task length (section 5) has one seed and no per-stage
probes, so no NBT. Per-seed values and intervals: appendix B.*

| Replay budget | Seeds | Spatial | Goal | LIBERO-10 (tasks 2, 3, 9) | Retention: Spatial / Goal / LIBERO-10 | NBT, new tasks |
|---|---|---|---|---|---|---|
| θ0 | | 73.5 | 77 | 61 | 1 | |
| none | 1 | 0 | 0 | 0 | 0 / 0 / 0 | 0.72 |
| 0.2% (k = 14) | 3 | 23.5 ± 12 | 24 ± 10 | 1 ± 2 | 0.32 / 0.31 / 0.02 | 0.45 ± 0.06 |
| 2% (k = 140) | 2 | 61 | 59 | 15 | 0.83 / 0.77 / 0.24 | 0.27 |
| every frame | 2 | 73 | 72 | 52 | 1.00 / 0.93 / 0.85 | 0.19 |
| 2%, split by task length | 1 | 57 | 56.5 | 29 | 0.78 / 0.73 / 0.47 | not measured |

![Retention after five new tasks, by replay budget](figures/fig11_m2_retention.png)

*Figure 3. Retention after five new tasks, by replay budget and probe. Dots are seeds, bars the
seed mean; 1.00 is θ0 on the same protocol. Retention rises with the budget on every probe, and
LIBERO-10 lags the other two at 2%.*

Replaying every frame keeps θ0's level: Spatial ends where θ0 is, Goal about 5 pts below, inside the
noise floor. The budgets differ only in how many distinct frames the buffer holds; the steps and the
replay frames per batch are the same. So the expert has the capacity to keep θ0's tasks while it
learns five new ones, and smaller budgets fall short for another reason: too few distinct frames,
or the same few frames repeated more often, which these runs do not separate. Two exceptions remain
even with every frame (section 5): LIBERO-10 task 9 ends below half of θ0's level, and new task 23 is
kept in only one of two seeds.

θ0's LIBERO-10 tasks keep much less than Spatial and Goal at small budgets: a quarter of θ0's success
at 2%, against 0.85 with every frame. Both 2% seeds end at the same 15%, so the loss is not seed
noise. Section 5 looks at what exactly is lost.

![Per-stage subsets over the sequence](figures/fig12_m2_stages.png)

*Figure 4. Retention of θ0's skills on the per-stage subsets after each stage, by replay budget,
against θ0 on the same subsets (dashed line); thin lines are seeds, markers seed means. No replay was
probed after stages 1 and 5 only. At 0.2% both probes fall over the sequence; with every frame
neither does.*

At 0.2% retention keeps falling over the five stages; with every frame it does not (Figure 4). On
the per-stage subsets, which use the same protocol after every stage, 0.2% keeps about 0.6 of θ0
after the first new task and about 0.35 after the fifth, on both probes. At 2%, Goal falls from
about 0.95 to 0.8 over the sequence and Spatial holds. Two things grow together over the stages: the
number of new tasks learned, and how often the same buffered frames have been replayed (at 0.2%
about 100 more times each per stage), so the decline may come from either. Either way, a test after a single
new task overstates what a small budget keeps.

Replay also keeps part of the earlier new tasks, and more with a larger budget. Without replay each
new task is lost one stage after it is learned (NBT 0.72); with replay NBT falls to 0.45 at 0.2%,
0.27 at 2% and 0.19 with every frame. At 0.2% the average success over the five new tasks at the end
(14 to 28% across seeds) shows no detectable difference from no replay (17%): the earlier tasks keep
more, but each new task is learned less (section 6). From 2% up the final average is 55 to 60%.

## 5. What replay at small budgets does not keep

Two losses survive a 2% budget: θ0's LIBERO-10 tasks, which have two sub-goals and demos twice as
long, and new task 23. The first does not depend on the scene, is partly a matter of how the budget
is split, and returns with every frame; the second is only partly kept even then.

At small budgets the policy reaches the first sub-goal of θ0's kitchen LIBERO-10 tasks more often
than the second (Figure 5). The second can only hold once the first has been done, though, so its
share falls whenever the first's does, and the per-episode logs separate the two. At 2%, once the
first sub-goal is reached, the second is kept about as well as the first: retention of the first
sub-goal is 0.47, of the second given the first 0.53 (tasks 2, 3 and 9 and both seeds pooled; 95%
interval of the difference −0.21 to 0.08). The 2% loss on these tasks is spread over both sub-goals
and compounds. At 0.2% the second sub-goal is lost beyond that: 0.24 for the first, 0.08 for the
second given the first (difference 0.08 to 0.23). With every frame both return to θ0's level. We
planned the anatomy, with a prediction, before the second 2% seed and the every-frame runs; the
per-episode split was added after a review pointed out the dependence, without a prediction.

![Failure anatomy of the LIBERO-10 tasks and task 23](figures/fig14_m2_anatomy.png)

*Figure 5. Share of episodes in which each sub-goal holds at the end, by replay budget (final
checkpoint, 50 episodes per task, seed means); reference: θ0 for the LIBERO-10 tasks, the evaluation
right after learning for task 23. The second sub-goal needs the first, so it falls whenever the first
does; at 2% the second is lost no more than that implies (text). With every frame both return for
LIBERO-10 tasks 2 and 3, half of task 9 does, and task 23's second sub-goal does not.*

The loss does not depend on the scene: five LIBERO-10 tasks with two sub-goals in living-room scenes
that no new task uses lose as much as the kitchen ones (Table 4). θ0 is weak on these tasks (35%), so
only pooled numbers carry the comparison. Their sub-goals can be reached in either order; taken in
the order the instruction names them, at 2% the first keeps 0.67 of θ0's rate and the second, once
the first is reached, 0.40 (difference 0.03 to 0.51), a loss beyond compounding at the edge of what
two seeds show. We added this control after the anatomy result, with its reading rule fixed before
the runs: retention at 2% below 0.40 would mean that the loss follows the suite, not the scene. It
came out at 0.27.

*Table 4. Retention of θ0's LIBERO-10 tasks near and far from the new tasks; θ0 on the same
protocol (kitchen tasks 50 episodes each, living-room tasks 20). 2% and every frame: two seeds
pooled; the split by task length: one seed.*

| LIBERO-10 tasks | Scenes | θ0 | 2% | 2%, split by task length | Every frame |
|---|---|---|---|---|---|
| 2, 3, 9 | the new tasks' kitchens | 61% | 0.24 | 0.47 | 0.85 |
| 0, 1, 4, 6, 7 | living rooms no new task uses | 35% | 0.27 | 0.57 | 0.84 |

Random frames do not under-sample the second sub-goal either. Phase-aware replay work argues that
uniform sampling starves short, critical phases of a task [10]. In both 2% buffers, frames from the
second sub-goal of the three kitchen tasks appear at their share of the demos, between a quarter and
a half depending on the task. This check was added after the anatomy result, with its prediction
written first.

Three explanations remain for what 2% loses on LIBERO-10, and all three change together with the
suite. LIBERO-10 demos are about twice as long, so small errors have more steps to compound. θ0
holds these tasks less firmly (35 to 61% against 74 to 77% on Spatial and Goal); among the 20
Spatial and Goal tasks θ0's per-task success does not predict retention, but that test covers only
tasks θ0 solves 40 to 100% of the time. And the equal split gives each LIBERO-10 task about half a
demo's worth of frames against one for Spatial and Goal.

The third explanation accounts for part of the loss. Splitting the same 2% in proportion to task
length, so that every task keeps 2.05% of its frames (156 to 260 per LIBERO-10 task instead of 140),
about doubles LIBERO-10 retention in one seed, from 0.24 to 0.47 in the kitchens and from 0.27 to
0.57 in the living rooms, with no detectable cost on Spatial or Goal (Tables 3 and 4). That closes
about a third of the gap to every-frame replay in the kitchens and about half in the living rooms.
By the reading rule fixed before the run, coverage is part of the loss but not most of it. Near the
new tasks the gain follows the added frames task by task; in the living rooms it does not, so we
claim no dose response within the suite. We added this run after the far-scene control; a
second seed was started and stopped after two of five stages, so the result rests on one seed. The
rest of the gap needs other experiments: rollouts restarted from the demo state at the first sub-goal would
isolate compounding error, and short tasks on which θ0 is weak would isolate its hold.

One new task, whose demos never close the gripper, is lost at 0.2% and 2% and only partly kept with
every frame. With replay, task 23 is learned about as well as without (46 to 62% across runs). It
falls when task 27 is learned next in the same scene. At the small budgets it ends near zero (at
most 16% in any of five runs), and with every frame at 8 and 32% in the two runs. What survives is
its first sub-goal: in most runs the policy still closes the bottom drawer and rarely opens the top
one. Three explanations are open. Task 23 gets 1/44 of the replay samples at every budget, like every
other task; it is the weakest-learned task in most runs, so there is less to keep; and every other
task in its buffer grasps, while task 23 never closes the gripper. Raising task 23's share of replay
at a fixed budget would test the first; we have not run it.

Even with every frame, LIBERO-10 task 9 ends below half of θ0's success. Whether the last stage,
which trains a new task in the same scene with the same mug, causes this was inconclusive by a rule
fixed in advance (appendix D).

## 6. What replay costs the new tasks

At 0.2%, three of the five new tasks learn 15 to 32 pts less than without replay, and the five lose
about 10 pts on average; from 2% up the average cost is inside the noise (Figure 6). The no-replay
reference averages two runs per task, the no-replay sequence and θ0's single-task fine-tunes (Table
1); we added the second after the fact, without a prediction. Against episode noise the 0.2% cost is
real (95% interval about 5 to 15 pts), but both references share one training seed, so seed-to-seed
variation in the reference is not in that interval. Replay also doubles the optimizer steps of a
stage. Task 23 learns 7 to 10 pts more with replay than without, and the longer training may account
for that: in the step-count runs of section 2.4 it reached 64% at 2k steps against 54% at 1k, a
difference we could not detect.

At 2%, the cost on single tasks depends on the seed: task 27 is learned at 72% in one seed and 42% in
the other, against 73% without replay. With every frame, no task's cost is detectable.

![Learning cost and memorization](figures/fig13_m2_learning.png)

*Figure 6. Left: success on the task just learned (50 episodes) by replay budget; small dots are
seeds, large markers seed means, open circles the single-task fine-tunes from θ0 (task 26 has none:
its single-task run is the first stage of the no-replay sequence). Right: offline error of the final
policy on θ0 frames in its buffer and on other frames of the same tasks, relative to θ0's own error
(dots: the three 0.2% seeds; 2% was measured on one seed). At 0.2% three tasks learn less, and the
buffered frames are fit to about a fifth of θ0's error while the others are not fit better at all.*

The 0.2% buffer is memorized. On its 14 frames per task, the final policy's error against the demo
actions is about a fifth of θ0's own. On other frames of the same tasks its error equals θ0's, though
its predictions there have moved away from θ0's (0.25 to 0.28 std in translation): they changed, but
not toward the demos. On the current task its error equals the 2% run's. Each buffered frame is
replayed about 100 times per stage, and the training loss ends about half as high as at 2%. The 2% buffer is fit more tightly than other frames too, to 0.7 of
θ0's error. Memorization is shown; that it causes the learning cost is not, since the policy's offline
fit to the current task does not suffer. Continual classification work disagrees on whether fitting a
small buffer to near-zero loss hurts: Chaudhry et al. find that tiny memories still help
[11], Verwimp et al. that rehearsal overfits the buffer and leaves the model near a
high-loss region of the old task [12].

## 7. Noise: what these seeds can detect

At 0.2%, the seed moves the final success by up to 23 pts, several times the evaluation noise. The
three 0.2% seeds end at 14, 19.5 and 37% on Spatial and at 13, 28 and 31% on Goal (SD 12 and 10
pts), while a 200-episode probe carries about ±5 pts of episode noise. A seed value sets both which
frames the buffer holds and the training randomness, so we cannot say which of the two moves the
result; the distinction matters for any method that chooses the buffer's frames. Measured across the
three seeds, the 95% interval of the 0.2% mean runs from 0 to about 0.7 of θ0 on Spatial and to
about 0.6 on Goal, far wider than the episode-noise intervals of appendix B. At 2% and with every
frame the two seeds differ by at most 9.5 pts, but two seeds cannot estimate a spread.

With three seeds per condition at the 0.2% spread, only differences of about 30 pts are detectable.
At 80% power and a two-sided 5% test with four degrees of freedom, the smallest detectable difference
is 36 pts on Spatial and 29 on Goal. A method compared against random replay at a small budget needs
more seeds than this, or a budget at which the seeds agree more closely.

Offline measures do not substitute for closed-loop evaluation here. The mean change of θ0's
predicted arm translation on fixed Spatial frames ranks the budgets, but within 0.2% the three seeds
agree on it to 0.03 at every stage while their final Spatial success spans 14 to 37% (appendix D).

## 8. Related work

Continual-learning studies of VLAs replay the tasks of the training sequence and measure forgetting
of those tasks. Liu et al. teach π0 and GR00T N1.5 sequences of LIBERO tasks and find little forgetting
with experience replay, even with buffers of a few percent of each earlier task, and substantial
forgetting without it [2]. LIBERO's own sequential baselines forget heavily [6]. We replay
θ0's pretraining data and measure θ0's pretraining skills, so our NBT on the new tasks is not
comparable with theirs; besides the setting, the definition of NBT differs (section 2.6). Our 2%
budget still loses a fifth of θ0's Spatial and Goal success and three quarters of its LIBERO-10
success, which reads as more forgetting than they report, though the measures differ. So do the
models (theirs are larger, with other training recipes) and what is replayed (40 pretraining tasks
rather than the few earlier tasks of the sequence); we have not tested which of these matters.

That fine-tuning a pretrained VLA erases its earlier skills, and that mixing pretraining data back in
helps, has been shown before. RETAIN fine-tunes π0 on a new LIBERO task, reports its success on the
pretraining tasks falling as training proceeds, and finds that co-training with pretraining data
helps, and that merging the fine-tuned and pretrained weights keeps the generalist's skills
[3]. Garg et al. report that fine-tuning π0.5 erases previously learned behaviour, with the loss
concentrated after the grasp [13]. In a search of arXiv and recent robot-learning and
machine-learning venues (November 2026) we did not find a sweep of how much pretraining data to
replay through a task sequence, or a sub-goal-level account of what small budgets fail to keep.

Other routes reduce forgetting in VLAs without stored data, and we tested none of them. Task-specific
adapters on a frozen backbone avoid forgetting by construction [14]. LoRA combined with on-policy
reinforcement learning lets a pretrained VLA learn task sequences with little forgetting, and
removing either part brings it back [15]. More generally, forgetting tracks how far fine-tuning
moves the policy from the base model on the new task's inputs, which on-policy updates keep small
[16]. θ0's recipe, supervised fine-tuning of the whole action expert, is the setting in which
these studies report the most forgetting.

Replay-design work argues that uniform frame sampling under-samples short, critical phases such as
grasps, and reports large gains from phase-aware sampling on LIBERO-Goal and LIBERO-Long [10]. At
2% we find the second sub-goals of LIBERO-10 tasks sampled at their share of the data, so phase
starvation does not explain our loss. Our equal split per task does give long tasks less coverage
than short ones, an allocation problem of a different kind, and splitting by task length recovers
part of the loss (section 5).

LIBERO success largely reflects reproduction of trained layouts: OpenVLA and π0 collapse when objects
are displaced slightly, and policies keep their trajectories under nonsense instructions
[4]; a blank instruction barely changes OpenVLA-OFT's success except on LIBERO-Long
[17]. Our retention numbers inherit that scope. They measure whether θ0 still reproduces
trained behaviour from known layouts; θ0 does use its instruction on Goal, where only the instruction
tells tasks apart (section 2.3).

## 9. Limitations

The study tests one model in one setting. θ0 is a 605M-parameter VLA whose trained part has 97.5M
parameters, and every run uses its supervised recipe for 8 passes. In image and text
classification, pretrained and larger models forget less [18]; whether larger VLAs need
smaller replay budgets we cannot say.

The tasks are few and fixed: five new tasks in one order, in three kitchen scenes, evaluated from
LIBERO's fixed initial layouts. LIBERO-Object is not probed. Retention here is retention of
in-distribution behaviour, not of skill under new layouts or instructions.

Seeds are few. The no-replay sequence has one seed, 2% and every frame two, 0.2% three, and the
split by task length one. One seed value sets both the buffer and the training randomness,
so the two sources of spread are not separated, and the learning cost is measured against references
that share one training seed. Replay stages take twice the optimizer steps of no-replay stages,
which would, if anything, add forgetting under replay.

We compare replay only with no replay. Methods that need no stored data, such as merging the
fine-tuned and θ0's weights [3], adapters [14] or a penalty on moving away from θ0, are left for
later work.

Success is binary per episode, and repeated evaluations do not reproduce exactly. θ0 is weak on
LIBERO-10 and below the SmolVLA paper's numbers on every probe (section 2.5), which makes retention
on LIBERO-10 noisier than on Spatial and Goal.

## 10. Conclusion

For a deployed SmolVLA at this scale, fine-tuning on a new task without its own data in the batch
removes its skills (on the one old task we tracked, within the first hundred steps), and a lower
learning rate does not prevent it. Replaying a fixed random sample of its own training data keeps
them, and more of them the larger the budget: 2% of θ0's data, 5,600 frames here, keeps about four
fifths of its success on suites in other scenes through five new tasks, and all of it keeps nearly
everything. Long tasks keep least, and splitting the same budget by task length rather than equally
recovers part of what they lose, in one seed. For a setup like ours, this argues for splitting
a budget by task length, judging it after several new tasks rather than one, and running enough
seeds to see a difference before claiming one. The levers left are how a fixed budget is split
across tasks and methods that need no stored data.

## References

1. Shukor et al. SmolVLA: A vision-language-action model for affordable and efficient robotics. arXiv:2506.01844, 2025.
2. Liu, Kim, Liu, Liu and Zhu. Pretrained vision-language-action models are surprisingly resistant to forgetting in continual learning. arXiv:2603.03818, 2026.
3. Yadav, Zhou, Wagenmaker, Pertsch and Levine. Robust finetuning of vision-language-action robot policies via parameter merging. ICLR 2026. arXiv:2512.08333.
4. Zhou et al. LIBERO-PRO: Towards robust and fair evaluation of vision-language-action models beyond memorization. arXiv:2510.03827, 2025.
5. Hugging Face. `HuggingFaceVLA/smolvla_libero`, model card, revision 6721902.
6. Liu et al. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. NeurIPS 2023 Datasets and Benchmarks Track. arXiv:2306.03310.
7. Hugging Face. `lerobot/libero`, dataset (LIBERO-Spatial, Object, Goal and 10 in LeRobot format).
8. NVIDIA. `nvidia/LIBERO_LeRobot_v3`, dataset, revision e590737.
9. Cadene et al. LeRobot: An open-source library for end-to-end robot learning. ICLR 2026. arXiv:2602.22818. Used at version 0.6.2, commit 9a6bb61.
10. Chen et al. PHASER: Phase-aware and semantic experience replay for vision-language-action models. arXiv:2606.03598, 2026.
11. Chaudhry et al. On tiny episodic memories in continual learning. arXiv:1902.10486, 2019.
12. Verwimp, De Lange and Tuytelaars. Rehearsal revealed: The limits and merits of revisiting samples in continual learning. ICCV 2021. arXiv:2104.07446.
13. Garg, Xing, Yaugand, Gupta and Hoiem. Fine-tuning VLAs with self-demonstrated generative control for multi-task manipulation. arXiv:2608.19490, 2026.
14. Liu et al. TAIL: Task-specific adapters for imitation learning with large pretrained models. ICLR 2024. arXiv:2310.05905.
15. Hu et al. Simple recipe works: Vision-language-action models are natural continual learners with reinforcement learning. RLC 2026. arXiv:2603.11653.
16. Shenfeld, Pari and Agrawal. RL's Razor: Why online reinforcement learning forgets less. arXiv:2509.04259, 2025.
17. Fei et al. LIBERO-Plus: In-depth robustness analysis of vision-language-action models. arXiv:2510.13626, 2025.
18. Ramasesh, Lewkowycz and Dyer. Effect of scale on catastrophic forgetting in neural networks. ICLR 2022 (OpenReview GhVS8_yPeEa).

## Appendix A. Reproduction and cost

- **Pins.** LeRobot 0.6.2 at commit `9a6bb61`; torch 2.11.0 with CUDA 12.6, transformers 5.5.4, numpy
  2.2.6, MuJoCo 3.8.1, robosuite 1.4.0, hf_libero 0.1.4, PyAV 15.1.0 (`requirements-lock.txt`). θ0:
  `HuggingFaceVLA/smolvla_libero` at `6721902`; its VLM, `SmolVLM2-500M-Video-Instruct`, at `7b375e1`;
  `nvidia/LIBERO_LeRobot_v3` at `e590737`; LIBERO assets at `0b3ea86` (`adaptive_vla/config.py`).
- **Compute** (A100 GPU-hours): single-task experiments of section 3 and the setup, 14.5; replay
  experiments, 71.4, including 1.2 for a second seed of the split by task length that was stopped
  after two stages; a few more for mechanism checks not reported here. One five-stage replay
  sequence with its final probes takes 6 to 7. Every run trains 97.5M parameters. Buffer at the last
  stage: 616 frames at 0.2%, 6,160 at 2%, 305,311 with every frame (stored as indices into the
  dataset). As decoded observations that is 0.24, 2.4 and 120 GB; the dataset's compressed video
  takes far less.
- **Rerunning.** Environment and assets: the README. A replay sequence:
  `experiments/m2/er_seed_chain.sh K BUFFER_SEED TRAINING_SEED`; a single-task fine-tune:
  `scripts/finetune_eval.sh`; probes: `scripts/probe_eval.sbatch`. Tables and figures:
  `scripts/collect_results.py`, `experiments/m2/m2_metrics_ci.py`, `experiments/m2/m2_task_analysis.py`,
  `experiments/m2/phase_coverage.py`, `experiments/m2/step_conditional.py`, `scripts/make_figures.py`. Every number in this report traces to a row of `results/*.csv`, except a
  few computed directly from per-episode evaluation outputs or dataset metadata, which are not in the
  repository: the paired render-size test, the agreement between repeated evaluations, and frame
  counts. For each section, `results/README.md` lists the experiment ids, the launcher and the
  run labels in `results/runs.csv`.
- **Further pitfalls.** SmolVLA loads its VLM and tokenizer by name without a revision, so an offline
  cache needs that name's default revision pinned. Fine-tuned checkpoints save tokenizer files that
  differ from the original's yet tokenize identically; compare tokenizer outputs, not files. The two
  LIBERO copies label their frame rate 10 and 20 fps, but both step at 20 Hz. Two frames of one
  upstream LIBERO-Goal wrist-camera video cannot be decoded; they are dropped from the every-frame
  buffer.

## Appendix B. Per-run results

*Table B1. θ0 by protocol. Each comparison in the report uses the row of its own protocol.*

| Protocol | Spatial | Goal | LIBERO-10 | Used in |
|---|---|---|---|---|
| single fine-tune: 10 x 10; same-scene task x 50 | 76/100 | 74/100 (first 10 initial states of a 152/200 run) | task 3: 42/50 | section 3 |
| final probes: 10 x 20; tasks 2, 3, 9 x 50; every run pooled | 147/200 | 308/400 (156 + 152) | 52/100, 83/100, 49/100 (two runs each) | sections 4 to 7 |
| per-stage subsets: 5 x 10, 6 x 10; task 3 x 20; two runs pooled | 83/100 | 110/120 | 39/40 | Figure 4 |
| far-scene tasks 0, 1, 4, 6, 7 x 20 | | | 35/100 | section 5 |

The failure anatomy and the sub-goal analysis of section 5 need per-episode sub-goal logs, which one
run of each LIBERO-10 task lacks; their θ0 reference is the run that has them (27/50, 41/50, 21/50).

*Table B2. Final probes per seed (%, 95% bootstrap interval over episodes), NBT on the new tasks, and
the final average over the five new tasks (%). Source: `results/m2_metrics_ci.csv`.*

| Budget | Seed | Spatial | Goal | LIBERO-10 | NBT | New tasks |
|---|---|---|---|---|---|---|
| none | 1 | 0 (of 100) | 0 (of 100) | 0 (of 60) | 0.72 [0.66, 0.77] | 17 |
| 0.2% | 1 | 37 [31, 42.5] | 13 [9, 17] | 0 | 0.39 [0.31, 0.46] | 26 |
| 0.2% | 2 | 19.5 [15, 24] | 28 [23.5, 32] | 0 | 0.51 [0.43, 0.59] | 14 |
| 0.2% | 3 | 14 [10, 18.5] | 31 [26, 36] | 3 [1, 6] | 0.46 [0.38, 0.53] | 28 |
| 2% | 1 | 59 [52.5, 65] | 54.5 [48, 60.5] | 15 [10, 21] | 0.29 [0.20, 0.37] | 55 |
| 2% | 2 | 63.5 [57.5, 69] | 64 [58.5, 69.5] | 15 [9, 20] | 0.26 [0.18, 0.34] | 58 |
| every frame | 1 | 75.5 [70, 80.5] | 74 [68, 79.5] | 53 [45, 60] | 0.21 [0.12, 0.30] | 60 |
| every frame | 2 | 71 [64.5, 77] | 69.5 [64, 75] | 52 [45, 59] | 0.17 [0.08, 0.26] | 60 |
| 2%, split by task length | 1 | 57 [51, 63.5] | 56.5 [51, 62] | 29 [21, 36] | not measured | not measured |

Retention of the seed mean, Spatial / Goal / LIBERO-10, with 95% intervals that resample θ0's episodes
too: 0.2% 0.32 [0.28, 0.37] / 0.31 [0.28, 0.35] / 0.02 [0.00, 0.04]; 2% 0.83 [0.75, 0.93] / 0.77
[0.71, 0.84] / 0.24 [0.18, 0.31]; every frame 1.00 [0.91, 1.09] / 0.93 [0.87, 1.00] / 0.85 [0.75,
0.96]; 2% split by task length 0.78 [0.68, 0.89] / 0.73 [0.66, 0.81] / 0.47 [0.35, 0.59]. These
intervals cover episode noise only. At 0.2%, where three seeds allow it, a t-interval over seeds of
the mean retention runs from 0 to 0.73 on Spatial and from 0 to 0.62 on Goal.

*Table B3. Success on the task just learned (%, 50 episodes per seed, seed means) and the mean
learning cost against the no-replay reference (pts, 95% interval over episodes).*

| | Task 26 | Task 18 | Task 23 | Task 27 | Task 34 | Mean cost |
|---|---|---|---|---|---|---|
| no-replay reference | 88 | 82 | 45 | 73 | 87 | |
| 0.2% | 93 | 65 | 55 | 41 | 72 | -9.9 [-14.5, -5.2] |
| 2% | 92 | 75 | 52 | 57 | 86 | -2.6 [-7.8, 2.8] |
| every frame | 93 | 79 | 52 | 65 | 84 | -0.4 [-5.6, 5.0] |

![Success matrices](figures/fig10_m2_matrices.png)

*Figure B1. Success on each new task after each stage, seed means; diagonal: the task just learned
(50 episodes), below it: earlier tasks (20 episodes).*

## Appendix C. Predictions and outcomes

Predictions were written into a private development log before each run; its timestamps are not
published, so Table C1 is our record of the predictions rather than something a reader can check
against the log. One line per experiment, compressed from the log entry; "n of m" counts the parts
of the prediction that held. The single-task experiments of section 3 are left out because
predictions were not written for all of them.

*Table C1. Predictions and outcomes, one line per experiment.*

| Experiment | What was predicted | Outcome |
|---|---|---|
| Single-stage pilot | every-frame replay near θ0; a knee between 0.2% and 2%; task 23 needs more replay than task 26; offline drift falls with the budget | 2 of 4: every-frame replay and drift held; no knee (0.2% already kept most after one stage); task 23 needed no more |
| 0.2%, three seeds | learning within 10 pts of no replay; earlier new tasks 10 to 50% kept; NBT 0.3 to 0.6; Spatial 40 to 60%, Goal 45 to 70%; seeds within 15 pts | NBT held; earlier tasks partly (0 to 85%); learning, θ0's skills and seed spread wrong (learning and skills lower, spread 18 to 23 pts) |
| 2%, seed 1 | Spatial 35 to 55%, Goal 35 to 60%, LIBERO-10 at most 15%; NBT 0.15 to 0.35; learning like 0.2% | Goal, LIBERO-10 and NBT held; Spatial above range; learning wrong (better than at 0.2%) |
| θ0, final protocol | Spatial 70 to 82%; Goal within 6 pts of 76%; LIBERO-10 task 2 40 to 60%, task 9 46 to 66%; θ0 fails these by not starting them | Spatial, Goal and task 2 held; task 9 just below (42%); "not starting" held for task 9, wrong for task 2 |
| 2%, seed 2 | Spatial 45 to 65%, Goal 40 to 60%, both above every 0.2% seed; learning within 10 pts of no replay on 4 of 5 tasks; task 23 at most 10%; LIBERO-10 tasks 2 and 9 at most 15%; NBT 0.2 to 0.4 | Spatial, "above every 0.2% seed", LIBERO-10 and NBT held; Goal and task 23 just above range; learning held on 2 of 5 |
| Every frame, seed 1 | 5 to 15 pts above 2% but below θ0; task 23 and LIBERO-10 tasks 2 and 9 below 20%; learning within 10 pts of 2% | wrong on the first (reached θ0) and on LIBERO-10; held for task 23 and for learning on 4 of 5 tasks |
| Every frame, seed 2 | Spatial and Goal 65 to 80%, LIBERO-10 35 to 60%, NBT 0.15 to 0.30, task 23 at most 20%, task 27 below 70% | 4 of 6: task 23 (32%) and task 27 (72%) wrong |
| Memorization | buffered frames at most 0.7x θ0's error at 0.2%, other frames at least 1.3x; the current task fit worse at 0.2% than at 2% | 2 of 4: the buffer fit held; no spill-over to other frames, and the current task fits as well as at 2% |
| Failure anatomy, margin | the first sub-goal held at least twice as often as success at 2%, with no such gap for θ0; θ0's margin predicts retention (ρ ≥ 0.4) | anatomy held on 3 of 4 tasks (1.9x on the fourth); θ0 without the gap except on task 2; margin wrong |
| Far-scene control | θ0 60 to 85%; retention at 2% 0.50 to 0.80, scene the larger factor; every frame at least 0.85 | 1 of 5: θ0 35%; retention 0.27, by the pre-set rule the loss follows the suite; every frame just short (0.84) |
| Phase coverage | the second sub-goal sampled at its share of the demos | held |
| Task 9 at stage 4 | a recency drop of at least 15 pts | wrong; inconclusive by the pre-set rule |
| Proportional split | LIBERO-10 tasks 2, 3, 9 18 to 30%; far-scene 10 to 20%; Spatial and Goal within 8 pts of 2% | all four held (29%, 20%, 4 and 3 pts lower); in the middle band of the reading rule |
| Second sub-goal given the first | no prediction (added after review) | not scored; result in section 5 |

## Appendix D. Further checks

**Task 9 under every-frame replay.** Even with every frame, LIBERO-10 task 9 ends at 18% against θ0's
49%, and most of that loss happens before the last stage, which trains a new task in the same scene
with the same mug. Evaluated after stage 4, task 9 is 8 pts higher than after stage 5, pooled over
the two seeds, inside the noise; one seed shows no drop at all. We added this check after the second
every-frame run, with a reading rule fixed beforehand: at least 35% after stage 4 and 15 pts above
stage 5 would mean a recency effect of the last stage, at most 25% that the sequence wears the task
down. At 26% the result is inconclusive.

**Offline drift.** We measure drift as the mean absolute change of the first 10 predicted actions
from θ0's prediction on fixed Spatial demo frames, in units of θ0's action standard deviation. After
five stages the translation drift is about 0.7 without replay, 0.3 at 0.2%, 0.17 at 2% and 0.12 with
every frame; the rotation drift does not separate no replay from 0.2% (0.40 and 0.40 to 0.43).
Within 0.2%, the three seeds' translation drift agrees to 0.03 at every stage while their final
Spatial success spans 14 to 37%. New task 23 shows the same gap: offline, its frames stay close to
the fit it had right after training, while in closed loop it mostly fails. A mean over frames hides
the rarer, decision-critical errors that end an episode, so offline drift can rank conditions but
cannot replace closed-loop evaluation.
