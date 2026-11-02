"""Collect every eval and training run under $AVLA_RUNS into CSV tables committed to git.

  results/runs.csv                one row per (eval run, task): success, CI, policy, settings, provenance
  results/train_runs.csv          one row per training run: task, steps, passes, replay (k, mix, ledger), seed,
                                  start policy, LR, wall-clock, provenance
  results/train_curves.csv        one row per logged training step (every 50): loss, grad norm, LR
  results/offline_action_error.csv  predicted-action change on old-task demo frames (offline_action_error.py)
  results/weight_drift.csv        relative L2 drift vs θ0 per module group (weight_drift.py)
  results/lang_diag.csv           language-diagnostic outcome rates per condition
  results/coverage_offline.csv    per replayed task and k: random vs greedy k-center coverage (E40)
  results/coverage_buffers.csv    coverage of the actual replay buffers of the sequences (E40)
  results/buffer_memorization.csv  offline error on buffered vs unbuffered frames per replay run (E44)

Run ids are the output directory names (Slurm job id plus a label); paths are stored relative to
$AVLA_RUNS, so the tables carry no machine-specific paths. Rerun after new jobs (needs LIBERO for
task names: source scripts/env.sh first):
  python scripts/collect_results.py              writes results/, the committed tables
  python scripts/collect_results.py --out DIR    writes DIR instead (your own runs, kept apart from ours)
A table with no runs is skipped.
"""
import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from libero.libero.benchmark import get_benchmark_dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_vla.libero_data import frames_for_tasks, task_index_for_instruction  # noqa: E402

RUNS = Path(os.environ["AVLA_RUNS"])
CONV = Path(os.environ["AVLA_DSS"]) / "data/libero90_theta0"
ap = argparse.ArgumentParser(description="Collect the runs under $AVLA_RUNS into CSV tables.")
ap.add_argument("--out", default="results", help="output directory (default: results/, the committed tables)")
OUT = Path(ap.parse_args().out)
OURS = OUT.resolve() == Path("results").resolve()


def slurm_log(job):
    """A job's log: under $AVLA_RUNS, else in the current directory (the repository root, where a
    launcher without a log path writes it)."""
    p = RUNS / f"slurm-{job}.out"
    return p if p.exists() else Path(f"slurm-{job}.out")


def have(path):
    """Inputs of our own offline analyses (fixed job ids): required when writing results/, skipped
    when collecting someone else's runs into another directory."""
    if path.exists():
        return True
    if OURS:
        raise FileNotFoundError(path)
    return False

# Evals of a materialized (rebuilt-from-slim) checkpoint made before eval.sh recorded
# `policy_source` (2026-10-23): the source checkpoint per run, from the sbatch scripts.
MATERIALIZED = {
    "113802-ft23-1k": "train/113795/checkpoints/last",
    "114663-ft26": "train/113804/checkpoints/last",
    "114664-ft27": "train/113805/checkpoints/last",
    "113823-lr1e-5": "train/113823/checkpoints/last",
}
_bench = get_benchmark_dict()
_names = {}


def task_name(suite, tid):
    if suite not in _names:
        b = _bench[suite]()
        _names[suite] = [b.get_task(i).language for i in range(b.n_tasks)]
    return _names[suite][tid]


def policy_label(path, source, run):
    p = source or path
    if "models--HuggingFaceVLA--smolvla_libero" in p:
        return "theta0"
    m = re.search(r"train/(\d+)/checkpoints/([^/]+)", p)
    if m:
        return f"train/{m.group(1)}/checkpoints/{m.group(2)}"
    for key, src in MATERIALIZED.items():
        if run.startswith(key) or f"materialized/{key}" in p:
            return src
    return "unknown"


def arg(cmd, name, default=""):
    m = re.search(rf"--{re.escape(name)}=(\S+)", cmd)
    return m.group(1) if m else default


rows = []
for info_path in sorted(RUNS.glob("eval/*/*/eval_info.json")):
    run_dir = info_path.parent
    suite, run = run_dir.parent.name, run_dir.name
    info = json.load(open(info_path))
    ri = json.load(open(run_dir / "run_info.json")) if (run_dir / "run_info.json").exists() else {}
    cmd = ri.get("command", "")
    sub = {}
    if (run_dir / "subgoals_summary.json").exists():
        sub = json.load(open(run_dir / "subgoals_summary.json"))
    date = datetime.fromtimestamp(info_path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d")
    for t in sorted(info["per_task"], key=lambda t: t["task_id"]):
        tid = int(t["task_id"])
        s = sub.get(str(tid), {})
        lo, hi = t.get("pc_success_ci95", [None, None])
        rows.append({
            "run": run, "date": date, "suite": suite, "task_id": tid, "task": task_name(suite, tid),
            "policy": policy_label(ri.get("policy", ""), ri.get("policy_source"), run),
            "n_episodes": t["n_episodes"], "n_success": t["n_success"],
            "success_pct": round(100 * t["n_success"] / t["n_episodes"], 1),
            "ci95_lo": None if lo is None else round(lo, 1), "ci95_hi": None if hi is None else round(hi, 1),
            "render": arg(cmd, "env.observation_height", "360 (env default)"),
            "seed": arg(cmd, "seed"), "batch_size": arg(cmd, "eval.batch_size"),
            "subgoal_partial": s.get("partial", ""),
            "subgoals_final": ";".join(f"{g['goal']}={g['final']:.2f}" for g in s.get("goals", [])),
            "repo_commit": ri.get("repo_commit", "")[:10],
        })

def init_label(init, run):
    """Start policy of a training run as a policy label. Runs before the replay sequences (no `init`
    in run_info) all started from θ0. A materialized init (seq_stage.sbatch) is deleted after
    training, so its source comes from materialized_from.json if the dir survives, else from the
    stage's `from <PREV>` line (seq_stage.sbatch and er_stage.sbatch)."""
    if not init:
        return "theta0"
    m = re.search(r"/materialized/([^/]+)$", init.rstrip("/"))
    if not m:
        return policy_label(init, None, run)
    src = Path(init) / "materialized_from.json"
    if src.exists():
        return policy_label(json.load(open(src))["source"], None, run)
    hit = re.search(r"^== (?:Sequential|ER) stage \d+: .* from (\S+?)(?:;|$)",
                    slurm_log(run.split("-")[0]).read_text(errors="replace"), re.M)
    if not hit:
        raise ValueError(f"no source for materialized init {m.group(1)} of run {run}")
    return policy_label(hit.group(1), None, run)


tdf = pq.read_table(CONV / "meta" / "tasks.parquet").to_pandas()
tcol = tdf["task"] if "task" in tdf.columns else tdf.index.to_series()
train = []
for ri_path in sorted(RUNS.glob("train/*/run_info.json")):
    ri = json.load(open(ri_path))
    run = ri_path.parent.name
    if "--policy.device=cpu" in ri.get("command", ""):
        continue                         # CPU plumbing check (e.g. the replay smoke), not a result
    if ri["tasks"] == "-":
        # Replay run (train.sh with AVLA_REPLAY_SPEC): the spec names the current task, and only the
        # (1 - mix) share of each batch is current-task frames.
        spec = json.load(open(ri_path.parent / "replay_spec.json"))
        suite, instruction = spec["current_key"].split(":", 1)
        assert suite == "libero_90", spec["current_key"]
        idx = [task_index_for_instruction(str(CONV), instruction)]
        frames, mix, led = spec["ledger"]["current_frames"], spec["mix"], spec["ledger"]
        replay = {"k": spec["k"], "mix": mix, "replay_frames": led["replay_frames"],
                  "replay_mb_obs": led["replay_mb_obs"], "replay_seed": spec["seed"]}
    else:
        idx = json.loads(ri["tasks"])
        frames, mix = frames_for_tasks(str(CONV), idx), 0.0
        replay = {"k": "", "mix": 0.0, "replay_frames": 0, "replay_mb_obs": 0.0, "replay_seed": ""}
    names = [str(tcol[tdf["task_index"].to_numpy() == i].iloc[0]) for i in idx]
    # E35 (scripts/optim_train.py): runs stopped early inside a longer schedule, SGD, warmed Adam
    stop = ri.get("stop_step", "none")
    steps = int(stop) if stop not in ("none", "") else ri["steps"]
    cmd = ri.get("command", "")
    opt = re.search(r"--optimizer\.type=(\S+)", cmd)
    optimizer = opt.group(1) if opt else "adamw"
    sgd_lr = re.search(r"--optimizer\.lr=(\S+)", cmd)
    if int(ri.get("warm_adam", 0) or 0):
        optimizer += f" (warmed, {ri['warm_adam']} zero-LR updates)"
    ckpts = sorted(p.name for p in (ri_path.parent / "checkpoints").iterdir() if p.name != "last") \
        if (ri_path.parent / "checkpoints").exists() else []
    train.append({
        "run": run, "dataset_task_index": ";".join(map(str, idx)), "task": "; ".join(names),
        "steps": steps, "schedule_steps": ri["steps"], "optimizer": optimizer, "batch_size": ri["batch_size"],
        "passes": round(steps * ri["batch_size"] * (1 - mix) / frames, 2), "task_frames": frames,
        **replay,
        # train.sh passed no --seed before the replay sequences: the pinned lerobot-train default is 1000
        "seed": ri.get("seed", 1000), "init": init_label(ri.get("init"), run),
        "peak_lr": sgd_lr.group(1) if sgd_lr else
                   {"default": "default (1e-4)"}.get(ri.get("peak_lr", "default"), ri.get("peak_lr")),
        "loss_dims": ri.get("loss_dims", "all"), "checkpoints": ";".join(ckpts),
        "wall_clock_s": ri.get("wall_clock_s", ""), "repo_commit": ri.get("repo_commit", "")[:10],
    })

# --- training curves: `step:50 ... loss:... grdn:... lr:...` every log_freq=50 steps (train.sh) --
def train_log(run):
    """Log text of one training run. Most jobs train once (slurm-<run>.out); a job that trains several
    runs in sequence (E26's counterfactuals: <job>-arm, <job>-grip) logs them one after another, each
    segment ending in train.sh's `TRAIN DONE -> .../train/<run>` line."""
    job = run.split("-")[0]
    text = slurm_log(job).read_text(errors="replace")
    if run == job:
        return text
    start = 0
    for m in re.finditer(r"TRAIN DONE -> \S*/train/(\S+)", text):
        if m.group(1) == run:
            return text[start:m.start()]
        start = m.end()
    raise ValueError(f"no training segment for {run} in slurm-{job}.out")


curves = []
for tr in train:
    lines = re.findall(r"step:\S+ smpl:\S+ ep:\S+ epch:(\S+) loss:(\S+) grdn:(\S+) lr:(\S+)", train_log(tr["run"]))
    if not lines:
        raise ValueError(f"no training-step lines for run {tr['run']}")
    for i, (epoch, loss, grdn, lr) in enumerate(lines, start=1):
        # lerobot's epch counts passes over the whole loaded dataset; a replay run loads the merged
        # libero_all_theta0, so its passes over the current task come from the batch share instead
        passes = float(epoch) if tr["mix"] == 0 else \
            round(50 * i * tr["batch_size"] * (1 - tr["mix"]) / tr["task_frames"], 3)
        curves.append({"run": tr["run"], "step": 50 * i, "passes": passes, "loss": float(loss),
                       "grad_norm": float(grdn), "lr": float(lr)})

# --- offline action error: printed tables of scripts/offline_action_error.py --------------------
# Labels were given on the command line; their checkpoints (from the submitting commands):
OFFLINE_LOGS = ("5683297", "5683518", "5685430", "5688449",
                "5688443", "5688444", "5688445", "5688446", "5688447", "5688448", "5688450",   # E30, E32
                "5689835", "5689836", "5689837",                                                # E36, E37
                "5696717",                                                                     # E39
                "5699031", "5699032",                                                          # E42, E43
                "5699711")                                                                     # E46
PILOTS = {"p26k14": (115608, 664), "p26k140": (115609, 664), "p26k700": (115610, 664),   # label: (job, mid step)
          "p26kall": (115611, 664), "p23k140": (115612, 1010), "p23kall": (115613, 1010)}
OFFLINE_LABELS = {
    **{f"{p}-{w}": f"train/{j}/checkpoints/{d}" for p, (j, mid) in PILOTS.items()
       for w, d in (("s100", "000100"), ("mid", f"{mid:06d}"), ("last", "last"))},     # E30 pilot
    **{f"seq-s{s}": f"train/{115601 + s}/checkpoints/last" for s in range(2, 6)},       # E32 Sequential
    "holdout-noGoal": "train/115617/checkpoints/last",                                   # E36
    **{f"er-b{b}-s{s}": f"train/{j}/checkpoints/last" for b, jobs in                    # E37 ER k = 14, 3 seeds
       ((0, (115608, 115620, 115621, 115622, 115623)), (1, range(115626, 115631)), (2, range(115632, 115637)))
       for s, j in enumerate(jobs, 1) if (b, s) != (0, 1)},                              # b0-s1 is p26k14-last
    **{f"er140-b0-s{s}": f"train/{115756 + s}/checkpoints/last" for s in range(2, 6)},  # E39 k = 140 (s1 = p26k140-last)
    **{f"er140-b1-s{s}": f"train/{115785 + s}/checkpoints/last" for s in range(1, 6)},  # E42 k = 140 seed 2
    **{f"erall-b0-s{s}": f"train/{115790 + s}/checkpoints/last" for s in range(2, 6)},  # E43 full (s1 = p26kall-last)
    **{f"erall-b1-s{s}": f"train/{115814 + s}/checkpoints/last" for s in range(1, 6)},  # E46 full seed 2
    "theta0": "theta0", "ft1k": "train/113795/checkpoints/last", "ft2k": "train/113796/checkpoints/last",
    "ft3k": "train/113797/checkpoints/last", "ft18": "train/113803/checkpoints/last",
    "ft26": "train/113804/checkpoints/last", "ft27": "train/113805/checkpoints/last",
    "s100": "train/113822/checkpoints/000100", "s200": "train/113822/checkpoints/000200",
    "s400": "train/113822/checkpoints/000400", "s700": "train/113822/checkpoints/000700",
    "lr1e-5": "train/113823/checkpoints/last",
    **{f"i23-a{a}": f"interp/113795/a{a}" for a in ("0.1", "0.2", "0.3", "0.5", "0.7")},   # E31 interpolations
}
num = r"(\d+\.\d+)"
row_re = re.compile(rf"^(\S+)\s+(spatial|goal|l10_s4|task23)\s+{num} / {num} / {num}\s+{num} / {num} / {num}\s*$", re.M)
offline, seen = [], set()
for job in OFFLINE_LOGS:
    if not have(RUNS / f"slurm-{job}.out"):
        continue
    text = (RUNS / f"slurm-{job}.out").read_text(errors="replace")
    if "OFFLINE ACTION ERROR DONE" not in text:
        raise ValueError(f"offline action-error job {job} did not finish")
    for m in row_re.finditer(text):
        label, fset, *v = m.groups()
        if label not in OFFLINE_LABELS:
            raise ValueError(f"unknown offline label {label!r} in job {job}")
        if (label, fset) in seen:
            continue                     # θ0 is re-evaluated in every job (identical: fixed frames and noise)
        seen.add((label, fset))
        v = [float(x) for x in v]
        offline.append({"label": label, "checkpoint": OFFLINE_LABELS[label], "frame_set": fset,
                        "vs_demo_trans": v[0], "vs_demo_rot": v[1], "vs_demo_grip": v[2],
                        "vs_theta0_trans": v[3], "vs_theta0_rot": v[4], "vs_theta0_grip": v[5], "job": job})

# --- weight drift: `== 1k: 97.4M parameters changed, relative L2 drift 0.0587` + group lines -----
DRIFT_LOG, DRIFT_LABELS = "113800", {"1k": "train/113795/checkpoints/last", "2k": "train/113796/checkpoints/last",
                                     "3k": "train/113797/checkpoints/last"}
drift, cur = [], None
drift_log = RUNS / f"slurm-{DRIFT_LOG}.out"
for line in drift_log.read_text(errors="replace").splitlines() if have(drift_log) else []:
    m = re.match(r"== (\S+): ([\d.]+)M parameters changed, relative L2 drift ([\d.]+)", line)
    if m:
        cur = m.group(1)
        drift.append({"label": cur, "checkpoint": DRIFT_LABELS[cur], "group": "all", "params_M": float(m.group(2)),
                      "rel_l2": float(m.group(3))})
        continue
    m = re.match(r"\s{3}(.+?)\s+([\d.]+)M\s+rel L2 ([\d.]+)$", line)
    if m and cur:
        drift.append({"label": cur, "checkpoint": DRIFT_LABELS[cur], "group": m.group(1).strip(),
                      "params_M": float(m.group(2)), "rel_l2": float(m.group(3))})
if not drift and drift_log.exists():
    raise ValueError("no weight-drift lines found")

# --- offline buffer coverage (experiments/coverage/coverage_offline.py, E40) ----------------------------
COVERAGE_JOB = "5696915"
coverage, coverage_buf = ([dict(r, job=COVERAGE_JOB) for r in csv.DictReader(open(RUNS / "coverage" / COVERAGE_JOB / f))]
                          if have(RUNS / "coverage" / COVERAGE_JOB) else []
                          for f in ("coverage_per_task.csv", "coverage_actual_buffers.csv"))

# --- buffer memorization (experiments/replay/buffer_memorization.sbatch, E44) -----------------------
MEMO_RUNS = {"memo-k14-b0": "train/115623/checkpoints/last", "memo-k14-b1": "train/115630/checkpoints/last",
             "memo-k14-b2": "train/115636/checkpoints/last", "memo-k140-b0": "train/115761/checkpoints/last"}
memo = []
for run, ckpt in MEMO_RUNS.items():
    if not have(RUNS / "memorization" / run / "table.csv"):
        continue
    for r in csv.DictReader(open(RUNS / "memorization" / run / "table.csv")):
        memo.append({"run": run, "model": "theta0" if r["label"] == "theta0" else ckpt, **r})

# --- language diagnostic ------------------------------------------------------------------------
lang = []
for summ in sorted(RUNS.glob("lang_diag/*/lang_diag_summary.json")):
    if summ.parent.name.endswith("-cpu"):
        continue                         # CPU plumbing check, not a result
    d = json.load(open(summ))
    for cond, c in d["conditions"].items():
        for outcome in ("success", "own_goal", "partner_goal", "no_goal"):
            o = c[outcome]
            lang.append({"run": summ.parent.name, "condition": cond, "outcome": outcome, "k": o["k"], "n": o["n"],
                         "pct": o["pc"], "ci95_lo": round(o["ci95"][0], 1), "ci95_hi": round(o["ci95"][1], 1)})

OUT.mkdir(parents=True, exist_ok=True)
for name, data in (("runs.csv", rows), ("train_runs.csv", train), ("train_curves.csv", curves),
                   ("offline_action_error.csv", offline), ("weight_drift.csv", drift), ("lang_diag.csv", lang),
                   ("coverage_offline.csv", coverage), ("coverage_buffers.csv", coverage_buf),
                   ("buffer_memorization.csv", memo)):
    if not data:
        print(f"skipped {name}: no runs")
        continue
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    print(f"wrote {OUT / name}: {len(data)} rows")
