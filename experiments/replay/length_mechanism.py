"""E57 + E58: why do long tasks lose more? Two mechanism tests on E56's per-task retention table.

  python experiments/replay/length_mechanism.py

E57 (per-task offline drift): per checkpoint, arm drift of each old task = mean of translation and
rotation change of the predicted actions vs θ0 (runs/offline/e57-*.csv, θ0 action-std units).
  (a) the policy changes more on long tasks:   mean over checkpoints of ρ(arm drift, length) ≥ +0.4
  (b) similar change, more steps to compound: that mean < +0.2 and the length-retention ρ stays at
      most −0.4 after partialling out drift (mean over the 9 informative checkpoints)
  anything else is mixed.
E58 (noise tolerance of θ0): tolerance = mean over σ = 0.5, 1, 2 of success / θ0's setup success
(eval logs of job 116077). ρ(tolerance, length) ≤ −0.4 and ρ(tolerance, retention) ≥ +0.4 on
either target means long tasks lose more at least partly through lower tolerance of per-step error;
ρ(tolerance, length) > −0.2 means not through this kind of error; anything else is mixed.

Retention targets come from results/task_survival.csv (E56): single_mean, replay_mean. Spearman ρ
with permutation p over the 20 tasks; partial ρ = Pearson partial correlation of ranks.
Writes results/length_mechanism.csv and prints the readings. Torch-free.
"""
import csv
import os
import re
from pathlib import Path

import numpy as np

from task_survival import perm_p, ranks, spearman

REPO = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ["AVLA_RUNS"])
E57_CSVS = sorted((RUNS / "offline").glob("e57-*.csv"))
E58_JOB, SIGMAS = "116077", ("0.5", "1", "2")
FRAME_SET = {"spatial_t": "libero_spatial", "goal_t": "libero_goal"}
# E56's informative runs: the replay sequences and the single fine-tunes that do not zero every task.
INFORMATIVE = ["replay_k14-b0", "replay_k14-b1", "replay_k14-b2", "replay_k140-b0", "replay_k140-b1",
               "replay_kall-b0", "replay_kall-b1", "single_ft26", "single_ft27"]
TARGETS = ("single_mean", "replay_mean")


def partial(x, y, z):
    """Spearman-type partial correlation of x and y given z (Pearson on ranks)."""
    rx, ry, rz = ranks(x), ranks(y), ranks(z)
    c = np.corrcoef([rx, ry, rz])
    return float((c[0, 1] - c[0, 2] * c[1, 2]) / np.sqrt((1 - c[0, 2] ** 2) * (1 - c[1, 2] ** 2)))


def load_survival():
    with open(REPO / "results/task_survival.csv") as f:
        rows = list(csv.DictReader(f))
    return {(r["suite"], r["task"]): r for r in rows}


def load_drift(table):
    """label -> {(suite, task): arm drift}."""
    drift = {}
    for p in E57_CSVS:
        with open(p) as f:
            for r in csv.DictReader(f):
                if r["label"] == "theta0":
                    continue
                key = (FRAME_SET[r["frame_set"]], r["task"])
                assert key in table, f"E57 task not in task_survival.csv: {key}"
                arm = (float(r["vs_theta0_trans"]) + float(r["vs_theta0_rot"])) / 2
                drift.setdefault(r["label"], {})[key] = arm
    return drift


def load_noise():
    """(suite, task_id) -> {sigma: success share}."""
    out = {}
    for s in SIGMAS:
        for suite in FRAME_SET.values():
            log = RUNS / "eval" / f"{E58_JOB}-n{s}-{suite}.log"
            for m in re.finditer(r"^task\s+(\d+): success (\d+)/(\d+)", log.read_text(), re.M):
                out.setdefault((suite, int(m[1])), {})[s] = int(m[2]) / int(m[3])
    return out


def main():
    table = load_survival()
    keys = sorted(table, key=lambda k: (k[0], int(table[k]["task_id"])))
    length = np.array([float(table[k]["demo_frames"]) for k in keys])
    target = {t: np.array([float(table[k][t]) for k in keys]) for t in TARGETS}

    drift = load_drift(table)
    assert all(len(d) == len(keys) for d in drift.values()), "a checkpoint is missing tasks"
    print(f"== E57: arm drift vs θ0 ({len(drift)} checkpoints, {len(keys)} tasks)")
    rho_len = {}
    for label in sorted(drift):
        d = np.array([drift[label][k] for k in keys])
        rho_len[label] = spearman(d, length)
        line = f"  {label:20s} mean drift {d.mean():.3f}  range {d.min():.3f}-{d.max():.3f}  ρ(drift, length) {rho_len[label]:+.2f}"
        if label in INFORMATIVE:
            ret = np.array([float(table[k][label]) for k in keys])
            line += (f"  ρ(drift, ret) {spearman(d, ret):+.2f}  ρ(len, ret) {spearman(length, ret):+.2f}"
                     f"  partial ρ(len, ret | drift) {partial(length, ret, d):+.2f}")
        print(line)
    mean_all = float(np.mean(list(rho_len.values())))
    mean_inf = float(np.mean([rho_len[l] for l in INFORMATIVE]))
    part = [partial(length, np.array([float(table[k][l]) for k in keys]),
                    np.array([drift[l][k] for k in keys])) for l in INFORMATIVE]
    mean_part = float(np.mean(part))
    print(f"  mean ρ(drift, length): all {mean_all:+.2f}, informative {mean_inf:+.2f}")
    print(f"  mean partial ρ(length, retention | drift), informative: {mean_part:+.2f}")
    if mean_all >= 0.4:
        e57 = "(a) the policy changes more on long tasks"
    elif mean_all < 0.2 and mean_part <= -0.4:
        e57 = "(b) similar change, long tasks have more steps to compound it"
    else:
        e57 = "mixed"
    print(f"  reading: {e57}")
    mean_drift = np.array([np.mean([drift[l][k] for l in INFORMATIVE]) for k in keys])

    noise = load_noise()
    theta0 = np.array([float(table[k]["theta0_setup"]) for k in keys])
    succ = {s: np.array([noise[(k[0], int(table[k]["task_id"]))][s] for k in keys]) for s in SIGMAS}
    tol = np.mean([succ[s] / theta0 for s in SIGMAS], axis=0)
    print(f"\n== E58: noise tolerance of θ0 ({len(keys)} tasks)")
    for s in SIGMAS:
        print(f"  σ {s:3s}: success {succ[s].mean():.2f}  (θ0 {theta0.mean():.2f})")
    r_tl, p_tl = spearman(tol, length), perm_p(tol, length)
    print(f"  ρ(tolerance, length) {r_tl:+.2f}  p {p_tl:.3f}")
    r_tr = {}
    for t in TARGETS:
        r_tr[t] = spearman(tol, target[t])
        print(f"  {t:12s} ρ(tol, ret) {r_tr[t]:+.2f}  p {perm_p(tol, target[t]):.3f}"
              f"  ρ(len, ret) {spearman(length, target[t]):+.2f}"
              f"  partial ρ(len, ret | tol) {partial(length, target[t], tol):+.2f}"
              f"  partial ρ(tol, ret | len) {partial(tol, target[t], length):+.2f}")
    print(f"  ρ(tolerance, θ0 success) {spearman(tol, theta0):+.2f}  ρ(tolerance, mean drift) {spearman(tol, mean_drift):+.2f}")
    if r_tl <= -0.4 and max(r_tr.values()) >= 0.4:
        e58 = "long tasks lose more at least partly through lower tolerance of per-step error"
    elif r_tl > -0.2:
        e58 = "the length effect does not act through this kind of error"
    else:
        e58 = "mixed"
    print(f"  reading: {e58}")

    with open(REPO / "results/length_mechanism.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "task_id", "task", "demo_frames", "theta0_setup", *TARGETS,
                    "mean_arm_drift", *[f"noise_success_s{s}" for s in SIGMAS], "noise_tolerance"])
        for i, k in enumerate(keys):
            w.writerow([k[0], table[k]["task_id"], k[1], round(length[i], 1), theta0[i],
                        *[round(target[t][i], 3) for t in TARGETS], round(mean_drift[i], 4),
                        *[succ[s][i] for s in SIGMAS], round(tol[i], 3)])


if __name__ == "__main__":
    main()
