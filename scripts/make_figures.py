"""Figures for docs/report.md and the README (fig5, fig10 to fig14, fig16), rebuilt from results/*.csv
only; fig16 is a schematic. PNG for the docs, SVG alongside.

Style: Okabe-Ito categorical slots blue / orange / aqua (colour-blind safe), one-hue blue ramp for
ordered conditions, 2 px lines, ringed markers >= 8 px, thin bars, hairline solid grid, text in ink
colours (never the series colour), a legend for two or more series plus selective direct labels, no
dual axes (two measures become two panels).

Usage (CPU, needs matplotlib from the venv), from the repo root:
    python scripts/make_figures.py [fig11 fig14 ...]    # only those; default all
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RES, OUT = Path("results"), Path("docs/figures")
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#1a1a1a", "#404040", "#808080", "#e5e5e5", "#b0b0b0"
BLUE, ORANGE, AQUA = "#0072b2", "#d55e00", "#009e73"   # Okabe-Ito (colour-blind safe)
RAMP = [matplotlib.colors.to_hex(c) for c in plt.cm.Blues(np.linspace(0.25, 1.0, 7))]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.labelcolor": INK2, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "svg.hashsalt": "adaptive-vla",   # fixed SVG ids, so reruns give identical files
})
MARK = dict(marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)


def read(name):
    with open(RES / name) as f:
        return list(csv.DictReader(f))


RUNS = read("runs.csv")


def ev(run, suite=None, task=None):
    rows = [r for r in RUNS if r["run"] == run and (suite is None or r["suite"] == suite)
            and (task is None or int(r["task_id"]) == task)]
    if not rows:
        raise KeyError((run, suite, task))
    k, n = sum(int(r["n_success"]) for r in rows), sum(int(r["n_episodes"]) for r in rows)
    return k, n


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight",
                    metadata={"Date": None} if ext == "svg" else None)   # reruns give identical files
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def pct_axis(ax, lo=0, hi=100):
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))


# --- Fig 5: onset, success and offline drift vs training step (two panels) ---------------------------
def fig5():
    off = {(r["label"], r["frame_set"]): r for r in read("offline_action_error.csv")}
    steps = [0, 100, 200, 400, 700, 1000]
    probe = [ev("113038-l10kitchen-r256", task=3)] + [ev(f"113822-s000{s:03d}-r256", "libero_10") for s in (100, 200, 400, 700)] \
        + [ev("113800-ft23-1k-r256")]
    new = [ev("113037-r256", task=23)] + [ev(f"113822-s000{s:03d}-r256", "libero_90") for s in (100, 200, 400, 700)] \
        + [ev("113795-r256")]
    labels = ["theta0", "s100", "s200", "s400", "s700", "ft1k"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    for series, c, name in ((probe, BLUE, "old skill: same-scene LIBERO-10 task"), (new, ORANGE, "new skill: task 23")):
        pv = [100 * k / n for k, n in series]
        a1.plot(steps, pv, color=c, label=name, **MARK)
    a1.annotate("84%", (0, 84), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    a1.annotate("75%", (700, 75), xytext=(-12, 0), textcoords="offset points", ha="right", va="center", color=INK, fontsize=9)
    pct_axis(a1, -3, 100)
    a1.set_xlabel("training step (task 23, LR 1e-4)")
    a1.set_ylabel("success")
    a1.set_title("Forgetting comes before learning")
    a1.legend(fontsize=8.5, loc="upper right")
    for dim, c, name in (("trans", BLUE, "translation"), ("rot", ORANGE, "rotation"), ("grip", AQUA, "gripper")):
        v = [0.0 if l == "theta0" else float(off[(l, "spatial")][f"vs_theta0_{dim}"]) for l in labels]
        a2.plot(steps, v, color=c, label=name, **MARK)
    lr = off[("lr1e-5", "spatial")]
    for dim, c in (("trans", BLUE), ("rot", ORANGE), ("grip", AQUA)):
        a2.plot(1170, float(lr[f"vs_theta0_{dim}"]), marker="D", ls="none", markersize=7, color=c,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    a2.plot([], [], marker="D", ls="none", markersize=7, color=MUTED, label="separate run at LR 1e-5, end")
    a2.axvline(1075, color=AXIS, lw=1)       # the diamonds are another run, not a later step
    a2.set_xlim(-40, 1250)
    a2.set_xticks([0, 200, 400, 600, 800, 1000, 1170], ["0", "200", "400", "600", "800", "1000", "LR\n1e-5"])
    a2.set_xlabel("training step")
    a2.set_ylabel("change vs θ0 (action std units)")
    a2.set_ylim(0, 0.7)
    a2.set_title("Offline action change on Spatial frames")
    a2.legend(fontsize=8.5, loc="upper right", ncols=2)
    fig.tight_layout()
    save(fig, "fig5_onset")


# --- schematics (drawn, not data) --------------------------------------------------------------------
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PALE = {"blue": "#e5f0f7", "orange": "#fae7dc", "aqua": "#dff1ea", "gray": "#f2f2f2"}


def box(ax, x, y, w, h, title, body, fill, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                                facecolor=fill, edgecolor=edge, linewidth=1.2))
    ax.text(x + 0.015, y + h - 0.03, title, ha="left", va="top", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(x + 0.015, y + h - 0.085, body, ha="left", va="top", fontsize=8.3, color=INK2, linespacing=1.35)


def arrow(ax, a, b, color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=color, linewidth=1.4,
                                 shrinkA=2, shrinkB=2))


def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# --- Replay figures (fig10 to fig14): the five-task sequence ---------------------------------------
# Conditions are replay budgets, an ordered quantity, so they take steps of the one-hue blue ramp
# (no replay in gray); probes are categories and take the blue / orange / aqua slots.
M2_TASKS = [26, 18, 23, 27, 34]
M2_SEQ = {   # budget -> one list of stage training jobs (stage 1 first) per seed
    "none": [[113804, 115603, 115604, 115605, 115606]],
    "0.2%": [[115608, 115620, 115621, 115622, 115623], list(range(115626, 115631)), list(range(115632, 115637))],
    "2%": [[115609, 115758, 115759, 115760, 115761], list(range(115786, 115791))],
    "100%": [[115611, 115792, 115793, 115794, 115795], list(range(115815, 115820))],
}
M2_K = {"none": "no replay", "0.2%": "k = 14 (0.2%)", "2%": "k = 140 (2%)", "100%": "every frame"}
M2_COL = {"none": MUTED, "0.2%": RAMP[2], "2%": RAMP[4], "100%": RAMP[6]}
SPATIAL_SUB, GOAL_SUB = [1, 2, 3, 7, 8], [1, 2, 4, 7, 8, 9]


def m2_cells():
    """(job, suite, task, final?) -> [successes, episodes, subgoal string] over the training-run evals."""
    cells = defaultdict(lambda: [0, 0, ""])
    for r in RUNS:
        p = r["policy"].split("/")
        if len(p) < 3 or p[0] != "train" or not p[1].isdigit():
            continue
        c = cells[(int(p[1]), r["suite"], int(r["task_id"]), "final" in r["run"])]
        c[0] += int(r["n_success"])
        c[1] += int(r["n_episodes"])
        c[2] = r["subgoals_final"]
    return cells


def pooled(cells, job, suite, tasks, final):
    ks = [cells[(job, suite, t, final)] for t in tasks]
    k, n = sum(c[0] for c in ks), sum(c[1] for c in ks)
    return (k, n) if n else None


def theta0_final():
    """θ0 on the final-probe protocol, every run of it pooled, counts per probe."""
    parts = {"libero_spatial": [ev("115785-theta0-final-r256", "libero_spatial")],
             "libero_goal": [ev("115785-theta0-final-r256", "libero_goal"), ev("113038-goal-r256", "libero_goal")],
             "libero_10": [ev(r, "libero_10", t) for r, t in (("115785-theta0-final-r256", 2), ("113038-l10kitchen-r256", 2),
                                                              ("113801-r256", 3), ("113038-l10kitchen-r256", 3),
                                                              ("115785-theta0-final-r256", 9), ("113038-l10kitchen-r256", 9))]}
    return {s: (sum(k for k, _ in v), sum(n for _, n in v)) for s, v in parts.items()}


def fig10():
    """Success matrices: rows = after stage i, columns = new task; seed means."""
    cells = m2_cells()
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (b, seeds) in zip(axes, M2_SEQ.items()):
        ax.grid(False)
        for i in range(5):
            for j in range(i + 1):
                v = np.mean([cells[(jobs[i], "libero_90", M2_TASKS[j], False)][0]
                             / cells[(jobs[i], "libero_90", M2_TASKS[j], False)][1] for jobs in seeds])
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=cmap(v), edgecolor=SURFACE, lw=2))
                ax.text(j + 0.5, i + 0.5, f"{100 * v:.0f}", ha="center", va="center", fontsize=9,
                        color="white" if v > 0.45 else INK, fontweight="bold" if i == j else "normal")
        ax.set_xlim(0, 5)
        ax.set_ylim(5, 0)
        ax.set_xticks([j + 0.5 for j in range(5)], [str(t) for t in M2_TASKS], fontsize=9)
        ax.set_yticks([i + 0.5 for i in range(5)],
                      [f"after stage {i + 1}" if ax is axes[0] else "" for i in range(5)], fontsize=9)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.xaxis.tick_top()
        n = len(seeds)
        ax.set_title(f"{M2_K[b]}" + (f", mean of {n} seeds" if n > 1 else ", 1 seed"), fontsize=10, pad=24)
    fig.suptitle("Success on each new task (%) after each stage; diagonal = just learned (x 50), the rest x 20",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.08)
    save(fig, "fig10_m2_matrices")


def fig11():
    """Retention (% of θ0) after five new tasks vs replay budget, one panel per probe (final probes)."""
    cells = m2_cells()
    t0 = theta0_final()
    probes = [("libero_spatial", "LIBERO-Spatial\n(10 tasks x 20 episodes", list(range(10))),
              ("libero_goal", "LIBERO-Goal\n(10 tasks x 20 episodes", list(range(10))),
              ("libero_10", "LIBERO-10 tasks 2, 3, 9\n(3 tasks x 50 episodes", [2, 3, 9])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, (suite, name, tasks) in zip(axes, probes):
        k0, n0 = t0[suite]
        ax.axhline(1, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(-0.35, 1.01, "θ0", color=INK2, fontsize=9, va="bottom")
        means = []
        for x, (b, seeds) in enumerate(M2_SEQ.items()):
            vals = []
            for jobs in seeds:
                c = pooled(cells, jobs[-1], suite, tasks, True)
                vals.append((c[0] / c[1]) / (k0 / n0))
            ax.scatter([x] * len(vals), vals, s=22, color=M2_COL[b], edgecolor=INK2, lw=0.6, zorder=3)
            m = np.mean(vals)
            means.append(m)
            ax.plot([x - 0.22, x + 0.22], [m, m], color=M2_COL[b], lw=3, solid_capstyle="butt", zorder=4)
            ax.annotate(f"{m:.2f}", (x + 0.26, m), va="center", fontsize=8.5, color=INK, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none"))
        ax.set_xticks(range(4), [M2_K[b].replace(" (", "\n(") for b in M2_SEQ], fontsize=8.5)
        ax.set_xlim(-0.4, 3.7)
        ax.set_title(f"{name}; θ0 {100 * k0 / n0:.1f}%)", fontsize=10)
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
    axes[0].set_ylabel("retention (final success / θ0's)")
    fig.text(0.01, -0.08, "After five new tasks. Dots: seeds (no replay 1, k = 14 3, k = 140 2, every frame 2); "
             "bars and labels: retention of the seed mean,\nagainst θ0 on the same protocol. "
             "No replay was probed with fewer episodes (10 per Spatial and Goal task, 20 per LIBERO-10 task).", color=MUTED, fontsize=8.5)
    save(fig, "fig11_m2_retention")


def fig12():
    """θ0-skill subsets after each stage, per budget, as retention against θ0 on the same subsets (seeds as thin lines)."""
    cells = m2_cells()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (suite, tasks, title, refn) in zip(axes, [("libero_spatial", SPATIAL_SUB, "LIBERO-Spatial subset (5 x 10)", 100),
                                                     ("libero_goal", GOAL_SUB, "LIBERO-Goal subset (6 x 10)", 120)]):
        k0 = sum(ev(r, suite, t)[0] for r in ("115594-theta0-refA-r256", "115595-theta0-refB-r256") for t in tasks)
        r0 = k0 / refn
        ax.axhline(1, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(5.22, 1.01, f"θ0\n({100 * r0:.0f}%)", color=INK2, fontsize=8.5, va="bottom")
        ax.set_xlim(0.75, 5.45)
        for b, seeds in M2_SEQ.items():
            runs = []
            for jobs in seeds:
                ys = []
                for i, job in enumerate(jobs):
                    c = pooled(cells, job, suite, tasks, False) if b != "none" or i == 0 else \
                        (pooled(cells, job, suite, tasks, True) if i == 4 else None)
                    ys.append(np.nan if c is None else c[0] / c[1] / r0)
                runs.append(ys)
                if len(seeds) > 1:
                    ax.plot(range(1, 6), ys, color=M2_COL[b], lw=1, alpha=0.5)
            if b == "none":                      # probed after stages 1 and 5 only: markers, no line
                ax.plot([1, 5], [runs[0][0], runs[0][4]], color=M2_COL[b], lw=0, **MARK,
                        label=M2_K[b] + " (probed after stages 1 and 5)")
            else:
                m = np.mean(runs, axis=0)
                ax.plot(range(1, 6), m, color=M2_COL[b], **MARK, label=M2_K[b])
        ax.set_xticks(range(1, 6), [f"{s}\n(task {t})" for s, t in zip(range(1, 6), M2_TASKS)], fontsize=9)
        ax.set_xlabel("after stage")
        ax.set_title(title)
        ax.set_ylim(-0.04, 1.2)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
    axes[0].set_ylabel("retention (success / θ0's)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.28), ncol=4, fontsize=8.5)
    save(fig, "fig12_m2_stages")


def fig13():
    """Learning cost: success on the task just learned, per budget; and E44 buffer memorization."""
    cells = m2_cells()
    memo = read("buffer_memorization.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [3, 2]})
    for bi, (b, seeds) in enumerate(M2_SEQ.items()):
        off = (bi - 1.5) * 0.15
        vals = np.array([[100 * cells[(jobs[i], "libero_90", t, False)][0] / cells[(jobs[i], "libero_90", t, False)][1]
                          for i, t in enumerate(M2_TASKS)] for jobs in seeds])
        if len(seeds) > 1:
            for v in vals:
                a1.scatter(np.arange(5) + off, v, s=14, color=M2_COL[b], alpha=0.55, lw=0, zorder=3)
        a1.plot(np.arange(5) + off, vals.mean(0), color=M2_COL[b], lw=0, **MARK, zorder=4, label=M2_K[b])
    # second no-replay reference (E48a): M1 single-task fine-tunes from θ0 (task 26's is Sequential stage 1)
    m1 = {18: 113803, 23: 113795, 27: 113805, 34: 113806}
    xs = [i + (0 - 1.5) * 0.15 for i, t in enumerate(M2_TASKS) if t in m1]
    ys = [100 * cells[(m1[t], "libero_90", t, False)][0] / cells[(m1[t], "libero_90", t, False)][1]
          for t in M2_TASKS if t in m1]
    a1.scatter(xs, ys, s=60, facecolor=SURFACE, edgecolor=M2_COL["none"], lw=1.6, zorder=4,
               label="no replay, single task from θ0")
    a1.set_xticks(range(5), [f"task {t}" for t in M2_TASKS])
    a1.set_xlabel("stage (task learned)")
    a1.set_title("Success on the task just learned (x 50)")
    pct_axis(a1, 0, 103)
    a1.legend(loc="lower left", fontsize=8.5, ncol=2)
    groups = [("memo-k14-b0", "memo-k14-b1", "memo-k14-b2"), ("memo-k140-b0",)]
    for gi, (runs, b) in enumerate(zip(groups, ("0.2%", "2%"))):
        for si, fset in enumerate(("buffered", "unbuffered")):
            ratios = []
            for run in runs:
                rr = {(r["model"] == "theta0", r["frame_set"]): float(r["vs_demo_trans"]) for r in memo if r["run"] == run}
                ratios.append(rr[(False, fset)] / rr[(True, fset)])
            x = gi * 2.6 + si
            a2.bar(x, np.mean(ratios), width=0.7, color=M2_COL[b] if fset == "buffered" else SURFACE,
                   edgecolor=M2_COL[b], lw=1.5)
            a2.scatter([x] * len(ratios), ratios, s=12, color=INK2, zorder=3) if len(ratios) > 1 else None
            a2.text(x, np.mean(ratios) + 0.04, f"{np.mean(ratios):.2f}", ha="center", fontsize=9, color=INK)
    a2.axhline(1, color=INK2, lw=1, ls=(0, (4, 3)))
    a2.set_xticks([0, 1, 2.6, 3.6], ["in buffer", "not in\nbuffer", "in buffer", "not in\nbuffer"], fontsize=9)
    a2.text(0.5, 1.24, "k = 14 (0.2%), 3 seeds", ha="center", fontsize=9, color=INK2)
    a2.text(3.1, 1.24, "k = 140 (2%), 1 seed", ha="center", fontsize=9, color=INK2)
    a2.set_ylim(0, 1.35)
    a2.set_ylabel("error vs demo, ratio to θ0's")
    a2.set_title("θ0 frames: offline error after stage 5")
    save(fig, "fig13_m2_learning")


def fig14():
    """Failure anatomy of the two-sub-goal tasks: each sub-goal's share of episodes, by budget."""
    cells = m2_cells()
    SG = {("libero_90", 23): ("close white_cabinet_1_bottom_region", "open white_cabinet_1_top_region",
                              "New task 23:\nclose bottom drawer, open top"),
          ("libero_10", 2): ("turnon flat_stove_1", "on moka_pot_1 flat_stove_1_cook_region",
                             "LIBERO-10 task 2:\nstove on, moka pot on it"),
          ("libero_10", 3): ("in akita_black_bowl_1 white_cabinet_1_bottom_region", "close white_cabinet_1_bottom_region",
                             "LIBERO-10 task 3:\nbowl in drawer, close it"),
          ("libero_10", 9): ("in white_yellow_mug_1 microwave_1_heating_region", "close microwave_1",
                             "LIBERO-10 task 9:\nmug in microwave, close it")}

    def parse(s):
        return {k: float(v) for k, v in (p.rsplit("=", 1) for p in s.split(";") if p)}

    t0 = {("libero_10", 2): next(r for r in RUNS if r["run"] == "115785-theta0-final-r256" and r["task_id"] == "2"),
          ("libero_10", 9): next(r for r in RUNS if r["run"] == "115785-theta0-final-r256" and r["task_id"] == "9"),
          ("libero_10", 3): next(r for r in RUNS if r["run"] == "113801-r256" and r["suite"] == "libero_10")}
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), sharey=True)
    axes = axes.ravel()
    for ax, (key, (p1, p2, title)) in zip(axes, SG.items()):
        cols = []
        if key in t0:
            g = parse(t0[key]["subgoals_final"])
            cols.append(("θ0", g[p1], g[p2]))
        else:                                    # task 23: just-learned evals (stage 3) of the replay runs
            gs = [parse(cells[(jobs[2], "libero_90", 23, False)][2]) for b in ("0.2%", "2%", "100%") for jobs in M2_SEQ[b]]
            cols.append(("just\nlearned", np.mean([g[p1] for g in gs]), np.mean([g[p2] for g in gs])))
        for b in ("0.2%", "2%", "100%"):
            gs = [parse(cells[(jobs[-1], key[0], key[1], True)][2]) for jobs in M2_SEQ[b]]
            cols.append((M2_K[b].split(" (")[-1].rstrip(")").replace("every frame", "every\nframe"),
                         np.mean([g[p1] for g in gs]), np.mean([g[p2] for g in gs])))
        x = np.arange(len(cols))
        ax.bar(x - 0.18, [100 * c[1] for c in cols], width=0.34, color=ORANGE, label="first sub-goal holds at the end")
        ax.bar(x + 0.18, [100 * c[2] for c in cols], width=0.34, color=AQUA, label="second sub-goal holds at the end")
        for xi, c in zip(x, cols):                # mark zeros, which otherwise look like missing bars
            for dx, v in ((-0.18, c[1]), (0.18, c[2])):
                if v < 0.005:
                    ax.text(xi + dx, 1, "0", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x, [c[0] for c in cols], fontsize=9)
        ax.set_title(title, fontsize=9.5)
        pct_axis(ax, 0, 105)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    fig.tight_layout(h_pad=2.0)
    fig.text(0.01, -0.03, "Replay budgets at the final checkpoint (x 50; seed means). Sub-goals in the order the demos reach them.",
             color=MUTED, fontsize=8.5)
    save(fig, "fig14_m2_anatomy")


def fig16_protocol():
    """Schematic of the five-task sequence: buffer above each stage, probes after each stage and at the end."""
    fig, ax = blank((16, 5.4))
    ax.set_xlim(-0.01, 1.045)
    names = ["wine bottle in\nbottom drawer", "frying pan\non stove", "close bottom\ndrawer, open top",
             "wine bottle\non wine rack", "mug in front of\nthe white mug"]
    w, gap, x0 = 0.125, 0.02, 0.14
    xs = [x0 + i * (w + gap) for i in range(5)]
    by, bh, sy, sh, py, ph = 0.72, 0.20, 0.38, 0.26, 0.04, 0.20
    box(ax, 0.0, sy, w, sh, "θ0", "SmolVLA, 605M\nskills: 40 tasks\nof 4 LIBERO suites", PALE["blue"], BLUE)
    for i, (x, t, nm) in enumerate(zip(xs, M2_TASKS, names)):
        box(ax, x, sy, w, sh, f"stage {i + 1}", f"task {t}:\n{nm}", PALE["orange"], ORANGE)
        learned = ", ".join(str(u) for u in M2_TASKS[:i])
        box(ax, x, by, w, bh, f"buffer: {40 + i} tasks", "θ0's 40" + (f"\n+ {learned}" if learned else ""),
            PALE["gray"], AXIS)
        arrow(ax, (x + w / 2, by - 0.005), (x + w / 2, sy + sh + 0.01))
        arrow(ax, (x + w / 2, sy - 0.01), (x + w / 2, py + ph + 0.01))
    for a, b in zip([0.0] + xs[:-1], xs):
        arrow(ax, (a + w + 0.004, sy + sh / 2), (b - 0.004, sy + sh / 2))
    fx, fw = xs[-1] + w + gap, 0.17
    box(ax, fx, sy, fw, sh, "final probes", "Spatial, Goal: 10 tasks x 20\nLIBERO-10 2, 3, 9: x 50\n"
        "control: LIBERO-10\n0, 1, 4, 6, 7: x 20", PALE["aqua"], AQUA)
    arrow(ax, (xs[-1] + w + 0.004, sy + sh / 2), (fx - 0.004, sy + sh / 2))
    box(ax, x0, py, xs[-1] + w - x0, ph, "after every stage",
        "θ0's skills: per-stage subsets, 5 Spatial and 6 Goal tasks x 10 episodes\n"
        "new tasks: the one just learned x 50, the earlier ones x 20", PALE["aqua"], AQUA)
    ax.text(0.0, by + bh + 0.01, "batch: 32 new-task\nframes + 32 replay\n\nbuffer: k frames per\ntask, drawn once;\n"
            "k = 14 (0.2%), 140\n(2%) or every frame", fontsize=8, color=INK2, va="top", linespacing=1.3)
    save(fig, "fig16_m2_protocol")


FIGURES = (fig5, fig10, fig11, fig12, fig13, fig14, fig16_protocol)

if __name__ == "__main__":
    names = set(sys.argv[1:])
    for f in FIGURES:
        if not names or f.__name__ in names:
            f()
