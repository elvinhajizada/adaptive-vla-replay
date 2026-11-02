"""Continual-learning metrics computed from a success matrix.

S is a (k, k) array. S[i, j] = success rate on task j measured after the policy
has finished training stage i (i.e. through task i). Only the lower triangle
(i >= j) is valid; the upper triangle is NaN.

`zeroshot` is an optional length-k vector: success on each task BEFORE any
task-specific training, used for forward-learning.

All functions are pure NumPy and run on CPU. No torch.
"""
from typing import Callable, Dict, Optional, Tuple
import numpy as np


def final_avg_success(S: np.ndarray) -> float:
    """Mean success over all tasks, measured at the final stage (last row)."""
    k = S.shape[0]
    return float(np.mean(S[k - 1, :k]))


def backward_transfer(S: np.ndarray) -> float:
    """Mean over earlier tasks j of (success at the end) - (success right after learning j).

    Negative means forgetting; closer to 0 (or positive) is better.
    Returns NaN for a single-task run.
    """
    k = S.shape[0]
    if k < 2:
        return float("nan")
    vals = [S[k - 1, j] - S[j, j] for j in range(k - 1)]
    return float(np.mean(vals))


def forgetting(S: np.ndarray) -> float:
    """Mean over earlier tasks j of (best success ever seen on j) - (final success on j).

    Standard CL forgetting metric. Non-negative; lower is better.
    """
    k = S.shape[0]
    if k < 2:
        return float("nan")
    vals = []
    for j in range(k - 1):
        col = S[j:k, j]                      # stages i = j .. k-1
        vals.append(float(np.nanmax(col)) - S[k - 1, j])
    return float(np.mean(vals))


def forward_learning(S: np.ndarray, zeroshot: Optional[np.ndarray]) -> float:
    """Mean over tasks j of (success right after learning j) - (zero-shot success on j).

    How well each new task is learned, above the pretrained zero-shot baseline.
    Requires the zeroshot vector; returns NaN otherwise.
    """
    if zeroshot is None:
        return float("nan")
    k = S.shape[0]
    diag = np.array([S[j, j] for j in range(k)])
    return float(np.mean(diag - np.asarray(zeroshot, dtype=float)[:k]))


def nbt(S: np.ndarray) -> float:
    """Negative backward transfer, LIBERO's forgetting metric (Liu et al. 2023, used by Liu et al. 2026).

    NBT_j = mean over later stages i > j of (S[j, j] - S[i, j]); NBT = mean of NBT_j over tasks
    j = 0 .. k-2 (the last task has no later stage). Lower is better. Difference from LIBERO:
    S[j, j] is the stage-end success, not the best success seen during task j's training.
    """
    k = S.shape[0]
    if k < 2:
        return float("nan")
    return float(np.mean([np.mean(S[j, j] - S[j + 1:k, j]) for j in range(k - 1)]))


def normalized_forgetting(S: np.ndarray) -> float:
    """Mean over earlier tasks j of (S[j, j] - S[T, j]) / S[j, j], T the last stage.

    The share of what was learned that is lost by the end, so tasks learned to different levels
    weigh equally. Tasks with S[j, j] == 0 learned nothing and are skipped (NaN if all are).
    """
    k = S.shape[0]
    vals = [(S[j, j] - S[k - 1, j]) / S[j, j] for j in range(k - 1) if S[j, j] > 0]
    return float(np.mean(vals)) if vals else float("nan")


def probe_normalized_forgetting(theta0: float, stage: float) -> float:
    """(θ0 - stage) / θ0 for an old-skill probe. NaN when θ0 scores 0 on it."""
    return float((theta0 - stage) / theta0) if theta0 > 0 else float("nan")


def gap_recovered(method: float, sequential: float, ceiling: float) -> float:
    """(method - sequential) / (ceiling - sequential), the share of the forgetting gap a method closes.

    The ceiling is full replay (the joint-training upper bound from the pilot), not θ0.
    NaN when the ceiling does not exceed Sequential (there is no gap to recover).
    """
    gap = ceiling - sequential
    return float((method - sequential) / gap) if gap > 0 else float("nan")


def success_matrix(outcomes: np.ndarray) -> np.ndarray:
    """Success matrix from per-episode outcomes.

    outcomes is a (k, k) object array; each valid cell holds a 1-d array of 0/1 episode
    successes, invalid cells (upper triangle, or not evaluated) hold None.
    """
    k = outcomes.shape[0]
    S = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            if outcomes[i, j] is not None:
                S[i, j] = float(np.mean(outcomes[i, j]))
    return S


def bootstrap_ci(metric: Callable[[np.ndarray], float], outcomes: np.ndarray, n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Point estimate and percentile bootstrap interval of metric(success_matrix(outcomes)).

    Each cell's episodes are resampled independently with replacement (episode noise only; seed
    variance is reported separately, across seeds). Returns (point, lo, hi); NaN resamples are
    dropped (e.g. normalized forgetting when a resampled diagonal is 0).
    """
    rng = np.random.default_rng(seed)
    point = metric(success_matrix(outcomes))
    k = outcomes.shape[0]
    vals = []
    for _ in range(n_boot):
        S = np.full((k, k), np.nan)
        for i in range(k):
            for j in range(k):
                ep = outcomes[i, j]
                if ep is not None:
                    S[i, j] = float(np.mean(rng.choice(ep, size=len(ep), replace=True)))
        v = metric(S)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def summarize(S: np.ndarray, zeroshot: Optional[np.ndarray] = None) -> Dict[str, float]:
    return {
        "final_avg_success": final_avg_success(S),
        "backward_transfer": backward_transfer(S),
        "forgetting": forgetting(S),
        "nbt": nbt(S),
        "normalized_forgetting": normalized_forgetting(S),
        "forward_learning": forward_learning(S, zeroshot),
    }


def format_matrix(S: np.ndarray, zeroshot: Optional[np.ndarray] = None) -> str:
    k = S.shape[0]
    lines = []
    header = "stage\\task " + " ".join(f"  t{j}" for j in range(k))
    lines.append(header)
    if zeroshot is not None:
        lines.append("zero-shot  " + " ".join(f"{zeroshot[j]:.2f}" for j in range(k)))
    for i in range(k):
        cells = []
        for j in range(k):
            cells.append(f"{S[i, j]:.2f}" if not np.isnan(S[i, j]) else "  . ")
        lines.append(f"after t{i}   " + " ".join(cells))
    return "\n".join(lines)
