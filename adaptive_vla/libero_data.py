"""Convert the nvidia LIBERO-90 split to θ0's conventions (docs/LIBERO.md, section 5).

nvidia/LIBERO_LeRobot_v3 holds the same demos as lerobot/libero (θ0's training set),
but with different conventions. Training on it unconverted fails silently, so this writes
a converted LeRobot v3 dataset:

  1. wrist camera key  observation.images.wrist_image -> observation.images.image2
  2. gripper action    nvidia 1 open / 0 close -> θ0 -1 open / +1 close: g' = 1 - 2g
  3. drop              observation.states.* (duplicates of observation.state)
  4. stats.json        action / observation.state mean+std = θ0's normalizer stats.
                       lerobot-train (pinned 9a6bb61, lerobot_train.py:533,556-558) overwrites
                       a pretrained policy's normalizer with dataset.meta.stats when
                       fine-tuning, so the dataset must carry θ0's stats, or every stage
                       re-normalises θ0 and old skills drift (fake forgetting).

Videos are not re-encoded: the converted root symlinks the source video dirs under the
new key. Parquet is rewritten (small). Select tasks at train time with --dataset.episodes.

The pure functions here (gripper, stats, column / feature mapping) are unit-tested on CPU in
tests/test_libero_data.py. convert() does file I/O; run it in a batch job, not on a shared login host.
"""
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SRC_WRIST = "observation.images.wrist_image"
DST_WRIST = "observation.images.image2"
DROP_PREFIX = "observation.states."
GRIPPER = 6                     # index of the gripper in the 7-d action


# --------------------------------------------------------------------------- pure helpers
def gripper_to_theta0(g: np.ndarray) -> np.ndarray:
    """nvidia gripper (1 open, 0 close) -> θ0 convention (-1 open, +1 close)."""
    g = np.asarray(g)
    if not np.isin(g, (0.0, 1.0)).all():
        bad = np.unique(g[~np.isin(g, (0.0, 1.0))])[:5]
        raise ValueError(f"nvidia gripper values must be in {{0, 1}}, got e.g. {bad}")
    out = 1.0 - 2.0 * g
    assert np.isin(out, (-1.0, 1.0)).all()
    return out.astype(g.dtype)


def map_key(key: str) -> Optional[str]:
    """New name for a feature key, or None if the feature is dropped."""
    if key.startswith(DROP_PREFIX):
        return None
    return DST_WRIST if key == SRC_WRIST else key


def map_column(col: str) -> Optional[str]:
    """New name for a flat column like 'videos/<key>/chunk_index' or 'stats/<key>/mean'."""
    if col == SRC_WRIST or col.startswith(DROP_PREFIX):
        return map_key(col)
    parts = col.split("/")
    if len(parts) >= 2 and parts[0] in ("videos", "stats"):
        new = map_key(parts[1])
        return None if new is None else "/".join([parts[0], new] + parts[2:])
    return col


def affine_gripper_stats(stats: Dict[str, list]) -> Dict[str, list]:
    """Push per-feature action stats through g' = 1 - 2g on the gripper dim only.

    For y = a + b x with b < 0: min' = a + b max, max' = a + b min, mean' = a + b mean,
    std' = |b| std. Fields: min, max, mean, std, count (no quantiles in this dataset).
    """
    a, b = 1.0, -2.0
    unknown = set(stats) - {"min", "max", "mean", "std", "count"}
    if unknown:
        raise ValueError(f"unhandled stat fields {unknown}: extend affine_gripper_stats")
    out = {k: list(np.asarray(v, dtype=float).tolist()) for k, v in stats.items()}
    lo, hi = stats["min"][GRIPPER], stats["max"][GRIPPER]
    out["min"][GRIPPER] = a + b * hi
    out["max"][GRIPPER] = a + b * lo
    out["mean"][GRIPPER] = a + b * stats["mean"][GRIPPER]
    out["std"][GRIPPER] = abs(b) * stats["std"][GRIPPER]
    return out


def convert_stats(src_stats: Dict[str, dict], theta0: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """Dataset stats.json for the converted root.

    Keys renamed/dropped, gripper stats transformed, then mean/std of action and
    observation.state replaced by θ0's normalizer values (the only stats θ0's
    MEAN_STD normalizer reads; VISUAL features are IDENTITY).
    """
    out = {}
    for key, s in src_stats.items():
        new = map_key(key)
        if new is None:
            continue
        out[new] = affine_gripper_stats(s) if key == "action" else s
    for key in ("action", "observation.state"):
        for field in ("mean", "std"):
            val = np.asarray(theta0[f"{key}.{field}"], dtype=float)
            assert len(val) == len(out[key][field]), (key, field)
            out[key][field] = val.tolist()
    return out


def convert_features(features: Dict[str, dict]) -> Dict[str, dict]:
    """info.json 'features' with keys renamed/dropped (order preserved)."""
    return {map_key(k): v for k, v in features.items() if map_key(k) is not None}


# --------------------------------------------------------------------------- file I/O
def load_theta0_stats(theta0_dir: str) -> Dict[str, np.ndarray]:
    """theta0's normalizer stats ({'action.mean': ..., ...}) from its preprocessor files."""
    from safetensors.numpy import load_file
    path = Path(theta0_dir) / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    stats = load_file(str(path))
    need = {"action.mean", "action.std", "observation.state.mean", "observation.state.std"}
    assert need <= set(stats), f"theta0 normalizer missing {need - set(stats)}"
    return stats


def _rewrite_parquet(src: Path, dst: Path, fix_action: bool) -> int:
    """Rename/drop columns (and fix the gripper if fix_action); keep schema metadata in sync."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(src)
    cols, names = [], []
    for name, col in zip(table.column_names, table.columns):
        new = map_column(name)
        if new is None:
            continue
        if fix_action and name == "action":
            arr = np.stack(col.to_numpy(zero_copy_only=False)).astype(np.float32)
            arr[:, GRIPPER] = gripper_to_theta0(arr[:, GRIPPER])
            col = pa.chunked_array([pa.FixedSizeListArray.from_arrays(pa.array(arr.ravel()), arr.shape[1])])
        cols.append(col)
        names.append(new)
    out = pa.Table.from_arrays(cols, names=names)

    meta = dict(table.schema.metadata or {})
    if b"huggingface" in meta:              # datasets' feature spec: keep it consistent
        hf = json.loads(meta[b"huggingface"])
        feats = hf.get("info", {}).get("features", {})
        hf["info"]["features"] = {map_column(k): v for k, v in feats.items() if map_column(k) is not None}
        meta[b"huggingface"] = json.dumps(hf).encode()
    out = out.replace_schema_metadata(meta)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst)
    return out.num_rows


def convert(src_root: str, dst_root: str, theta0_dir: str, overwrite: bool = False) -> Dict[str, int]:
    """Write the converted dataset at dst_root. Returns row counts for logging."""
    src, dst = Path(src_root), Path(dst_root)
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists; pass overwrite=True to rebuild")
        shutil.rmtree(dst)
    (dst / "meta").mkdir(parents=True)

    counts = {"data_rows": 0, "episode_rows": 0}
    for f in sorted((src / "data").rglob("*.parquet")):
        counts["data_rows"] += _rewrite_parquet(f, dst / f.relative_to(src), fix_action=True)
    for f in sorted((src / "meta" / "episodes").rglob("*.parquet")):
        counts["episode_rows"] += _rewrite_parquet(f, dst / f.relative_to(src), fix_action=False)
    # The per-episode action stats in meta/episodes still use nvidia's gripper convention.
    # They are only read when stats are re-aggregated (recording / dataset edits), which this
    # project never does; the training stats come from stats.json below.

    shutil.copy2(src / "meta" / "tasks.parquet", dst / "meta" / "tasks.parquet")

    info = json.loads((src / "meta" / "info.json").read_text())
    info["features"] = convert_features(info["features"])
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    src_stats = json.loads((src / "meta" / "stats.json").read_text())
    stats = convert_stats(src_stats, load_theta0_stats(theta0_dir))
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    (dst / "videos").mkdir()
    for vdir in (src / "videos").iterdir():
        os.symlink(vdir.resolve(), dst / "videos" / map_key(vdir.name))

    (dst / "CONVERSION.json").write_text(json.dumps({
        "source": str(src.resolve()),
        "theta0": str(Path(theta0_dir).resolve()),
        "changes": ["wrist key -> image2", "gripper g' = 1 - 2g", "drop observation.states.*",
                    "stats.json action/state mean+std := theta0 normalizer"],
        **counts,
    }, indent=2))
    return counts


def task_index_for_instruction(dst_root: str, instruction: str) -> int:
    """Dataset task_index of an instruction (meta/tasks.parquet: task string -> task_index)."""
    import pandas as pd
    df = pd.read_parquet(Path(dst_root) / "meta" / "tasks.parquet")
    tasks = df["task"] if "task" in df.columns else pd.Series(df.index, index=df.index)
    hits = df.loc[(tasks == instruction).to_numpy(), "task_index"].tolist()
    if len(hits) != 1:
        raise KeyError(f"instruction {instruction!r} matches {len(hits)} dataset tasks, expected 1")
    return int(hits[0])


def steps_for_passes(passes: float, frames: int, batch_size: int) -> int:
    """Training steps for `passes` passes over `frames` frames at `batch_size` (rounded up).

    Replay-run protocol (fixed 2026-10-22): each task trains for a fixed number of passes over its own
    demos, not a fixed step count, so step counts differ by task.
    """
    if passes <= 0 or frames <= 0 or batch_size <= 0:
        raise ValueError(f"need positive passes, frames, batch_size; got {passes}, {frames}, {batch_size}")
    return math.ceil(passes * frames / batch_size)


def frames_for_tasks(dst_root: str, task_indices: List[int]) -> int:
    """Number of frames (dataset rows) of the given task_index values."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    n = 0
    for f in sorted((Path(dst_root) / "data").rglob("*.parquet")):
        t = pq.read_table(f, columns=["task_index"])
        n += int(pc.sum(pc.is_in(t["task_index"], value_set=pa.array(task_indices))).as_py() or 0)
    return n


def episodes_for_tasks(dst_root: str, task_indices: List[int]) -> List[int]:
    """Episode indices of the given task_index values (for --dataset.episodes)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    eps = set()
    for f in sorted((Path(dst_root) / "data").rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "task_index"])
        mask = pc.is_in(t["task_index"], value_set=pa.array(task_indices))
        eps.update(t.filter(mask)["episode_index"].to_pylist())
    return sorted(eps)


def frame_groups(merged_root: str) -> Dict[str, List[range]]:
    """Frame ranges per task of the merged replay dataset (scripts/convert_theta0_suites.py).

    Keys are "<source>:<instruction>" (e.g. "libero_10:put both moka pots on the stove"), so an
    instruction shared by two sources stays two tasks; values are the [from, to) row ranges of the
    task's episodes. Sources come from meta/avla_sources.json.
    """
    import pyarrow.parquet as pq
    root = Path(merged_root)
    sources = json.loads((root / "meta" / "avla_sources.json").read_text())["sources"]
    groups: Dict[str, List[range]] = {}
    for f in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "tasks", "dataset_from_index", "dataset_to_index"]).to_pydict()
        for ep, tasks, lo, hi in zip(t["episode_index"], t["tasks"], t["dataset_from_index"], t["dataset_to_index"]):
            src = [n for n, r in sources.items() if r["episodes"][0] <= ep < r["episodes"][1]]
            assert len(src) == 1 and len(tasks) == 1, (ep, src, tasks)
            groups.setdefault(f"{src[0]}:{tasks[0]}", []).append(range(lo, hi))
    return groups
