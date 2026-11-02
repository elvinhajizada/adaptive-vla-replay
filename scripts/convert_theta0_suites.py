"""Build the M2 replay dataset: LIBERO-90 plus θ0's four training suites, in θ0's conventions.

The replay buffer draws from earlier new tasks *and* θ0's own training data
(Spatial/Object/Goal/10). The nvidia copies of those suites hold the same demos as θ0's
training set (docs/LIBERO.md, section 5) and get the same conversion as LIBERO-90
(`libero_data.convert`). lerobot-train reads one dataset, so the converted suites and the
converted LIBERO-90 are merged with the pinned LeRobot's `aggregate_datasets`, and θ0's normalizer
stats are written back into the merged stats.json (aggregation recomputes them).

Outputs under $AVLA_DSS/data:
  <suite>_theta0/      per-suite conversion (parquet rewritten, videos symlinked into the HF cache)
  libero_all_theta0/   merged dataset (videos copied, not re-encoded) + meta/avla_sources.json,
                       the episode range of each source in the merged index

Verify (always runs): per-source episode and frame counts, θ0's stats, gripper in {-1, 1} on
every frame, arm actions / state / frame indices equal to the per-suite source for one episode
per source, and decoded camera frames equal to the source's for the same frames.
Run in a CPU job: scripts/convert_theta0_suites.sbatch.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_vla import libero_data as ld  # noqa: E402
from adaptive_vla.config import Config  # noqa: E402

cfg = Config()
HUB = Path(os.environ["HF_HOME"]) / "hub"
NVIDIA = HUB / "datasets--nvidia--LIBERO_LeRobot_v3" / "snapshots" / cfg.dataset_revision
THETA0 = HUB / "models--HuggingFaceVLA--smolvla_libero" / "snapshots" / cfg.pretrained_revision
DATA = Path(os.environ["AVLA_DSS"]) / "data"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]   # θ0's training suites
SOURCES = {"libero_90": DATA / "libero90_theta0", **{s: DATA / f"{s}_theta0" for s in SUITES}}
DST = DATA / "libero_all_theta0"


def convert_suites(overwrite: bool) -> None:
    for s in SUITES:
        counts = ld.convert(str(NVIDIA / s), str(SOURCES[s]), str(THETA0), overwrite=overwrite)
        print(f"[convert] {s}: {counts}", flush=True)


def _same_shapes(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        (a[k]["dtype"], list(a[k]["shape"])) == (b[k]["dtype"], list(b[k]["shape"])) for k in a)


def aligned_roots() -> dict:
    """Source roots whose info.json features equal libero_spatial's, as aggregate_datasets requires.

    The sources differ only in metadata: nvidia labels LIBERO-90's rotation dims "roll, pitch,
    yaw" where the suites say "axis_angle1..3", and some suites carry per-feature "fps" keys.
    The values are axis-angle in all of them (components reach 4.8 and -4.4, outside any
    roll/pitch/yaw range, with no ±π wraps; checked 2026-10-25, job 5688247). The suites'
    roots are ours and get the canonical features written in. LIBERO-90's root fed M1's reported
    runs and is not edited: it is merged through a shadow root (symlinks plus an aligned info.json).
    """
    canon = json.loads((SOURCES["libero_spatial"] / "meta" / "info.json").read_text())["features"]
    roots = {}
    for n, root in SOURCES.items():
        info = json.loads((root / "meta" / "info.json").read_text())
        if info["features"] == canon:
            roots[n] = root
            continue
        assert _same_shapes(info["features"], canon), f"{n}: features differ beyond labels"
        info["features"] = canon
        if n in SUITES:
            (root / "meta" / "info.json").write_text(json.dumps(info, indent=4))
            roots[n] = root
        else:
            shadow = DATA / "_merge_src" / root.name
            if shadow.exists():
                shutil.rmtree(shadow)
            (shadow / "meta").mkdir(parents=True)
            for sub in ("data", "videos"):
                os.symlink(root / sub, shadow / sub)
            for f in (root / "meta").iterdir():
                if f.name != "info.json":
                    os.symlink(f, shadow / "meta" / f.name)
            (shadow / "meta" / "info.json").write_text(json.dumps(info, indent=4))
            roots[n] = shadow
        print(f"[align] {n}: features relabelled to libero_spatial's (labels and fps keys only)", flush=True)
    return roots


def merge(overwrite: bool) -> None:
    from lerobot.datasets.aggregate import aggregate_datasets

    if DST.exists():
        if not overwrite:
            raise FileExistsError(f"{DST} exists; pass --overwrite to rebuild")
        shutil.rmtree(DST)
    names = list(SOURCES)
    roots = aligned_roots()
    aggregate_datasets(
        repo_ids=[f"local/{n}_theta0" for n in names],
        aggr_repo_id="local/libero_all_theta0",
        roots=[roots[n] for n in names],
        aggr_root=DST,
        concatenate_videos=False,        # copy each source mp4 as is: no re-encode, no re-mux
    )
    # Aggregation recomputes stats.json from the sources' episode stats. θ0's normalizer must win
    # (docs/LIBERO.md, section 5, step 4), or lerobot-train re-normalises θ0 at every stage.
    theta0 = ld.load_theta0_stats(str(THETA0))
    stats = json.loads((DST / "meta" / "stats.json").read_text())
    for key in ("action", "observation.state"):
        for field in ("mean", "std"):
            stats[key][field] = np.asarray(theta0[f"{key}.{field}"], dtype=float).tolist()
    (DST / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    ranges, start = {}, 0
    for n in names:
        info = json.loads((SOURCES[n] / "meta" / "info.json").read_text())
        ranges[n] = {"episodes": [start, start + info["total_episodes"]], "frames": info["total_frames"]}
        start += info["total_episodes"]
    (DST / "meta" / "avla_sources.json").write_text(json.dumps({
        "sources": ranges,
        "note": "episode ranges [from, to) in the merged index, in aggregation order",
        "nvidia_revision": cfg.dataset_revision,
        "theta0_revision": cfg.pretrained_revision,
    }, indent=2))
    print(f"[merge] {DST}: {ranges}", flush=True)


def _rows(root: Path, episode: int, columns):
    """Parquet rows of one episode, ordered by frame_index."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    for f in sorted((root / "data").rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index"] + columns)
        t = t.filter(pc.equal(t["episode_index"], episode))
        if t.num_rows:
            return t.sort_by("frame_index").to_pydict()
    raise KeyError(f"episode {episode} not in {root}")


def verify() -> None:
    import pyarrow.parquet as pq
    from lerobot.datasets import LeRobotDataset

    theta0 = ld.load_theta0_stats(str(THETA0))
    info = json.loads((DST / "meta" / "info.json").read_text())
    src = json.loads((DST / "meta" / "avla_sources.json").read_text())["sources"]
    assert info["total_episodes"] == max(r["episodes"][1] for r in src.values()), info["total_episodes"]
    assert info["total_frames"] == sum(r["frames"] for r in src.values()), info["total_frames"]

    grip = []
    for f in sorted((DST / "data").rglob("*.parquet")):
        a = pq.read_table(f, columns=["action"])["action"].to_numpy(zero_copy_only=False)
        grip.append(np.stack(a)[:, ld.GRIPPER])
    grip = np.concatenate(grip)
    assert len(grip) == info["total_frames"], (len(grip), info["total_frames"])
    assert np.isin(grip, (-1.0, 1.0)).all(), np.unique(grip)

    probes = {n: (r["episodes"][0] + r["episodes"][1]) // 2 for n, r in src.items()}
    ds = LeRobotDataset("local/libero_all_theta0", root=DST, episodes=sorted(probes.values()))
    for k in ("action", "observation.state"):
        for f in ("mean", "std"):
            np.testing.assert_allclose(np.asarray(ds.meta.stats[k][f]).ravel(), theta0[f"{k}.{f}"], rtol=1e-6)

    cams = ("observation.images.image", ld.DST_WRIST)
    for n, ep in probes.items():
        src_ep = ep - src[n]["episodes"][0]
        a = _rows(DST, ep, ["action", "observation.state", "task_index"])
        b = _rows(SOURCES[n], src_ep, ["action", "observation.state", "task_index"])
        assert a["frame_index"] == b["frame_index"], n
        for col in ("action", "observation.state"):
            np.testing.assert_array_equal(np.asarray(a[col]), np.asarray(b[col]), err_msg=f"{n} {col}")
        ref = LeRobotDataset(f"local/{n}_theta0", root=SOURCES[n], episodes=[src_ep])
        ep_meta = ds.meta.episodes[ep]
        lo = ep_meta["dataset_from_index"]
        rel = ds.absolute_to_relative_idx
        for j in (0, len(b["frame_index"]) // 2, len(b["frame_index"]) - 1):
            x = ds[rel[lo + j] if rel is not None else lo + j]
            y = ref[j]
            for cam in cams:
                assert tuple(x[cam].shape) == (3, 256, 256), (n, cam, x[cam].shape)
                np.testing.assert_array_equal(x[cam].numpy(), y[cam].numpy(), err_msg=f"{n} {cam} frame {j}")
            assert x["task"] == y["task"], (n, x["task"], y["task"])
        print(f"[verify] {n}: episode {ep} (source {src_ep}) matches the source in actions, state, "
              f"task and decoded frames", flush=True)
    print(f"[verify] ok: {info['total_episodes']} episodes, {info['total_frames']} frames, "
          f"{info['total_tasks']} tasks; gripper in {{-1,1}} on every frame (close fraction "
          f"{np.mean(grip > 0):.2f}); stats == θ0 normalizer", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if not args.verify_only:
        convert_suites(args.overwrite)
        merge(args.overwrite)
    verify()
