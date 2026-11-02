"""Decode every video of a converted dataset end to end and report files that fail (CPU).

  python experiments/replay/video_integrity.py [DATASET_ROOT] [--workers 16]
  python experiments/replay/video_integrity.py --scan observation.images.image2/chunk-000/file-013.mp4
  python experiments/replay/video_integrity.py --rows 4889

E46's stage 1 (115803) died on an AV1 InvalidDataError in a dataloader worker; LeRobot does not log
the file. This decodes every packet of every mp4 under DATASET_ROOT/videos with PyAV (as the
loader does) and prints, per file, frames decoded and the first error with its packet time.
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import av


def check(path):
    n, err = 0, None
    try:
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            for pkt in c.demux(s):
                try:
                    for _ in pkt.decode():
                        n += 1
                except av.error.FFmpegError as e:
                    err = f"pts {pkt.pts} ({float(pkt.pts * s.time_base) if pkt.pts is not None else '?'} s): {e}"
                    break
    except av.error.FFmpegError as e:
        err = f"open: {e}"
    return str(path), n, err


def scan(root, rel):
    """Decode one file past errors; map failing packets to the episodes whose frames they hold."""
    import pyarrow.parquet as pq
    path = Path(root, "videos", rel)
    bad = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        n_pkt = 0
        for pkt in c.demux(s):
            if pkt.pts is None:
                continue
            n_pkt += 1
            try:
                pkt.decode()
            except av.error.FFmpegError:
                bad.append(float(pkt.pts * s.time_base))
    print(f"{rel}: {len(bad)} of {n_pkt} packets fail to decode", flush=True)
    if not bad:
        return
    print(f"  failing times (s): {bad[0]:.2f} to {bad[-1]:.2f}; first 10: {[round(t, 2) for t in bad[:10]]}")
    key = rel.split("/")[0]
    file_idx = int(Path(rel).stem.split("-")[1])
    col = f"videos/{key}"
    hits = []
    for f in sorted(Path(root, "meta", "episodes").rglob("*.parquet")):
        t = pq.read_table(f).to_pydict()
        for i, ep in enumerate(t["episode_index"]):
            if t[f"{col}/file_index"][i] != file_idx:
                continue
            lo, hi = t[f"{col}/from_timestamp"][i], t[f"{col}/to_timestamp"][i]
            k = sum(lo <= b < hi for b in bad)
            if k:
                hits.append((ep, t["tasks"][i][0], lo, hi, k))
    print(f"  {len(hits)} episodes hold failing packets:")
    for ep, task, lo, hi, k in hits:
        print(f"    episode {ep} ({lo:.2f} to {hi:.2f} s, {k} packets): {task}")


def rows(root, episode):
    """Load every row of one episode the way training does (LeRobotDataset, PyAV); print failing rows."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from adaptive_vla.lerobot_glue import open_dataset
    ds = open_dataset(str(root))
    e = ds.meta.episodes[episode]
    lo, hi = e["dataset_from_index"], e["dataset_to_index"]
    failing = []
    for i in range(lo, hi):
        try:
            ds[i]
        except Exception as err:          # noqa: BLE001 - any decode failure counts
            failing.append(i)
            print(f"  row {i} (frame {i - lo} of {hi - lo}): {type(err).__name__}", flush=True)
    print(f"episode {episode}: rows {lo} to {hi - 1}; {len(failing)} of {hi - lo} rows fail: {failing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?",
                    default=os.path.join(os.environ["AVLA_DSS"], "data", "libero_all_theta0"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--scan", default="", help="one file (relative to videos/): decode past errors, map to episodes")
    ap.add_argument("--rows", type=int, default=0, help="one episode: load every row through LeRobot, list failures")
    a = ap.parse_args()
    if a.scan:
        scan(a.root, a.scan)
        return
    if a.rows:
        rows(a.root, a.rows)
        return
    files = sorted(Path(a.root, "videos").rglob("*.mp4"))
    print(f"{len(files)} files", flush=True)
    bad = 0
    with Pool(a.workers) as p:
        for path, n, err in p.imap_unordered(check, files):
            print(f"{'BAD ' if err else 'ok  '}{n:8d} frames  {path}" + (f"  {err}" if err else ""), flush=True)
            bad += err is not None
    print(f"VIDEO INTEGRITY DONE: {bad} of {len(files)} files fail")


if __name__ == "__main__":
    main()
