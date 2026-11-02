"""Task choice: demos per candidate task in the converted LIBERO-90 dataset.

Maps libero_90 env ids to instructions through LIBERO's own benchmark (never by integer id
into the dataset), then counts episodes and frames per instruction in meta/episodes.
Pool rule (docs/LIBERO.md, section 11): single-scene instructions only, 34 to 50 demos each.

Usage: python experiments/setup/candidate_demos.py 18 19 22 ...   (CPU, seconds)
"""
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from libero.libero.benchmark import get_benchmark_dict

ids = [int(a) for a in sys.argv[1:]]
suite = get_benchmark_dict()["libero_90"]()
lang = [suite.get_task(i).language for i in range(suite.get_num_tasks())]
n_scenes = Counter(lang)

root = Path(os.environ["AVLA_DSS"]) / "data/libero90_theta0"
ep = pq.read_table(root / "meta/episodes", columns=["episode_index", "tasks", "length"]).to_pydict()
lengths = {}
for tasks, n in zip(ep["tasks"], ep["length"]):
    assert len(tasks) == 1, tasks
    lengths.setdefault(tasks[0], []).append(n)
print(f"dataset: {len(ep['episode_index'])} episodes, {len(lengths)} instructions")

print(f"\n{'id':>3}  {'scenes':>6}  {'demos':>5}  {'frames':>6}  {'len mean':>8}  {'min-max':>8}  pool  name")
for i in ids:
    t = suite.get_task(i)
    ls = np.array(lengths.get(t.language, []))
    ok = n_scenes[t.language] == 1 and 34 <= len(ls) <= 50
    span = f"{ls.min()}-{ls.max()}" if len(ls) else "-"
    mean = f"{ls.mean():.0f}" if len(ls) else "-"
    print(f"{i:>3}  {n_scenes[t.language]:>6}  {len(ls):>5}  {ls.sum():>6}  {mean:>8}  {span:>8}  "
          f"{'ok' if ok else 'NO':>4}  {t.name}")
