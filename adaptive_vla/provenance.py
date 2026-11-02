"""run_info.json: provenance written next to every run's outputs (torch-free).

One writer for eval.sh, train.sh, lang_diag.sh and throughput_sweep.sh (CLI:
scripts/write_run_info.py). Compute nodes have no git binary, so HEAD is resolved from .git
directly and repo_dirty is None there. Without a .git directory (a source download), both are
None. scripts/collect_results.py reads these keys; keep the names stable.
"""
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .config import LEROBOT_COMMIT, Config


def head_commit(git_dir=".git") -> Optional[str]:
    """Commit hash of HEAD (branch or detached), read from loose refs or packed-refs; None
    without a .git/HEAD."""
    g = Path(git_dir)
    if not (g / "HEAD").is_file():
        return None
    ref = (g / "HEAD").read_text().strip()
    if not ref.startswith("ref: "):
        return ref
    name = ref[5:]
    if (g / name).exists():
        return (g / name).read_text().strip()
    packed = g / "packed-refs"
    if packed.exists():
        for line in packed.read_text().splitlines():
            if line.endswith(" " + name):
                return line.split()[0]
    return None


def repo_dirty(git_dir=".git") -> Optional[bool]:
    if not shutil.which("git") or not Path(git_dir).is_dir():
        return None
    return bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())


def policy_source(policy: str) -> Optional[str]:
    """A materialized (rebuilt-from-slim) policy names the checkpoint it was rebuilt from."""
    p = Path(policy) / "materialized_from.json"
    return json.loads(p.read_text())["source"] if p.exists() else None


def run_info(policy: str, start: int, end: int, command: str, git_dir=".git", **extra) -> dict:
    c = Config()
    return {
        "repo_commit": head_commit(git_dir),
        "repo_dirty": repo_dirty(git_dir),   # None without git (compute nodes) or without .git
        "lerobot_commit": LEROBOT_COMMIT,
        "policy": policy,
        "policy_source": policy_source(policy),
        "theta0_revision": c.pretrained_revision,
        "vlm_revision": c.vlm_revision,
        "libero_assets_revision": c.libero_assets_revision,
        "dataset_revision": c.dataset_revision,
        **extra,
        "wall_clock_s": int(end) - int(start),
        "command": command,
    }


def write_run_info(out_dir, policy: str, start: int, end: int, command: str, **extra) -> dict:
    info = run_info(policy, start, end, command, **extra)
    with open(Path(out_dir) / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)
    return info
