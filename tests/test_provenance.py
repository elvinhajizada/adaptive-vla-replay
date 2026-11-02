import json
import subprocess
import sys
from pathlib import Path

from adaptive_vla import provenance as pv

SHA = "0123456789abcdef0123456789abcdef01234567"
REPO = Path(__file__).resolve().parents[1]


def _git(tmp_path, head, loose=None, packed=None):
    g = tmp_path / ".git"
    g.mkdir()
    (g / "HEAD").write_text(head + "\n")
    if loose:
        (g / "refs" / "heads").mkdir(parents=True)
        (g / "refs" / "heads" / "main").write_text(loose + "\n")
    if packed:
        (g / "packed-refs").write_text(f"# pack-refs with: peeled fully-peeled sorted\n{packed} refs/heads/main\n")
    return g


def test_head_detached(tmp_path):
    assert pv.head_commit(_git(tmp_path, SHA)) == SHA


def test_head_branch_loose(tmp_path):
    assert pv.head_commit(_git(tmp_path, "ref: refs/heads/main", loose=SHA)) == SHA


def test_head_branch_packed(tmp_path):
    # the old train.sh copy returned "ref: refs/heads/main" here
    assert pv.head_commit(_git(tmp_path, "ref: refs/heads/main", packed=SHA)) == SHA


def test_head_missing_ref(tmp_path):
    assert pv.head_commit(_git(tmp_path, "ref: refs/heads/main")) is None


def test_no_git_dir(tmp_path):
    # a source download (ZIP) has no .git: provenance is unknown, not an error
    assert pv.head_commit(tmp_path / ".git") is None
    assert pv.repo_dirty(tmp_path / ".git") is None
    ri = pv.run_info("/theta0", 0, 1, "cmd", git_dir=tmp_path / ".git")
    assert ri["repo_commit"] is None and ri["repo_dirty"] is None


def test_policy_source(tmp_path):
    assert pv.policy_source(str(tmp_path)) is None
    (tmp_path / "materialized_from.json").write_text(json.dumps({"source": "/runs/train/1/x"}))
    assert pv.policy_source(str(tmp_path)) == "/runs/train/1/x"


def test_cli_train_fields(tmp_path):
    """The keys and types scripts/collect_results.py reads from a training run."""
    subprocess.run([sys.executable, str(REPO / "scripts" / "write_run_info.py"), str(tmp_path), "/theta0",
                    "100", "160", "lerobot-train --x=1", "tasks=[13]", "steps=1000", "batch_size=64",
                    "peak_lr=1e-5", "loss_dims=all"], check=True, cwd=REPO)
    ri = json.loads((tmp_path / "run_info.json").read_text())
    assert json.loads(ri["tasks"]) == [13]
    assert ri["steps"] == 1000 and ri["batch_size"] == 64
    assert ri["peak_lr"] == "1e-5" and ri["wall_clock_s"] == 60
    assert ri["command"] == "lerobot-train --x=1" and ri["policy_source"] is None
    for k in ("repo_commit", "lerobot_commit", "theta0_revision", "dataset_revision"):
        assert k in ri
