"""Pre-download every asset the runs need, at pinned revisions, and set up LIBERO's config.

Run once on a machine with internet access:
    AVLA_ONLINE=1 source scripts/env.sh && python scripts/download_assets.py

Every later shell and job runs with HF_HUB_OFFLINE=1 (scripts/env.sh sets it).
Idempotent: snapshot_download skips files already in the cache.
"""
import importlib.util
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_vla.config import Config  # noqa: E402

cfg = Config()

# (repo_id, repo_type, revision). The VLM is fetched under the exact name in θ0's
# config.json (`vlm_model_name`): the Hub redirects it to SmolVLM2-500M-Video-Instruct,
# but the offline cache is keyed by the requested id.
HUB_ASSETS = [
    (cfg.pretrained, "model", cfg.pretrained_revision),
    ("HuggingFaceTB/SmolVLM2-500M-Instruct", "model", cfg.vlm_revision),
    (cfg.dataset_repo, "dataset", cfg.dataset_revision),
]

# hf-libero looks for sim assets here and otherwise downloads them on first env build,
# which fails offline (libero/libero/utils/download_utils.py).
LIBERO_ASSETS_DIR = os.path.expanduser("~/.cache/libero/assets")


def write_libero_config() -> str:
    """Write LIBERO's config.yaml so `import libero.libero` never calls input().

    Paths mirror libero.libero.get_default_path_dict() (package-relative), computed
    without importing the package, because the import itself is what prompts.
    """
    cfg_dir = os.environ.get("LIBERO_CONFIG_PATH")
    if not cfg_dir:
        sys.exit("LIBERO_CONFIG_PATH unset: source scripts/env.sh first")
    root = os.path.join(importlib.util.find_spec("libero").submodule_search_locations[0], "libero")
    paths = {
        "benchmark_root": root,
        "bddl_files": os.path.join(root, "bddl_files"),
        "init_states": os.path.join(root, "init_files"),
        "datasets": os.path.join(root, "..", "datasets"),
        "assets": LIBERO_ASSETS_DIR,
    }
    for key in ("bddl_files", "init_states"):
        assert os.path.isdir(paths[key]), paths[key]
    os.makedirs(cfg_dir, exist_ok=True)
    path = os.path.join(cfg_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(paths, f)
    return path


if os.environ.get("HF_HUB_OFFLINE") == "1":
    sys.exit("HF_HUB_OFFLINE=1: run with AVLA_ONLINE=1 on a machine with internet access")

def pin_main_ref(snapshot_path: str, rev: str) -> None:
    """Point the cache's refs/main at our pinned commit.

    Downloading by commit hash writes no refs/main, but LeRobot loads some repos by name
    with no revision (SmolVLA: AutoModelForImageTextToText.from_pretrained(vlm_model_name)),
    and offline that resolves through refs/main. Pinning it here makes "main" mean our
    pinned revision offline. (A later online call by name would move it to the real main.)
    """
    ref = Path(snapshot_path).parents[1] / "refs" / "main"
    ref.parent.mkdir(exist_ok=True)
    ref.write_text(rev)


for repo_id, repo_type, rev in HUB_ASSETS:
    path = snapshot_download(repo_id, repo_type=repo_type, revision=rev)
    pin_main_ref(path, rev)
    print(f"[ok] {repo_type:7} {repo_id}@{rev[:10]} -> {path} (refs/main pinned)")

snapshot_download("lerobot/libero-assets", repo_type="dataset",
                  revision=cfg.libero_assets_revision, local_dir=LIBERO_ASSETS_DIR)
print(f"[ok] libero assets@{cfg.libero_assets_revision[:10]} -> {LIBERO_ASSETS_DIR}")

print(f"[ok] libero config -> {write_libero_config()}")
