#!/bin/bash
# Build the project venv on a CPU-only machine (tested on Ubuntu under WSL2). Needs internet and uv.
# Same Python (3.13.7) and the same pinned versions as requirements-lock.txt, with the CUDA
# wheels swapped for CPU ones, so code tested here matches what runs on the GPU machine.
# Idempotent: re-running reuses .venv and only installs what is missing.
# System packages first (Ubuntu): git build-essential cmake libosmesa6-dev libgl1 libglfw3
# Usage: bash scripts/build_env_cpu.sh 2>&1 | tee build_env_cpu.log
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/.venv"
PYVER=3.13.7

command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
uv python install "$PYVER"
if [ ! -x "$VENV/bin/python" ]; then
    uv venv --seed --python "$PYVER" "$VENV"   # --seed: include pip, used below like build_env.sh
fi

set +u; source "$VENV/bin/activate"; set -u
python -c "import sys; assert sys.version_info[:3] == (3, 13, 7), sys.version; print(sys.executable)"

# CPU requirement set derived from the cluster lock (not committed): drop the CUDA runtime
# wheels, take torch/torchvision from the CPU index, torchcodec from PyPI (CPU build).
REQ="$(mktemp --suffix=-requirements-cpu.txt)"
trap 'rm -f "$REQ"' EXIT
sed -E \
    -e '/^(nvidia-|cuda-|triton==)/d' \
    -e 's#download\.pytorch\.org/whl/cu126#download.pytorch.org/whl/cpu#' \
    -e 's/^(torch|torchvision)==([^+]+)\+cu126$/\1==\2+cpu/' \
    -e 's/^torchcodec==([^+]+)\+cu126$/torchcodec==\1/' \
    "$REPO/requirements-lock.txt" > "$REQ"
if grep -qE 'cu126|^(nvidia-|cuda-)' "$REQ"; then
    echo "CUDA pins left in the derived requirements:"; grep -nE 'cu126|^(nvidia-|cuda-)' "$REQ"; exit 1
fi

python -m pip install --upgrade pip
pip install --no-cache-dir -r "$REQ"
pip check
python - <<'EOF'
import importlib.metadata as md
for p in ["lerobot", "torch", "torchvision", "transformers", "numpy", "mujoco", "robosuite", "hf-libero", "peft"]:
    print(f"{p:14} {md.version(p)}")
import torch
assert not torch.cuda.is_available() and torch.version.cuda is None, "expected a CPU-only torch"
EOF
echo "BUILD OK"
