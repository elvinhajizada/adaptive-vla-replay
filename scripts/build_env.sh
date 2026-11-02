#!/bin/bash
# Build the project venv on a GPU machine (CUDA 12.6 wheels). Needs internet: on a cluster, run it
# on the login node (compute nodes may not have it).
# Installs the exact versions in requirements-lock.txt. --relock instead installs requirements.txt
# (the loose pins) and rewrites requirements-lock.txt from the result; use it only to change the env.
# Idempotent: re-running reuses .venv and only installs what is missing.
# Needs Python 3.13 (python3.13 on PATH, or AVLA_PYTHON=/path/to/python3.13) and cmake (egl_probe
# builds a C extension). CPU-only machine: scripts/build_env_cpu.sh.
# Usage: bash scripts/build_env.sh [--relock] 2>&1 | tee build_env.log
set -euo pipefail

RELOCK=0
case "${1:-}" in
    --relock) RELOCK=1 ;;
    "") ;;
    *) echo "usage: bash scripts/build_env.sh [--relock]"; exit 1 ;;
esac

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO/.venv"
PYTHON="${AVLA_PYTHON:-python3.13}"
SITE_SHIM=""


if [ ! -x "$VENV/bin/python" ]; then
    "$PYTHON" -m venv "$VENV"
fi

# A site's activate shim (set above), appended once.
if [ -n "$SITE_SHIM" ] && ! grep -qF "$(echo "$SITE_SHIM" | sed -n 2p)" "$VENV/bin/activate"; then
    chmod u+w "$VENV/bin/activate"   # some module Pythons install the venv templates read-only
    echo "$SITE_SHIM" >> "$VENV/bin/activate"
fi

set +u; source "$VENV/bin/activate"; set -u
python -c "import sys, ctypes; assert sys.version_info[:2] == (3, 13), sys.version; print(sys.executable)"

python -m pip install --upgrade pip
# --no-cache-dir: the torch/CUDA wheels are several GB.
if [ "$RELOCK" = 1 ]; then
    pip install --no-cache-dir -r "$REPO/requirements.txt"
    { echo "--extra-index-url https://download.pytorch.org/whl/cu126"; pip freeze --exclude-editable; } > "$REPO/requirements-lock.txt"
    echo "rewrote requirements-lock.txt"
else
    pip install --no-cache-dir -r "$REPO/requirements-lock.txt"
fi
pip check
python - <<'EOF'
import importlib.metadata as md
for p in ["lerobot", "torch", "torchvision", "transformers", "numpy", "mujoco", "robosuite", "hf-libero", "peft"]:
    print(f"{p:14} {md.version(p)}")
EOF
echo "BUILD OK"
