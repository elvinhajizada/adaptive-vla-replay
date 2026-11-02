# Source me: `source scripts/env.sh` (interactive, salloc, and every sbatch script).
# Activates the venv and sets the storage layout and run-time env vars.
#
# Storage, all under $AVLA_DSS (default ~/avla-data; set AVLA_DSS before sourcing to move it):
#   hf_cache/  θ0, SmolVLM2, nvidia dataset (~12 GB)
#   data/      converted datasets (libero90_theta0; videos symlinked into hf_cache)
#   runs/      checkpoints, eval outputs, slurm logs
# LIBERO sim assets stay in ~/.cache/libero/assets (400 MB, path fixed by download_assets.py).
# Rendering: EGL where an NVIDIA GPU is visible, OSMesa (software) elsewhere; AVLA_GL=egl|osmesa
# overrides.

_AVLA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_AVLA_GL=""


export AVLA_DSS="${AVLA_DSS:-$HOME/avla-data}"
if [ -z "$_AVLA_GL" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then _AVLA_GL=egl; else _AVLA_GL=osmesa; fi
fi
export MUJOCO_GL="${AVLA_GL:-$_AVLA_GL}"
export PYOPENGL_PLATFORM="$MUJOCO_GL"

source "$_AVLA_REPO/.venv/bin/activate"

export HF_HOME="$AVLA_DSS/hf_cache"
export AVLA_RUNS="$AVLA_DSS/runs"
export LIBERO_CONFIG_PATH="$_AVLA_REPO/.libero"   # written by scripts/download_assets.py
export TOKENIZERS_PARALLELISM=false

# Compute nodes may lack internet: everything must come from the cache.
# Override with AVLA_ONLINE=1 where there is internet (downloads).
if [ "${AVLA_ONLINE:-0}" != "1" ]; then
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
fi

# Memory watchdog, once per batch job on a compute node (SLURMD_NODENAME is unset in a salloc
# shell outside a batch job). Slurm sites without cgroups do not enforce --mem, so the watchdog keeps
# the job inside it and kills the job rather than let it run a shared node out of memory
# (scripts/memguard.py).
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURMD_NODENAME:-}" ] && [ -z "${AVLA_MEMGUARD:-}" ] \
        && [ "${AVLA_MEMGUARD_OFF:-0}" != "1" ]; then
    export AVLA_MEMGUARD=$$
    # Started from a subshell, so it is not a job of the batch shell: a bare `wait` ignores it.
    ( python "$_AVLA_REPO/scripts/memguard.py" "$$" & )
fi

unset _AVLA_REPO _AVLA_GL
