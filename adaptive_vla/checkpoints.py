"""Slim checkpoints: store only the tensors a fine-tune changed relative to θ0.

θ0 fine-tunes update the action expert and state projection only (97.5M of 605M parameters;
vision and VLM frozen), yet lerobot-train saves the full 1.2 GB model plus 388 MB of optimizer
state. A slim checkpoint keeps the policy dir's small files (config, processors, tokenizer) and
replaces model.safetensors with trainable.safetensors: the tensors that differ from θ0's. The
full model is θ0 with those tensors overlaid, so slimming is lossless by construction, and
slim_checkpoint() proves it bit for bit before deleting anything.

Pure safetensors code (no LeRobot), unit-tested in tests/test_checkpoints.py.
"""
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file, save_file

MODEL = "model.safetensors"
DELTA = "trainable.safetensors"
SLIM_INFO = "slim.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def changed_tensors(base: Dict[str, torch.Tensor], tuned: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Tensors of `tuned` that differ from `base` (bitwise). Keys must match exactly."""
    if set(base) != set(tuned):
        raise ValueError(f"key sets differ: {sorted(set(base) ^ set(tuned))[:5]}")
    out = {}
    for k, t in tuned.items():
        b = base[k]
        if t.dtype != b.dtype or t.shape != b.shape:
            raise ValueError(f"{k}: dtype/shape differ ({b.dtype} {tuple(b.shape)} vs {t.dtype} {tuple(t.shape)})")
        if not torch.equal(t, b):
            out[k] = t
    return out


def merge(base: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    unknown = set(delta) - set(base)
    if unknown:
        raise ValueError(f"delta has keys not in base: {sorted(unknown)[:5]}")
    return {**base, **delta}


def slim_checkpoint(policy_dir: str, base_dir: str, drop_training_state: bool = True,
                    expected_params: Optional[int] = None) -> dict:
    """Replace policy_dir/model.safetensors by the delta vs base_dir/model.safetensors.

    Writes trainable.safetensors and slim.json, rebuilds the full model from base + delta and
    checks every tensor equals the original before deleting model.safetensors. With
    drop_training_state, also deletes the sibling training_state/ dir (optimizer, scheduler,
    RNG: only needed to resume, which this project never does). Idempotent.
    """
    pdir, bdir = Path(policy_dir), Path(base_dir)
    if (pdir / SLIM_INFO).exists():
        return json.loads((pdir / SLIM_INFO).read_text())
    base, tuned = load_file(str(bdir / MODEL)), load_file(str(pdir / MODEL))
    delta = changed_tensors(base, tuned)
    n_params = sum(t.numel() for t in delta.values())
    if expected_params is not None and n_params > expected_params:
        raise ValueError(f"{n_params} changed parameters > {expected_params} trainable: frozen weights moved?")
    save_file(delta, str(pdir / DELTA), metadata={"format": "pt"})

    rebuilt = merge(base, load_file(str(pdir / DELTA)))
    bad = [k for k in tuned if not torch.equal(rebuilt[k], tuned[k])]
    if bad:
        (pdir / DELTA).unlink()
        raise RuntimeError(f"rebuild mismatch on {len(bad)} tensors, e.g. {bad[:3]}; nothing deleted")

    info = {"base_dir": str(bdir.resolve()), "base_sha256": _sha256(bdir / MODEL),
            "original_sha256": _sha256(pdir / MODEL), "original_bytes": (pdir / MODEL).stat().st_size,
            "delta_bytes": (pdir / DELTA).stat().st_size, "n_tensors": len(delta), "n_params": n_params}
    (pdir / SLIM_INFO).write_text(json.dumps(info, indent=2))
    (pdir / MODEL).unlink()
    state = pdir.parent / "training_state"
    if drop_training_state and state.is_dir():
        shutil.rmtree(state)
    return info


def materialize(policy_dir: str, out_dir: str, base_dir: Optional[str] = None) -> Path:
    """Rebuild a loadable full policy dir (model.safetensors = base + delta) at out_dir."""
    pdir, odir = Path(policy_dir), Path(out_dir)
    info = json.loads((pdir / SLIM_INFO).read_text())
    bdir = Path(base_dir or info["base_dir"])
    if _sha256(bdir / MODEL) != info["base_sha256"]:
        raise ValueError(f"base {bdir} is not the model this checkpoint was slimmed against")
    odir.mkdir(parents=True, exist_ok=True)
    for f in pdir.iterdir():
        if f.name in (DELTA, SLIM_INFO):
            continue
        (shutil.copytree if f.is_dir() else shutil.copy2)(f, odir / f.name)
    save_file(merge(load_file(str(bdir / MODEL)), load_file(str(pdir / DELTA))), str(odir / MODEL),
              metadata={"format": "pt"})
    (odir / "materialized_from.json").write_text(json.dumps({"source": str(pdir.resolve())}))
    return odir


def interpolate(policy_dir: str, out_dir: str, alpha: float, base_dir: Optional[str] = None) -> Path:
    """Slim checkpoint at θ0 + alpha * (fine-tune - θ0), from a slim policy_dir (WiSE-FT / RETAIN-style).

    Only the stored (changed) tensors move, computed in float32 and cast back to their dtype.
    alpha = 0 gives θ0, alpha = 1 the fine-tune. Used for the drift-tolerance curve (E31) and as a
    weight-interpolation baseline. The output is an ordinary slim
    checkpoint (materialize() loads it); slim.json records the source and alpha.
    """
    pdir, odir = Path(policy_dir), Path(out_dir)
    info = json.loads((pdir / SLIM_INFO).read_text())
    bdir = Path(base_dir or info["base_dir"])
    if _sha256(bdir / MODEL) != info["base_sha256"]:
        raise ValueError(f"base {bdir} is not the model this checkpoint was slimmed against")
    base, delta = load_file(str(bdir / MODEL)), load_file(str(pdir / DELTA))
    mixed = {k: (base[k].float() + alpha * (t.float() - base[k].float())).to(t.dtype) for k, t in delta.items()}
    odir.mkdir(parents=True, exist_ok=True)
    for f in pdir.iterdir():
        if f.name in (DELTA, SLIM_INFO):
            continue
        (shutil.copytree if f.is_dir() else shutil.copy2)(f, odir / f.name, **({"dirs_exist_ok": True} if f.is_dir() else {}))
    save_file(mixed, str(odir / DELTA), metadata={"format": "pt"})
    (odir / SLIM_INFO).write_text(json.dumps(
        {**info, "interpolated_from": str(pdir.resolve()), "alpha": alpha,
         "delta_bytes": (odir / DELTA).stat().st_size, "original_sha256": None}, indent=2))
    return odir
