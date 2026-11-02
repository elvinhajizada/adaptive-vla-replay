"""Check that every checkpoint of a lerobot-train run kept θ0's normalizer stats.

Usage: python scripts/check_normalizer.py RUN_DIR THETA0_DIR

Compares the action / observation.state mean+std in each checkpoint's pre- and
post-processor (normalizer, unnormalizer) against θ0's. Any difference means the
training run re-normalised θ0 (fake forgetting); exits non-zero in that case.
"""
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

KEYS = ("action.mean", "action.std", "observation.state.mean", "observation.state.std")


def stats_files(model_dir: Path):
    return sorted(model_dir.glob("policy_*processor_step_*_*normalizer_processor.safetensors"))


def main(run_dir: str, theta0_dir: str) -> int:
    ref = load_file(str(Path(theta0_dir) / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    ckpts = sorted(Path(run_dir).glob("checkpoints/*/pretrained_model"))
    if not ckpts:
        print(f"[normalizer] no checkpoints under {run_dir}")
        return 1
    bad = 0
    for ck in ckpts:
        files = stats_files(ck)
        if not files:
            print(f"[normalizer] {ck}: no normalizer files"); bad += 1; continue
        for f in files:
            got = load_file(str(f))
            for k in KEYS:
                if k not in got:
                    continue
                d = float(np.max(np.abs(got[k] - ref[k])))
                status = "ok" if d < 1e-6 else "MISMATCH"
                bad += status != "ok"
                print(f"[normalizer] {ck.parent.name:>8} {f.name.split('_step_')[0]:22} {k:24} max|diff|={d:.2e} {status}")
    print("[normalizer] PASS: theta0 stats kept" if not bad else f"[normalizer] FAIL: {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
