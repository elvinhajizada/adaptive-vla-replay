"""Write slim checkpoints at θ0 + alpha * (fine-tune - θ0) for each alpha (adaptive_vla/checkpoints.py).

Usage: python scripts/interpolate_checkpoint.py SLIM_POLICY_DIR OUT_ROOT ALPHA [ALPHA ...]
Writes OUT_ROOT/a<alpha>/ for each alpha; evaluate with scripts/probe_eval.sbatch (it materializes).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adaptive_vla import checkpoints as ck  # noqa: E402

src, root, *alphas = sys.argv[1:]
for a in alphas:
    print("interpolated ->", ck.interpolate(src, os.path.join(root, f"a{float(a):g}"), float(a)))
