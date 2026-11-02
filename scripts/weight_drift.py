"""How far did fine-tuning move θ0? Per-module weight drift of one or more checkpoints (CPU).

A cheap proxy for forgetting pressure: the cumulative update grows with the area under the LR
curve, which differs between a 1k and a 3k run even on the same data. Per checkpoint: which
tensors changed (should be the 97.5M trainable parameters only), and per module group the
relative L2 drift ||θ - θ0|| / ||θ0||. With several checkpoints, also the cosine between their
update vectors (does a longer run move further in the same direction, or somewhere else?).

Usage: python scripts/weight_drift.py THETA0_DIR LABEL=POLICY_DIR [LABEL=POLICY_DIR ...]
POLICY_DIR may be a full or a slim checkpoint (adaptive_vla.checkpoints).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from adaptive_vla import checkpoints as ck

base_dir, runs = Path(sys.argv[1]), [a.split("=", 1) for a in sys.argv[2:]]
base = load_file(str(base_dir / ck.MODEL))


def tuned_tensors(pdir: Path):
    if (pdir / ck.SLIM_INFO).exists():
        return ck.merge(base, load_file(str(pdir / ck.DELTA)))
    return load_file(str(pdir / ck.MODEL))


def group(key: str) -> str:
    parts = [p for p in key.split(".") if not re.fullmatch(r"\d+", p)]
    if "lm_expert" in parts:
        return "action expert (lm_expert)"
    if "vision_model" in parts or "connector" in parts:
        return "vision + connector"
    if "text_model" in parts or "lm_head" in parts:
        return "VLM text model"
    return ".".join(parts[1:3]) if parts[0] == "model" else ".".join(parts[:2])


deltas, report = {}, {}
for label, pdir in runs:
    tuned = tuned_tensors(Path(pdir))
    changed = ck.changed_tensors(base, tuned)
    by = defaultdict(lambda: [0, 0.0, 0.0])          # n params, ||d||^2, ||θ0||^2 over changed tensors
    for k, t in changed.items():
        d = t.float() - base[k].float()
        g = by[group(k)]
        g[0] += t.numel(); g[1] += float(d.pow(2).sum()); g[2] += float(base[k].float().pow(2).sum())
    n = sum(g[0] for g in by.values())
    tot = (sum(g[1] for g in by.values()) / max(sum(g[2] for g in by.values()), 1e-30)) ** 0.5
    report[label] = {"changed_params": n, "rel_l2_total": tot,
                     "groups": {k: {"params": g[0], "rel_l2": (g[1] / max(g[2], 1e-30)) ** 0.5} for k, g in by.items()}}
    deltas[label] = torch.cat([(changed[k].float() - base[k].float()).flatten() for k in sorted(changed)])
    print(f"\n== {label}: {n / 1e6:.1f}M parameters changed, relative L2 drift {tot:.4f}")
    for k, g in sorted(report[label]["groups"].items(), key=lambda kv: -kv[1]["params"]):
        print(f"   {k:40s} {g['params'] / 1e6:7.1f}M  rel L2 {g['rel_l2']:.4f}")

labels = list(deltas)
if len(labels) > 1:
    print("\n== cosine between update vectors (θ - θ0); ratio = ||Δ_row|| / ||Δ_col||")
    for a in labels:
        for b in labels:
            if a < b and deltas[a].numel() == deltas[b].numel():
                cos = float(torch.nn.functional.cosine_similarity(deltas[a], deltas[b], dim=0))
                print(f"   {a} vs {b}: cos {cos:.3f}, norm ratio {deltas[a].norm() / deltas[b].norm():.2f}")
print(json.dumps(report))
