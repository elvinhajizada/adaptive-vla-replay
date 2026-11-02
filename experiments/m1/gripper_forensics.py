"""Gripper forensics: how a task-23 fine-tune erases θ0's grasping, and why (E26). CPU or GPU.

Task 23's demos never close the gripper, and after fine-tuning on them θ0's predicted gripper on
old-task frames moves by 0.42 std (E20). Everything here is teacher-forced on fixed demo frames
with fixed flow-matching noise per frame, as in offline_action_error.py:

  measure LABEL=DIR ... [--hooks] [--skip-theta0]
      Predicted first-10-action chunks per grasp phase (θ0 included unless --skip-theta0). With
      --hooks also the expert's residual stream at several depths (for probes) and its attention
      mass per prefix segment.
  patch FT_LABEL=DIR --part=I/N
      Weight patching: θ0 plus one parameter group of the fine-tune's change (inject) and the
      fine-tune minus it (revert), on the pre-grasp and hold frames. --part splits the variants
      across jobs.
  grad NAME=TASK_INDEX ...
      First-order attribution at θ0: how much the arm part and the gripper part of a new task's
      loss gradient would raise the flow loss on old pre-grasp and hold frames (SGD and
      Adam-preconditioned), in total and per parameter group.
  leakfree LABEL=DIR ... [--skip-theta0]
      E33: the expert's residual stream at the first denoising step (t = 1, where x_t is the
      frame's fixed noise for every checkpoint, so no model output can leak in) and at the last,
      on approach, near (10 steps before the close) and pre-grasp frames of 10 demos per task.
  report
      Reads the saved outputs and writes results/gripper_*.csv; with the E33 and E34 outputs also
      gripper_probes_leakfree.csv and gripper_drift_direction.csv.
A DIR of the form random:DIR stands for a random perturbation with the same per-tensor norms as
DIR's change (E34's "generic perturbation" control).

Frames: LIBERO-Spatial and LIBERO-Goal demos that grasp (3 per task), one frame per phase, found
from the demo's own gripper commands (θ0 convention, +1 close): approach = 30 steps before the
first close, pre-grasp = 5 before it (the 10-action window holds the close), hold = 20 after it
(window all closed), pre-release = 5 before the first re-open. Patching and gradients use the
pre-grasp and hold frames of the first 2 demos per task. Checkpoint dirs must be slim
(adaptive_vla.checkpoints): their changed tensors are overlaid on one loaded θ0, which is exact
because the fine-tunes change nothing else (configs and normalizer stats verified, E18).
Outputs: $AVLA_RUNS/forensics/. AVLA_FORENSICS_SMOKE=1 uses 2 tasks per suite, 1 demo each.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from safetensors.torch import load_file

from adaptive_vla import checkpoints as ck
from adaptive_vla.config import Config
from adaptive_vla.libero_data import load_theta0_stats

IMG, WRIST, STATE = "observation.images.image", "observation.images.image2", "observation.state"
H, CHUNK, N_IMG = 10, 50, 64
PHASES = ("approach", "pregrasp", "hold", "prerelease")
DEPTHS = (0, 4, 8, 12, 16, 20, 24, 28)        # residual stream entering these expert layers
SEGS = ("agentview", "wrist", "language", "state", "actions")
SMOKE = os.environ.get("AVLA_FORENSICS_SMOKE") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
c = Config()
HUB = os.environ["HF_HOME"] + "/hub"
THETA0 = f"{HUB}/models--HuggingFaceVLA--smolvla_libero/snapshots/{c.pretrained_revision}"
NV = f"{HUB}/datasets--nvidia--LIBERO_LeRobot_v3/snapshots/{c.dataset_revision}"
CONV = os.environ["AVLA_DSS"] + "/data/libero90_theta0"
OUT = Path(os.environ["AVLA_RUNS"]) / ("forensics-smoke" if SMOKE else "forensics")
RESULTS = OUT if SMOKE else Path(__file__).resolve().parents[2] / "results"
st = {k: torch.as_tensor(v).float() for k, v in load_theta0_stats(THETA0).items()}
a_mean, a_std = st["action.mean"], st["action.std"]


# ----------------------------------------------------------------------------- frames

def _episodes(ds):
    eps = ds.meta.episodes
    return [(eps[i]["tasks"][0], eps[i]["dataset_from_index"], eps[i]["length"]) for i in range(len(eps))]


def _actions(ds, start, n, nvidia):
    rows = ds.hf_dataset.select(range(start, start + n))["action"]
    a = torch.stack([torch.as_tensor(r) for r in rows]).float().numpy().copy()
    if nvidia:
        a[:, 6] = 1.0 - 2.0 * a[:, 6]
    return a


def _chunk(a, f):
    """Demo actions f..f+49 (padded with the last one) and the pad mask, as training sees them."""
    idx = np.arange(f, f + CHUNK)
    return a[np.minimum(idx, len(a) - 1)], idx >= len(a)


def _phase_frames(g, near=False):
    """Frame index per grasp phase for one demo's gripper commands g (+1 close), or {} if no grasp.
    near=True adds "near" = 10 steps before the first close (its 10-action window is all open)."""
    closed = g > 0
    if not closed.any():
        return {}
    c0 = int(np.argmax(closed))
    reopen = np.nonzero(~closed[c0:])[0]
    r = c0 + int(reopen[0]) if len(reopen) else None
    out = {}
    if c0 - 30 >= 0 and not closed[c0 - 30:c0 - 30 + H].any():
        out["approach"] = c0 - 30
    if near and c0 - 10 >= 0:
        out["near"] = c0 - 10
    if c0 - 5 >= 0 and not closed[c0 - 5]:
        out["pregrasp"] = c0 - 5
    f = c0 + 20
    if f + H <= (r if r is not None else len(g)) and closed[f:f + H].all():
        out["hold"] = f
    if r is not None and r - 5 >= c0 and closed[r - 5]:
        out["prerelease"] = r - 5
    return out


def _frame(ds, start, f, a, nvidia, **meta):
    s = ds[start + f]
    chunk, pad = _chunk(a, f)
    return {"obs": {IMG: s[IMG], WRIST: s["observation.images.wrist_image" if nvidia else WRIST],
                    STATE: s[STATE]},
            "chunk": torch.from_numpy(chunk), "pad": torch.from_numpy(pad), **meta}


def old_frames(n_demos=3, phases=PHASES, near=False):
    """Grasp-phase frames from Spatial and Goal demos (θ0's training suites)."""
    frames = []
    for suite in ("libero_spatial", "libero_goal"):
        ds = LeRobotDataset("local/x", root=f"{NV}/{suite}", video_backend="pyav")
        per_task = {}
        for task, start, n in _episodes(ds):
            per_task.setdefault(task, []).append((start, n))
        tasks = sorted(per_task)[:2] if SMOKE else sorted(per_task)
        for task in tasks:
            used = 0
            for start, n in per_task[task]:
                if used == (1 if SMOKE else n_demos):
                    break
                a = _actions(ds, start, n, True)
                ph = _phase_frames(a[:, 6], near)
                if "pregrasp" not in ph:
                    continue
                for phase, f in ((p, f) for p, f in ph.items() if p in phases):
                    frames.append(_frame(ds, start, f, a, True, suite=suite, task=task, demo=used,
                                         phase=phase, frame=f))
                used += 1
    for i, fr in enumerate(frames):
        fr["noise"] = torch.randn(CHUNK, 32, generator=torch.Generator().manual_seed(1000 + i))
    return frames


def patch_subset(frames):
    return [f for f in frames if f["phase"] in ("pregrasp", "hold") and f["demo"] < 2]


def new_task_frames(task_index, n_frames=48):
    """Random frames of one converted LIBERO-90 task, with chunks, for the gradient check."""
    ds = LeRobotDataset("local/x", root=CONV, video_backend="pyav")
    want = ds.meta.tasks.index[ds.meta.tasks["task_index"] == task_index][0]
    eps = [(s, n) for t, s, n in _episodes(ds) if t == want]
    rng = np.random.default_rng(1000)
    frames = []
    for i in range(8 if SMOKE else n_frames):
        start, n = eps[rng.integers(len(eps))]
        f = int(rng.integers(n))
        frames.append(_frame(ds, start, f, _actions(ds, start, n, False), False, task=want, phase="new",
                             frame=f))
    return frames


# ----------------------------------------------------------------------------- model

class Policy:
    """One θ0 policy; checkpoints are applied by overlaying their changed tensors."""

    def __init__(self):
        self.cfg = PreTrainedConfig.from_pretrained(THETA0)
        self.cfg.device = DEVICE
        self.pre, _ = make_pre_post_processors(self.cfg, pretrained_path=THETA0,
                                               preprocessor_overrides={"device_processor": {"device": DEVICE}})
        self.policy = SmolVLAPolicy.from_pretrained(THETA0, config=self.cfg).to(DEVICE).eval()
        self.params = dict(self.policy.named_parameters())
        self.theta0 = {}
        self.touched = set()

    def delta(self, pdir):
        """Fine-tune minus θ0, per changed tensor (random:DIR: random, same per-tensor norms)."""
        if pdir.startswith("random:"):
            gen = torch.Generator().manual_seed(1000)
            out = {}
            for k, d in sorted(self.delta(pdir[len("random:"):]).items()):
                r = torch.randn(d.shape, generator=gen).to(d)
                out[k] = r * (d.norm() / (r.norm() + 1e-20))
            return out
        p = Path(pdir)
        if not (p / ck.SLIM_INFO).exists():
            raise SystemExit(f"{pdir}: not a slim checkpoint")
        out = {}
        for k, v in load_file(str(p / ck.DELTA)).items():
            if k not in self.params:
                raise SystemExit(f"{pdir}: {k} is not a parameter of θ0")
            if k not in self.theta0:
                self.theta0[k] = self.params[k].detach().clone()
            out[k] = v.to(self.theta0[k].dtype).to(DEVICE) - self.theta0[k]
        return out

    @torch.no_grad()
    def set(self, delta):
        """Weights = θ0 + delta (delta: name -> tensor, any subset)."""
        for k in self.touched - set(delta):
            self.params[k].copy_(self.theta0[k])
        for k, d in delta.items():
            self.params[k].copy_(self.theta0[k] + d)
        self.touched = set(delta)

    def batch(self, frames):
        b = {k: torch.stack([f["obs"][k] for f in frames]) for k in (IMG, WRIST, STATE)}
        b["task"] = [f["task"] for f in frames]
        return self.pre(b)

    @torch.no_grad()
    def predict(self, frames, cap=None, bs=8):
        """First 10 predicted actions per frame, physical units (θ0 convention)."""
        out = []
        for i in range(0, len(frames), bs):
            fr = frames[i:i + bs]
            pb = self.batch(fr)
            if cap is not None:
                cap.start_batch(pb[OBS_LANGUAGE_TOKENS].shape[1])
            self.policy.reset()
            noise = torch.stack([f["noise"] for f in fr]).to(DEVICE)
            chunk = self.policy.predict_action_chunk(pb, noise=noise)[..., :7].float().cpu()
            out.append((chunk * a_std + a_mean)[:, :H])
            if cap is not None:
                cap.end_batch()
        return torch.cat(out)

    def flow_losses(self, frames, gen):
        """Per-entry flow-matching losses (B, 50, 7), pads zeroed, with fixed noise and time."""
        m = self.policy
        pb = self.batch(frames)
        pb["action"] = ((torch.stack([f["chunk"] for f in frames]) - a_mean) / a_std).to(DEVICE)
        pad = torch.stack([f["pad"] for f in frames]).to(DEVICE)
        images, img_masks = m.prepare_images(pb)
        state, actions = m.prepare_state(pb), m.prepare_action(pb)
        noise = torch.randn(actions.shape, generator=gen).to(DEVICE)
        time = (torch.distributions.Beta(1.5, 1.0).sample((len(frames),)) * 0.999 + 0.001).to(DEVICE)
        losses = m.model.forward(images, img_masks, pb[OBS_LANGUAGE_TOKENS], pb[OBS_LANGUAGE_ATTENTION_MASK],
                                 state, actions, noise, time)[:, :, :7]
        return losses * (~pad)[..., None]


class Capture:
    """Expert residual stream (first 10 action tokens; last and first denoising step) and attention mass."""

    def __init__(self, P):
        vwe = P.policy.model.vlm_with_expert
        self.n_layers = len(vwe.lm_expert.layers)
        self.layer, self.on, self.L = None, False, None
        self.h, self.h0, self.attn, self.n_steps = {}, {}, None, 0
        self.feats, self.feats0, self.masses = {}, {}, []
        for d in DEPTHS:
            vwe.lm_expert.layers[d].input_layernorm.register_forward_pre_hook(self._keep(f"L{d:02d}"))
        vwe.lm_expert.norm.register_forward_pre_hook(self._keep("L32"))
        P.policy.model.action_out_proj.register_forward_pre_hook(self._keep("out"))
        for name in ("forward_attn_layer", "forward_cross_attn_layer"):
            fn = getattr(vwe, name)
            setattr(vwe, name, self._tag(fn))
        self.orig_attn = vwe.eager_attention_forward
        vwe.eager_attention_forward = self._attn
        self.heads, self.kv_heads = vwe.num_attention_heads, vwe.num_key_value_heads

    def _keep(self, name):
        def hook(_mod, args):
            if self.on and args[0].shape[1] == CHUNK:
                self.h[name] = args[0][:, :H].detach().float().cpu()
                self.h0.setdefault(name, self.h[name])
                if name == "out":
                    self.n_steps += 1
        return hook

    def _tag(self, fn):
        def wrapped(model_layers, inputs_embeds, layer_idx, *a, **k):
            self.layer = layer_idx
            return fn(model_layers, inputs_embeds, layer_idx, *a, **k)
        return wrapped

    def _attn(self, attention_mask, batch_size, head_dim, q, k, v):
        out = self.orig_attn(attention_mask, batch_size, head_dim, q, k, v)
        if self.on and q.shape[1] == CHUNK:
            if self.attn is None:
                self.attn = torch.zeros(q.shape[0], self.n_layers, len(SEGS))
            with torch.no_grad():
                kk = k.repeat_interleave(self.heads // self.kv_heads, dim=2).float().transpose(1, 2)
                w = torch.matmul(q.float().transpose(1, 2)[:, :, :H], kk.transpose(2, 3)) * head_dim ** -0.5
                w = torch.where(attention_mask[:, None, :H, :], w, torch.finfo(w.dtype).min)
                p = torch.softmax(w, dim=-1).mean(dim=(1, 2)).cpu()      # (B, keys)
                P0 = 2 * N_IMG + self.L + 1
                if p.shape[1] not in (P0, P0 + CHUNK):
                    raise RuntimeError(f"prefix layout: {p.shape[1]} keys, expected {P0} (+{CHUNK})")
                bounds = [0, N_IMG, 2 * N_IMG, 2 * N_IMG + self.L, P0, p.shape[1]]
                seg = torch.stack([p[:, a:b].sum(1) for a, b in zip(bounds[:-1], bounds[1:])], dim=1)
                self.attn[:, self.layer] += seg
        return out

    def start_batch(self, lang_len):
        self.on, self.L, self.n_steps, self.attn = True, lang_len, 0, None
        self.h, self.h0 = {}, {}

    def end_batch(self):
        self.on = False
        for name, t in self.h.items():
            self.feats.setdefault(name, []).append(t)
            self.feats0.setdefault(name, []).append(self.h0[name])
        self.masses.append(self.attn / self.n_steps)

    def result(self):
        return {"feats": {k: torch.cat(v) for k, v in self.feats.items()},
                "feats_t1": {k: torch.cat(v) for k, v in self.feats0.items()}, "attn": torch.cat(self.masses)}


def meta(frames):
    return [{k: f[k] for k in ("suite", "task", "demo", "phase", "frame") if k in f}
            | {"demo_grip": f["chunk"][:H, 6].tolist()} for f in frames]


def parse_models(args):
    return [a.split("=", 1) for a in args if "=" in a and not a.startswith("--")]


# ----------------------------------------------------------------------------- subcommands

def cmd_measure(args):
    hooks = "--hooks" in args
    frames = old_frames()
    print(f"{len(frames)} frames: " + ", ".join(f"{p} {sum(f['phase'] == p for f in frames)}" for p in PHASES),
          flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames.json").write_text(json.dumps(meta(frames)))
    P = Policy()
    cap = Capture(P) if hooks else None
    models = ([] if "--skip-theta0" in args else [("theta0", None)]) + parse_models(args)
    for label, pdir in models:
        P.set({} if pdir is None else P.delta(pdir))
        if cap is not None:
            cap.feats, cap.feats0, cap.masses = {}, {}, []
        rec = {"label": label, "source": pdir, "pred": P.predict(frames, cap)}
        if cap is not None:
            rec.update(cap.result())
        torch.save(rec, OUT / f"measure-{label}.pt")
        print(f"done {label}", flush=True)
    print("MEASURE DONE")


def cmd_leakfree(args):
    frames = old_frames(n_demos=10, phases=("approach", "near", "pregrasp"), near=True)
    print(f"{len(frames)} frames: " + ", ".join(f"{p} {sum(f['phase'] == p for f in frames)}"
                                                for p in ("approach", "near", "pregrasp")), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leakfree-frames.json").write_text(json.dumps(
        [x | {"state": f["obs"][STATE].tolist()} for x, f in zip(meta(frames), frames)]))
    P = Policy()
    cap = Capture(P)
    models = ([] if "--skip-theta0" in args else [("theta0", None)]) + parse_models(args)
    for label, pdir in models:
        P.set({} if pdir is None else P.delta(pdir))
        cap.feats, cap.feats0, cap.masses = {}, {}, []
        pred = P.predict(frames, cap)
        r = cap.result()
        torch.save({"label": label, "source": pdir, "pred": pred,
                    "feats": {k: v.mean(1) for k, v in r["feats"].items()},          # mean of 10 tokens
                    "feats_t1": {k: v.mean(1) for k, v in r["feats_t1"].items()}},
                   OUT / f"leakfree-{label}.pt")
        print(f"done {label}", flush=True)
    print("LEAKFREE DONE")


def _layer(k):
    m = re.search(r"layers\.(\d+)\.", k)
    return int(m.group(1)) if m else None


GROUPS = {
    "readout": lambda k: "action_out_proj" in k,
    "action_in_time": lambda k: "action_in_proj" in k or "action_time_mlp" in k,
    "state_proj": lambda k: "state_proj" in k,
    "q_proj": lambda k: "q_proj" in k,
    "kv_self (even)": lambda k: ("k_proj" in k or "v_proj" in k) and _layer(k) % 2 == 0,
    "kv_cross (odd)": lambda k: ("k_proj" in k or "v_proj" in k) and _layer(k) % 2 == 1,
    "o_proj": lambda k: "o_proj" in k,
    "mlp": lambda k: ".mlp." in k,
    **{f"layers {a:02d}-{a + 3:02d}": (lambda k, a=a: _layer(k) is not None and a <= _layer(k) < a + 4)
       for a in range(0, 32, 4)},
}


def variants(delta):
    out = [("theta0", "-", {}), ("ft", "-", dict(delta))]
    grip_row = {}
    for k in ("model.action_out_proj.weight", "model.action_out_proj.bias"):
        d = torch.zeros_like(delta[k])
        d[6] = delta[k][6]
        grip_row[k] = d
    out.append(("readout gripper row", "inject", grip_row))
    out.append(("readout gripper row", "revert", {**delta, **{k: delta[k] - grip_row[k] for k in grip_row}}))
    for name, pred in GROUPS.items():
        keys = {k for k in delta if pred(k)}
        out.append((name, "inject", {k: delta[k] for k in keys}))
        out.append((name, "revert", {k: v for k, v in delta.items() if k not in keys}))
    return out


def cmd_patch(args):
    (label, pdir), = parse_models(args)
    part = next((a.split("=", 1)[1] for a in args if a.startswith("--part=")), "0/1")
    i, n = map(int, part.split("/"))
    frames = patch_subset(old_frames())
    print(f"{len(frames)} patch frames", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "patch-frames.json").write_text(json.dumps(meta(frames)))
    P = Policy()
    for j, (group, mode, d) in enumerate(variants(P.delta(pdir))):
        if j % n != i or (OUT / f"patch-{label}-{j:02d}.pt").exists():
            continue
        P.set(d)
        torch.save({"ft": label, "group": group, "mode": mode, "n_params": sum(t.numel() for t in d.values()),
                    "pred": P.predict(frames)}, OUT / f"patch-{label}-{j:02d}.pt")
        print(f"done {group} {mode}", flush=True)
    print("PATCH DONE")


def _flat_grad(loss, params, retain):
    g = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return [torch.zeros_like(p) if x is None else x.detach() for x, p in zip(g, params)]


def cmd_grad(args):
    P = Policy()
    ref = parse_models(["ref=" + a.split("=", 1)[1] for a in args if a.startswith("--ref=")])
    keys = sorted(P.delta(ref[0][1]))            # the trainable tensors a fine-tune changes
    P.set({})
    for name, p in P.policy.named_parameters():
        p.requires_grad_(name in keys)
    params = [P.params[k] for k in keys]
    old = patch_subset(old_frames())
    print(f"{len(old)} old frames, {sum(p.numel() for p in params) / 1e6:.1f}M parameters", flush=True)
    bs, rec = 8, {"keys": keys, "numel": [P.params[k].numel() for k in keys]}

    def accumulate(frames, parts, seed):
        """Sum over batches of the gradients of each loss part, plus squared total gradients."""
        gen = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)                  # flow times (Beta sampler uses the global RNG)
        acc = {p: [torch.zeros_like(x) for x in params] for p in parts}
        sq = [torch.zeros_like(x) for x in params]
        values = {p: 0.0 for p in parts}
        nb = 0
        for i in range(0, len(frames), bs):
            losses = P.flow_losses(frames[i:i + bs], gen)
            terms = {"arm": losses[..., :6].sum() / losses.numel(), "grip": losses[..., 6].sum() / losses.numel()}
            gs = {}
            for j, p in enumerate(parts):
                gs[p] = _flat_grad(terms[p], params, retain=j < len(parts) - 1)
                values[p] += float(terms[p])
                acc[p] = [a + g for a, g in zip(acc[p], gs[p])]
            sq = [s + sum(gs[p][q] for p in parts) ** 2 for q, s in enumerate(sq)]
            nb += 1
            print(f"  batch {nb}", flush=True)
        return {p: [a / nb for a in acc[p]] for p in parts}, [s / nb for s in sq], {p: values[p] / nb for p in parts}

    old_g, _, old_v = accumulate(old, ("arm", "grip"), 1)
    rec["old"] = {"loss": old_v, "grad": old_g}
    for name, idx in parse_models([a for a in args if not a.startswith("--")]):
        new_g, sq, new_v = accumulate(new_task_frames(int(idx)), ("arm", "grip"), 2)
        rec[name] = {"loss": new_v, "grad": new_g, "sq": sq}
        print(f"done {name}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(rec, OUT / "grad.pt")
    print("GRAD DONE")


# ----------------------------------------------------------------------------- report

def _auc(score, y):
    """ROC AUC via ranks (average ranks for ties)."""
    s, y = np.asarray(score, float), np.asarray(y, bool)
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        m = s == v
        ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def _ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda A: ((A - mu) / sd) @ w


def _write(name, header, rows):
    import csv
    with open(RESULTS / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote results/{name} ({len(rows)} rows)")


def _behaviour(pred, m, ref):
    """Per phase: gripper commands vs the demo's and vs θ0's, and arm change vs θ0."""
    g, demo = pred[..., 6].numpy(), np.array([x["demo_grip"] for x in m]) > 0
    d = ((pred - ref).abs() / a_std).numpy()
    rows = []
    for phase in PHASES + ("all",):
        sel = np.array([phase in ("all", x["phase"]) for x in m])
        if not sel.any():
            continue
        gs, ds_ = g[sel], demo[sel]
        rows.append([phase, int(sel.sum()),
                     round(float((gs[ds_] > 0).mean()), 3) if ds_.any() else "",     # close where demo closes
                     round(float((gs[~ds_] < 0).mean()), 3) if (~ds_).any() else "",  # open where demo opens
                     round(float(gs[:, 0].mean()), 3),
                     round(float(d[sel][..., 0:3].mean()), 3), round(float(d[sel][..., 3:6].mean()), 3),
                     round(float(d[sel][..., 6].mean()), 3)])
    return rows


def cmd_report(_args):
    m = json.loads((OUT / "frames.json").read_text())
    recs = {p.stem[len("measure-"):]: torch.load(p, weights_only=False) for p in sorted(OUT.glob("measure-*.pt"))}
    ref = recs["theta0"]["pred"]
    rows = []
    for label, r in recs.items():
        rows += [[label] + row for row in _behaviour(r["pred"], m, ref)]
    _write("gripper_behaviour.csv", ["model", "phase", "n_frames", "close_where_demo_closes",
                                     "open_where_demo_opens", "mean_grip_cmd_first", "vs_theta0_trans",
                                     "vs_theta0_rot", "vs_theta0_grip"], rows)

    # Probes: is "demo gripper closed at this action step" linearly readable from the expert?
    hooked = {k: r for k, r in recs.items() if "feats" in r}
    if "theta0" in hooked:
        y = (np.array([x["demo_grip"] for x in m]) > 0).reshape(-1)
        tasks = sorted({x["task"] for x in m})
        fold = np.repeat(np.array([tasks.index(x["task"]) % 5 for x in m]), H)
        rows, drift_rows = [], []
        for depth in [f"L{d:02d}" for d in DEPTHS] + ["L32", "out"]:
            X0 = hooked["theta0"]["feats"][depth].reshape(-1, hooked["theta0"]["feats"][depth].shape[-1]).numpy()
            for label, r in hooked.items():
                X = r["feats"][depth].reshape(X0.shape).numpy()
                fresh, transfer = np.zeros(len(y)), np.zeros(len(y))
                for k in range(5):
                    tr, te = fold != k, fold == k
                    fresh[te] = _ridge(X[tr], y[tr], 1e-2)(X[te])
                    transfer[te] = _ridge(X0[tr], y[tr], 1e-2)(X[te])
                rows.append([label, depth, round(_auc(fresh, y), 3), round(_auc(transfer, y), 3)])
                rel = np.linalg.norm(X - X0, axis=1) / (np.linalg.norm(X0, axis=1) + 1e-9)
                cos = (X * X0).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(X0, axis=1) + 1e-9)
                drift_rows.append([label, depth, round(float(rel.mean()), 4), round(float(cos.mean()), 4)])
        _write("gripper_probes.csv", ["model", "depth", "auc_own_probe", "auc_theta0_probe"], rows)
        _write("gripper_hidden_drift.csv", ["model", "depth", "rel_change_vs_theta0", "cosine_vs_theta0"],
               drift_rows)

        # Where the expert's first 10 action tokens look, per layer (mean over heads and steps).
        phase = np.array([x["phase"] for x in m])
        a_rows = []
        for label, r in hooked.items():
            for layer in range(r["attn"].shape[1]):
                a_rows.append([label, layer, "self" if layer % 2 == 0 else "cross"]
                              + [round(float(r["attn"][:, layer, s].mean()), 4) for s in range(len(SEGS))]
                              + [round(float(r["attn"][phase == "pregrasp", layer, 1].mean()), 4)])
        _write("gripper_attention.csv", ["model", "layer", "kind"] + [f"mass_{s}" for s in SEGS]
               + ["mass_wrist_pregrasp"], a_rows)

    # Weight patching.
    pm = json.loads((OUT / "patch-frames.json").read_text()) if (OUT / "patch-frames.json").exists() else None
    pr = [torch.load(p, weights_only=False) for p in sorted(OUT.glob("patch-*.pt"))]
    if pm and {"theta0", "ft"} <= {r["group"] for r in pr}:
        demo = np.array([x["demo_grip"] for x in pm]) > 0
        base = next(r["pred"] for r in pr if r["group"] == "theta0")
        ft = next(r["pred"] for r in pr if r["group"] == "ft")

        def stats(p):
            d = ((p - base).abs() / a_std).numpy()
            return (float((p[..., 6].numpy()[demo] > 0).mean()), float(d[..., 6].mean()),
                    float(d[..., 0:3].mean()), float(d[..., 3:6].mean()))
        ft_s = stats(ft)
        rows = []
        for r in pr:
            s = stats(r["pred"])
            share = s[1] / ft_s[1] if r["mode"] == "inject" else 1 - s[1] / ft_s[1] if r["mode"] == "revert" else ""
            rows.append([r["ft"], r["group"], r["mode"], r["n_params"], round(s[0], 3), round(s[1], 3),
                         round(share, 3) if share != "" else "", round(s[2], 3), round(s[3], 3)])
        _write("gripper_patching.csv", ["ft", "group", "mode", "n_params_patched", "close_where_demo_closes",
                                        "vs_theta0_grip", "share_of_ft_grip_change", "vs_theta0_trans",
                                        "vs_theta0_rot"], rows)

    # Gradient attribution at θ0.
    if (OUT / "grad.pt").exists():
        g = torch.load(OUT / "grad.pt", weights_only=False)
        keys = g["keys"]

        def dot(a, b, w=None):
            return float(sum(((x * y) if w is None else (x * y * z)).sum() for x, y, z in
                             zip(a, b, w if w is not None else a)))
        rows, grows = [], []
        for name in [k for k in g if k not in ("keys", "numel", "old")]:
            new = g[name]
            prec = [1.0 / (s.sqrt() + 1e-8) for s in new["sq"]]
            norm = {p: dot(new["grad"][p], new["grad"][p]) ** 0.5 for p in ("arm", "grip")}
            for target in ("grip", "arm"):
                o = g["old"]["grad"][target]
                on = dot(o, o) ** 0.5
                for part in ("arm", "grip"):
                    gp = new["grad"][part]
                    rows.append([name, f"old_{target}", part, round(new["loss"][part], 5), round(norm[part], 6),
                                 round(dot(o, gp) / (on * norm[part] + 1e-20), 4),
                                 f"{-dot(o, gp):.4e}", f"{-dot(o, gp, prec):.4e}"])
                    if target == "grip":
                        for gname, pred in GROUPS.items():
                            idx = [i for i, k in enumerate(keys) if pred(k)]
                            grows.append([name, part, gname,
                                          f"{-sum(float((o[i] * gp[i] * prec[i]).sum()) for i in idx):.4e}"])
        _write("gripper_gradients.csv", ["new_task", "old_objective", "new_loss_part", "new_loss_part_value",
                                         "grad_norm", "cosine_with_old_objective_grad",
                                         "sgd_first_order_change_per_unit_lr", "adam_first_order_change_per_unit_lr"],
               rows)
        _write("gripper_gradients_by_group.csv", ["new_task", "new_loss_part", "group",
                                                  "adam_first_order_change_of_old_grip_loss"], grows)
    report_leakfree()
    report_direction(recs, m)
    print("REPORT DONE")


def _grouped_cv(X, X0, y, groups, lam=1e-2):
    """5-fold by task: AUC of a probe fit on X itself, and of θ0's probe (fit on X0) applied to X."""
    fold = np.array([g % 5 for g in groups])
    fresh, transfer = np.zeros(len(y)), np.zeros(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        fresh[te] = _ridge(X[tr], y[tr], lam)(X[te])
        transfer[te] = _ridge(X0[tr], y[tr], lam)(X[te])
    return round(_auc(fresh, y), 3), round(_auc(transfer, y), 3)


def report_leakfree():
    """E33: is "the gripper closes within the next 10 actions" readable without output leakage?"""
    if not (OUT / "leakfree-theta0.pt").exists():
        return
    m = json.loads((OUT / "leakfree-frames.json").read_text())
    recs = {p.stem[len("leakfree-"):]: torch.load(p, weights_only=False) for p in sorted(OUT.glob("leakfree-*.pt"))}
    phase = np.array([x["phase"] for x in m])
    tasks = sorted({x["task"] for x in m})
    task = np.array([tasks.index(x["task"]) for x in m])
    state = np.array([x["state"] for x in m])
    rows = []
    for contrast, neg in (("approach_vs_pregrasp", "approach"), ("near_vs_pregrasp", "near")):
        sel = np.isin(phase, [neg, "pregrasp"])
        y = phase[sel] == "pregrasp"
        rows.append(["state_only", contrast, "-", "state", int(sel.sum()), *_grouped_cv(state[sel], state[sel], y, task[sel])])
        for step, key in (("t1", "feats_t1"), ("last", "feats")):
            for depth in recs["theta0"][key]:
                X0 = recs["theta0"][key][depth].numpy()[sel]
                for label, r in recs.items():
                    rows.append([label, contrast, step, depth, int(sel.sum()),
                                 *_grouped_cv(r[key][depth].numpy()[sel], X0, y, task[sel])])
        # Behaviour on the same frames: the predicted gripper (max over the window) as the score.
        for label, r in recs.items():
            g = r["pred"][..., 6].numpy()[sel]
            rows.append([label, contrast, "output", "predicted_gripper", int(sel.sum()), round(_auc(g.max(1), y), 3), ""])
    _write("gripper_probes_leakfree.csv", ["model", "contrast", "denoise_step", "depth", "n_frames",
                                           "auc_own_probe", "auc_theta0_probe"], rows)


# Dataset task index each measured checkpoint was fine-tuned on (E34's "own task").
TRAINED_ON = {"ft1k": 13, "ft2k": 13, "ft3k": 13, "s100": 13, "s200": 13, "s400": 13, "s700": 13,
              "lr1e-5": 13, "cf_arm": 13, "cf_grip": 13, "rand1k": 13, "ft26": 5, "ft18": 22, "ft27": 8,
              **{k: 13 for k in ("warm100", "sgd03_100", "sgd1_100", "sgd03_200", "sgd03_300", "sgd01_100",
                                 "sgd01_200", "sgd01_300")}}


def _task_mean_action(task_index):
    """Mean normalized arm action (6 dims) over all frames of one converted LIBERO-90 task."""
    ds = LeRobotDataset("local/x", root=CONV, video_backend="pyav")
    want = ds.meta.tasks.index[ds.meta.tasks["task_index"] == task_index][0]
    a = np.concatenate([_actions(ds, s, n, False) for t, s, n in _episodes(ds) if t == want])
    return ((torch.from_numpy(a.mean(0)) - a_mean) / a_std)[:6].numpy()


def report_direction(recs, m):
    """E34: does the change on old frames point toward the new task's mean action, and is it shared?"""
    if "theta0" not in recs or not any(k in TRAINED_ON for k in recs):
        return
    z0 = (recs["theta0"]["pred"] / a_std).numpy()[..., :6]                    # (N, 10, 6), normalized
    means = {t: _task_mean_action(t) for t in sorted(set(TRAINED_ON.values()))}
    rows = []
    for label, r in recs.items():
        if label not in TRAINED_ON:
            continue
        d = (r["pred"] / a_std).numpy()[..., :6] - z0
        D = d.reshape(len(d), -1)
        norm = np.linalg.norm(D, axis=1) + 1e-12
        shared = float(np.linalg.norm(D.mean(0)) ** 2 / (norm ** 2).mean())
        sv = np.linalg.svd(D, compute_uv=False)
        row = [label, TRAINED_ON[label], len(D), round(float(norm.mean() / np.sqrt(D.shape[1])), 4),
               round(shared, 3), round(float(sv[0] ** 2 / (sv ** 2).sum()), 3)]
        for t, mu in means.items():
            toward = (mu[None, None, :] - z0).reshape(len(d), -1)
            cos = (D * toward).sum(1) / (norm * np.linalg.norm(toward, axis=1) + 1e-12)
            row.append(round(float(cos.mean()), 3))
        rows.append(row)
    _write("gripper_drift_direction.csv", ["model", "trained_on_task_index", "n_frames", "rms_arm_change",
                                           "shared_share", "top1_share"]
           + [f"cos_toward_task{t}_mean" for t in means], rows)


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"measure": cmd_measure, "leakfree": cmd_leakfree, "patch": cmd_patch, "grad": cmd_grad,
     "report": cmd_report}[cmd](sys.argv[2:])
