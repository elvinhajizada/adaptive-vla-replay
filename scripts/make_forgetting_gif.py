"""Side-by-side rollout GIFs from saved eval videos (docs/figures/*.gif).

Each panel is TITLE|VERDICT|VIDEO, VIDEO relative to $AVLA_RUNS/eval, VERDICT success or failure
(shown once the clip ends). With no arguments, rebuilds docs/figures/m1_forgetting.gif:
  θ0 on the scene-4 near probe (LIBERO-10 task 3)             -> success
  task-23 1k fine-tune on the same probe episode              -> failure (forgetting)
  task-23 1k fine-tune on task 23 itself                      -> success (new skill)
Usage (CPU; PyAV and Pillow from the venv):
  python scripts/make_forgetting_gif.py [--out docs/figures/X.gif] ["TITLE|VERDICT|VIDEO" ...]
"""
import argparse
import os
from pathlib import Path

import av
from PIL import Image, ImageDraw

E = Path(os.environ["AVLA_RUNS"]) / "eval"
DEFAULT = [
    "theta0: put the bowl in the drawer and close it|success|libero_10/113038-l10kitchen-r256/videos/libero_10_3/eval_episode_0.mp4",
    "after learning task 23: same task, same start|failure|libero_10/113800-ft23-1k-r256/videos/libero_10_3/eval_episode_0.mp4",
    "after learning task 23: task 23 (new skill)|success|libero_90/113795-r256/videos/libero_90_23/eval_episode_0.mp4",
]
STRIDE, MAX_SRC, MS = 3, 330, 100        # every 3rd frame of 20 Hz video, about 1.5x real time

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="docs/figures/m1_forgetting.gif")
ap.add_argument("panels", nargs="*", default=DEFAULT)
args = ap.parse_args()
PANELS = [p.split("|", 2) for p in args.panels]
OUT = Path(args.out)


def frames(path):
    with av.open(str(E / path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)][:MAX_SRC:STRIDE]


clips = [frames(p) for _, _, p in PANELS]
n = max(len(c) for c in clips)
h, w = clips[0][0].shape[:2]
bar = 34
out = []
for i in range(n):
    canvas = Image.new("RGB", (w * len(clips), h + bar), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for k, ((title, verdict, _), clip) in enumerate(zip(PANELS, clips)):
        canvas.paste(Image.fromarray(clip[min(i, len(clip) - 1)]), (k * w, bar))
        d.text((k * w + 4, 2), title, fill=(0, 0, 0))
        done = i >= len(clip) - 1
        d.text((k * w + 4, 17), verdict.upper() if done else f"t = {i * STRIDE / 20:4.1f} s",
               fill=(0, 130, 0) if verdict == "success" and done else (190, 0, 0) if done else (90, 90, 90))
    out.append(canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
OUT.parent.mkdir(parents=True, exist_ok=True)
out[0].save(OUT, save_all=True, append_images=out[1:], duration=MS, loop=0, optimize=True)
print(f"wrote {OUT}: {len(out)} frames, {OUT.stat().st_size / 1e6:.1f} MB")
