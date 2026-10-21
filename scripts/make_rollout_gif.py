#!/usr/bin/env python
"""A GIF of one key run by two policies side by side, for the README. CPU only.

    uv run scripts/make_rollout_gif.py

Reads two videos written by scripts/rerun_with_video.py (by default A and A′ on task 1,
initial state 36, seed 0: same scene, same random numbers, A succeeds and A′ fails) and their
outcomes in results/tables/rq2_video_rerun.csv, and writes figures/rollout_pair.gif. A panel
whose episode has ended holds its last frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = {"smolvla_n10": "A  (SmolVLA, 10 actions)", "smolvla": "A′ (SmolVLA, 50 actions)"}
SURFACE, INK, INK_2 = (252, 252, 251), (11, 11, 11), (82, 81, 78)


def frames(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as video:
        return [f.to_ndarray(format="rgb24") for f in video.decode(video=0)]


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--policies", nargs=2, default=["smolvla_n10", "smolvla"])
    parser.add_argument("--key", type=int, nargs=3, default=[1, 36, 0], metavar=("TASK", "STATE", "SEED"))  # fmt: skip
    parser.add_argument("--videos-dir", type=Path, default=REPO_ROOT / "figures/videos")
    parser.add_argument(
        "--outcomes", type=Path, default=REPO_ROOT / "results/tables/rq2_video_rerun.csv"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "figures/rollout_pair.gif")
    parser.add_argument("--every", type=int, default=4, help="keep one simulator step in this many")
    parser.add_argument("--size", type=int, default=220, help="pixels per panel side")
    parser.add_argument(
        "--colors", type=int, default=64, help="palette size, to keep the file small"
    )
    args = parser.parse_args()

    task, state, seed = args.key
    outcomes = pd.read_csv(args.outcomes).set_index(["policy", "task_id", "init_state_id", "seed"])
    panels = []
    for name in args.policies:
        row = outcomes.loc[(name, task, state, seed)]
        if not row["same"]:
            raise SystemExit(f"{name}: the video differs from the stored episode, pick another key")
        stem = f"{name}__task{task:02d}_init{state:02d}_seed{seed}.mp4"
        panels.append((name, frames(args.videos_dir / stem), int(row["rerun_success"]), int(row["rerun_len"])))  # fmt: skip
    task_name = str(outcomes.loc[(args.policies[0], task, state, seed), "task_name"]).replace(
        "_", " "
    )

    n_steps = max(len(f) for _, f, _, _ in panels)
    steps = list(range(0, n_steps, args.every)) + [n_steps - 1]
    header, footer, gap = 34, 30, 8
    width = len(panels) * args.size + (len(panels) + 1) * gap
    height = header + args.size + footer
    big, small = font(13), font(12)

    images = []
    for step in steps:
        canvas = Image.new("RGB", (width, height), SURFACE)
        draw = ImageDraw.Draw(canvas)
        for k, (name, fr, success, length) in enumerate(panels):
            x = gap + k * (args.size + gap)
            done = step >= length
            frame = Image.fromarray(fr[min(step, len(fr) - 1)]).resize((args.size, args.size))
            canvas.paste(frame, (x, header))
            draw.text((x, 6), LABELS.get(name, name), fill=INK, font=big)
            if done:
                text = f"success at step {length}" if success else f"failure: step limit ({length} steps)"  # fmt: skip
                draw.text((x, header + args.size + 4), text, fill=INK, font=big)
            else:
                draw.text((x, header + args.size + 4), f"step {step}", fill=INK_2, font=small)
        draw.text((gap, 22), task_name, fill=INK_2, font=font(10))
        images.append(canvas.quantize(colors=args.colors, dither=Image.Dither.NONE))

    # About real time: the simulator runs at 20 control steps per second.
    durations = [int(1000 * args.every / 20)] * (len(images) - 1) + [2500]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(args.out, save_all=True, append_images=images[1:], duration=durations,
                   loop=0, optimize=True)  # fmt: skip
    print(f"wrote {args.out} ({len(images)} frames, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
