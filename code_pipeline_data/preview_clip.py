"""
Quick visual sanity check for ONE preprocessed clip: saves a contact
sheet (a few evenly spaced frames, side by side) as a PNG, so you can
actually look at it and confirm the trim window and RGB conversion are
correct -- before running the full dataset through the pipeline.

Run:
    python preview_clip.py data/processed/class_a/class_a_000.npy
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_contact_sheet(clip: np.ndarray, n_frames: int = 4) -> Image.Image:
    t = clip.shape[0]
    indices = np.linspace(0, t - 1, min(n_frames, t)).round().astype(int)
    frames = [Image.fromarray(clip[i]) for i in indices]
    w, h = frames[0].size
    sheet = Image.new("RGB", (w * len(frames), h))
    for i, frame in enumerate(frames):
        sheet.paste(frame, (i * w, 0))
    return sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n_frames", type=int, default=4)
    args = parser.parse_args()

    clip = np.load(args.npy_path)
    print(f"loaded {args.npy_path}: shape={clip.shape} dtype={clip.dtype} "
          f"min={clip.min()} max={clip.max()}")

    sheet = make_contact_sheet(clip, n_frames=args.n_frames)
    out_path = args.out or args.npy_path.with_suffix(".preview.png")
    sheet.save(out_path)
    print(f"saved preview -> {out_path}")


if __name__ == "__main__":
    main()
