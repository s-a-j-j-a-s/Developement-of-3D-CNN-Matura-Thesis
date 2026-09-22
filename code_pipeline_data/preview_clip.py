"""
Ist für den visual check der preprocessed clips. Man kann damit sicher stellen, dass es das richtige preprocessed
und danach auch die zeiten im config file anpassen
Run:
    python preview_clip.py data/processed/class_a/class_a_000.npy
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_contact_sheet(clip: np.ndarray, n_frames: int = 4) -> Image.Image:  #der clip wird in 4 frames 'zusammengefasst'
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
    parser.add_argument("--n_frames", type=int, default=4) # mit diesem Befehl kann man frames 'anhängen' z.b 5 -> clip wird in 5 frames zusammengefasst
    args = parser.parse_args()

    clip = np.load(args.npy_path)
    print(f"loaded {args.npy_path}: shape={clip.shape} dtype={clip.dtype} "   
          f"min={clip.min()} max={clip.max()}")

    sheet = make_contact_sheet(clip, n_frames=args.n_frames) 
    out_path = args.out or args.npy_path.with_suffix(".preview.png") # z.b start data\processed\spike\2026-07-09T17-57-40_3815_1.preview.png
    sheet.save(out_path)
    print(f"saved preview -> {out_path}")


if __name__ == "__main__":
    main()
