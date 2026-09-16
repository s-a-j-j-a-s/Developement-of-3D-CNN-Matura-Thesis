"""
Stage 1: trim (per-class rules) + convert to RGB + build a manifest.

Expected input layout:
    data/raw/<class_name>/<video_id>.mp4

Each class's trim window is read from config.json (start_sec / end_sec,
measured from the start of the 2-second clip). `target_num_frames` frames
are sampled uniformly from that window and saved as a single .npy file
per clip, shape (T, H, W, 3), dtype uint8, RGB order.

We decode the whole clip sequentially rather than seeking, since seeking
in compressed mp4 can land on the wrong frame -- and at ~2 seconds per
clip, decoding everything is cheap anyway.

Run:
    python trim_and_convert.py --config config.json
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def read_all_frames_rgb(video_path: Path) -> tuple[list[np.ndarray], float]:
    """Decode every frame of a video, converting BGR -> RGB."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def select_trimmed_indices(n_total: int, fps: float, start_sec: float,
                            end_sec: float, target_num_frames: int) -> np.ndarray:
    """Map a [start_sec, end_sec) window to `target_num_frames` frame indices,
    sampled uniformly, clamped to the frames actually available."""
    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))

    start_frame = max(0, min(start_frame, n_total - 1))
    end_frame = max(start_frame + 1, min(end_frame, n_total))

    if end_frame - start_frame < target_num_frames:
        # window too short for the requested frame count -- fall back to
        # the full clip so we don't sample duplicate frames excessively
        start_frame, end_frame = 0, n_total

    idx = np.linspace(start_frame, end_frame - 1, target_num_frames)
    return np.round(idx).astype(int)


def process_class(class_name: str, class_dir: Path, trim_cfg: dict,
                   out_dir: Path, target_num_frames: int, manifest_writer,
                   errors: list[str], limit_per_class: int | None = None) -> int:
    out_class_dir = out_dir / class_name
    out_class_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(class_dir.glob("*.mp4"))
    if limit_per_class is not None:
        video_paths = video_paths[:limit_per_class]

    n_written = 0
    for video_path in video_paths:
        try:
            frames, fps = read_all_frames_rgb(video_path)
            if len(frames) == 0:
                raise IOError("no frames decoded")

            indices = select_trimmed_indices(
                n_total=len(frames), fps=fps,
                start_sec=trim_cfg["start_sec"], end_sec=trim_cfg["end_sec"],
                target_num_frames=target_num_frames,
            )
            clip = np.stack([frames[i] for i in indices], axis=0)  # (T,H,W,3) uint8 RGB

            out_path = out_class_dir / f"{video_path.stem}.npy"
            np.save(out_path, clip)

            manifest_writer.writerow({
                "path": str(out_path),
                "label": class_name,
                "group_id": video_path.stem,  # override upstream if multiple
                                               # clips share a subject/session
                "n_frames": clip.shape[0],
                "height": clip.shape[1],
                "width": clip.shape[2],
                "source_fps": round(fps, 3),
            })
            n_written += 1
        except Exception as e:  # noqa: BLE001 - keep the batch going
            errors.append(f"{video_path}: {e}")

    return n_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--limit_per_class", type=int, default=None,
                         help="Only process this many videos per class -- "
                              "use this for a quick smoke test before running "
                              "on the full dataset.")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    raw_dir = Path(cfg["raw_dir"])
    out_dir = Path(cfg["processed_dir"])
    manifest_path = Path(cfg["manifest_path"])
    target_num_frames = cfg["target_num_frames"]
    class_cfgs = cfg["classes"]

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label", "group_id", "n_frames", "height", "width", "source_fps"]

    errors: list[str] = []
    total = 0
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for class_name, trim_cfg in class_cfgs.items():
            class_dir = raw_dir / class_name
            if not class_dir.is_dir():
                print(f"[skip] no raw folder for class '{class_name}' at {class_dir}")
                continue
            n = process_class(class_name, class_dir, trim_cfg, out_dir,
                               target_num_frames, writer, errors,
                               limit_per_class=args.limit_per_class)
            print(f"[{class_name}] wrote {n} clips (trim {trim_cfg['start_sec']}s-{trim_cfg['end_sec']}s)")
            total += n

    limit_note = f" (--limit_per_class={args.limit_per_class})" if args.limit_per_class else ""
    print(f"\nDone. {total} clips written{limit_note}. Manifest: {manifest_path}")
    if errors:
        print(f"\n{len(errors)} file(s) failed and were skipped:")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
