"""
Dataset + transforms for the preprocessed .npy clips.

Augmentation happens here, on load, NOT as a separate disk-writing
script -- so the train set gets fresh random crops/flips every epoch
for free, and only ever touches train.csv.

The one video-specific rule the transforms below follow: every random
parameter (crop offset, flip decision) is drawn ONCE per clip and then
applied to all T frames identically. If each frame got its own
independent random crop/flip, the clip would no longer look like
smooth motion to the 3D convolutions.

r3d_18 expects input shape (C, T, H, W) per sample (batches: N,C,T,H,W).
Default crop size is 112 to match the pretrained recipe -- bump `size`
in build_train_transform/build_eval_transform to 176 or 224 if you need
more spatial detail, but expect ~3-4x more compute/memory at 224, and
plan to fine-tune (not just linear-probe) at the new resolution.
"""
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Matches the mean/std torchvision's pretrained r3d_18 was trained with.
# Keep these if you're fine-tuning at or near 112x112. If you move to a
# substantially different resolution/domain, consider recomputing your
# own mean/std over the training set instead.
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]


# ---- clip-level transform primitives (all operate on (T,H,W,3) uint8 arrays) ----

def resize_clip(clip: np.ndarray, short_side: int) -> np.ndarray:
    t, h, w, _ = clip.shape
    if h <= w:
        new_h, new_w = short_side, int(round(w * short_side / h))
    else:
        new_w, new_h = short_side, int(round(h * short_side / w))
    resized = np.empty((t, new_h, new_w, 3), dtype=clip.dtype)
    for i in range(t):
        resized[i] = cv2.resize(clip[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


def random_crop_clip(clip: np.ndarray, size: int) -> np.ndarray:
    t, h, w, _ = clip.shape
    top = random.randint(0, max(0, h - size))
    left = random.randint(0, max(0, w - size))
    return clip[:, top:top + size, left:left + size, :]


def center_crop_clip(clip: np.ndarray, size: int) -> np.ndarray:
    t, h, w, _ = clip.shape
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return clip[:, top:top + size, left:left + size, :]


def random_hflip_clip(clip: np.ndarray, p: float = 0.5) -> np.ndarray:
    if random.random() < p:
        return clip[:, :, ::-1, :].copy()
    return clip


def clip_to_normalized_tensor(clip: np.ndarray, mean, std) -> torch.Tensor:
    # (T,H,W,C) uint8 -> (C,T,H,W) float32, normalized
    tensor = torch.from_numpy(clip).float() / 255.0
    tensor = tensor.permute(3, 0, 1, 2)  # C,T,H,W
    mean_t = torch.tensor(mean).view(3, 1, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1, 1)
    return (tensor - mean_t) / std_t


# ---- transform factories ----

def build_train_transform(size: int = 112, resize_short_side: int | None = None):
    resize_short_side = resize_short_side or int(size * 1.15)

    def _transform(clip: np.ndarray) -> torch.Tensor:
        clip = resize_clip(clip, resize_short_side)
        clip = random_crop_clip(clip, size)
        clip = random_hflip_clip(clip, p=0.5)
        return clip_to_normalized_tensor(clip, KINETICS_MEAN, KINETICS_STD)

    return _transform


def build_eval_transform(size: int = 112):
    def _transform(clip: np.ndarray) -> torch.Tensor:
        clip = resize_clip(clip, size)
        clip = center_crop_clip(clip, size)
        return clip_to_normalized_tensor(clip, KINETICS_MEAN, KINETICS_STD)

    return _transform


# ---- dataset ----

class VideoClipDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, transform=None, class_to_idx: dict | None = None):
        with open(manifest_csv, newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"manifest {manifest_csv} is empty")

        self.transform = transform
        if class_to_idx is None:
            labels = sorted({row["label"] for row in self.rows})
            class_to_idx = {label: i for i, label in enumerate(labels)}
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        clip = np.load(row["path"])  # (T,H,W,3) uint8 RGB
        if self.transform is not None:
            clip = self.transform(clip)
        label = self.class_to_idx[row["label"]]
        return clip, label
