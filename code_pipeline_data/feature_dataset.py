"""
Loads pre-extracted, cached feature vectors written by extract_features.py.
Used only inside the training loop (steps 4-10 of the pipeline) -- it
never touches raw video or the R3D-18 backbone.
"""
import csv

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, features_manifest_csv, class_to_idx: dict):
        with open(features_manifest_csv, newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"{features_manifest_csv} is empty")
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        feature = np.load(row["feature_path"])  # (512,) float32
        label = self.class_to_idx[row["label"]]
        return torch.from_numpy(feature).float(), label
