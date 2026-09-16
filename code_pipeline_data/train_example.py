"""
Minimal example wiring the pieces together. Not a full training loop --
just shows how the dataset/transforms feed r3d_18 with a swapped-out
head for your 3 classes, and confirms the shapes line up.

Run:
    python train_example.py
"""
import torch
from torch.utils.data import DataLoader
from torchvision.models.video import R3D_18_Weights, r3d_18

from video_dataset import VideoClipDataset, build_eval_transform, build_train_transform

CROP_SIZE = 112  # bump to 176/224 later if 112 can't separate your classes
NUM_CLASSES = 3

train_ds = VideoClipDataset(
    "data/train.csv",
    transform=build_train_transform(size=CROP_SIZE),
)
val_ds = VideoClipDataset(
    "data/val.csv",
    transform=build_eval_transform(size=CROP_SIZE),
    class_to_idx=train_ds.class_to_idx,  # keep label indices consistent across splits
)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# sanity check: one batch through the model
clips, labels = next(iter(train_loader))
print("batch shape:", clips.shape)  # (N, C, T, H, W)
logits = model(clips.to(device))
print("logits shape:", logits.shape)  # (N, NUM_CLASSES)

print("\nclass_to_idx:", train_ds.class_to_idx)
