"""
Step 0 in the diagram -- happens exactly ONCE, before any training epoch.

Freezes the pretrained R3D-18 backbone and runs every clip through it.
For train, this saves the deterministic center-crop version PLUS
N_AUGMENTED_COPIES randomly-cropped/flipped versions of each clip, each
as its own cached feature vector -- so the classifier head trains on
more, more varied examples without the backbone ever running more than
once per version, and without re-running anything during training.
val and test get ONLY the deterministic version, on purpose: augmenting
evaluation data would make your accuracy numbers dishonest.

After this script has run, train_classifier.py never touches the
backbone or raw video again -- it only reads the cached .npy files.

Re-run this script if you change CROP_SIZE, N_AUGMENTED_COPIES, the trim
windows in config.json, or the backbone checkpoint. Changing the
classifier or training hyperparameters does NOT require re-running this
-- that's the whole point of freezing the backbone.

Run:
    python extract_features.py
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18

from video_dataset import VideoClipDataset, build_eval_transform, build_train_transform

SPLITS = ["train", "val", "test"]
FEATURES_DIR = Path("data/features")
CROP_SIZE = 112              # must match what train_classifier.py's classifier was designed for
N_AUGMENTED_COPIES = 4       # extra randomly-cropped/flipped versions PER TRAIN CLIP, on top
                              # of the 1 deterministic version. val/test always get 0.


def build_frozen_backbone():
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = torch.nn.Identity()  # output the pooled 512-d feature instead of logits
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_augmented", type=int, default=N_AUGMENTED_COPIES,
                         help="Extra randomly-cropped/flipped copies per TRAIN clip "
                              "(val/test always get 0, regardless of this flag).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = build_frozen_backbone().to(device)

    # one shared label<->index mapping across train/val/test, saved to disk
    # so train_classifier.py (and later, inference) can't drift out of sync
    train_ds_for_labels = VideoClipDataset("data/train.csv", transform=None)
    class_to_idx = train_ds_for_labels.class_to_idx
    Path("data").mkdir(exist_ok=True)
    Path("data/class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2))
    print("class_to_idx:", class_to_idx)

    eval_transform = build_eval_transform(size=CROP_SIZE)
    train_transform = build_train_transform(size=CROP_SIZE)

    for split in SPLITS:
        manifest_path = Path(f"data/{split}.csv")
        if not manifest_path.exists():
            print(f"[skip] {manifest_path} not found")
            continue

        # loaded with transform=None: we apply transforms manually below, since
        # train clips need MULTIPLE different transformed versions from the
        # same raw clip, not just the one the Dataset would apply automatically
        raw_ds = VideoClipDataset(manifest_path, transform=None, class_to_idx=class_to_idx)
        out_dir = FEATURES_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        n_extra = args.n_augmented if split == "train" else 0

        feat_rows = []
        with torch.no_grad():
            for i in range(len(raw_ds)):
                row = raw_ds.rows[i]
                raw_clip = np.load(row["path"])  # (T,H,W,3) uint8, no transform yet

                # version 0: deterministic center-crop, always saved for every split
                variants = [("", eval_transform(raw_clip))]
                # versions 1..n_extra: random crop + flip, TRAIN ONLY
                for aug_idx in range(n_extra):
                    variants.append((f"_aug{aug_idx}", train_transform(raw_clip)))

                for suffix, clip_tensor in variants:
                    feature = backbone(clip_tensor.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                    out_path = out_dir / f"{row['group_id']}{suffix}.npy"
                    np.save(out_path, feature)
                    feat_rows.append({"feature_path": str(out_path), "label": row["label"]})

        feat_manifest_path = Path(f"data/{split}_features.csv")
        with open(feat_manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["feature_path", "label"])
            writer.writeheader()
            writer.writerows(feat_rows)

        note = f" ({n_extra} augmented copies/clip)" if n_extra else ""
        print(f"[{split}] extracted {len(feat_rows)} feature vectors{note} -> {feat_manifest_path}")


if __name__ == "__main__":
    main()
