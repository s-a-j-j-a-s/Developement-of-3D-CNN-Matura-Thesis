"""
Evaluation -- run this AFTER train_classifier.py has finished.

Implements the testing/evaluation diagram: takes the cached test-set
feature vectors (already produced by extract_features.py using the same
frozen R3D-18 backbone -- the backbone is never re-run or unfrozen
here), pushes them through the trained classifier in eval mode, and
reports accuracy, F1, ROC-AUC / PR-AUC, a per-class report, a confusion
matrix and ROC curves. All numbers are also saved to data/eval/metrics.json.

Why there's no backbone code in this file: extract_features.py already
wrote data/test_features.csv in the same single pass that produced the
train/val features, using identical weights. Re-running the backbone
here would just recompute the exact same numbers.

Run only ONCE, at the very end, after you've finished tuning on the
validation set. Every time you look at the test score and then change
something, the test set leaks a little into your decisions and stops
being an honest estimate of real-world performance. Use val for tuning.

Run:
    python evaluate.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNG without needing a display
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from torch.utils.data import DataLoader

from feature_dataset import FeatureDataset
from metrics_utils import (compute_metrics, confusion_matrix_torch,
                           format_per_class_report, per_class_auc, per_class_prf)

FEATURE_DIM = 512
DROPOUT_P = 0.5  # must match train_classifier.py so the state_dict shapes line up


def build_classifier(num_classes: int) -> torch.nn.Module:
    # Same architecture as in train_classifier.py. Dropout is included so
    # the layer indices in the state_dict match; calling .eval() below
    # turns it off automatically.
    return torch.nn.Sequential(
        torch.nn.Dropout(DROPOUT_P),
        torch.nn.Linear(FEATURE_DIM, num_classes),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/test_features.csv"),
                         help="Which split to evaluate. Point at val_features.csv "
                              "while you're still tuning; only use the test split "
                              "for your final, one-time number.")
    parser.add_argument("--ckpt", type=Path, default=Path("data/best_classifier.pt"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/eval"))
    args = parser.parse_args()

    class_to_idx = json.loads(Path("data/class_to_idx.json").read_text())
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    ds = FeatureDataset(args.features, class_to_idx)
    loader = DataLoader(ds, batch_size=32, shuffle=False)  # no shuffle: keeps
                                                           # predictions aligned
                                                           # with ds.rows order

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = build_classifier(len(class_names)).to(device)
    classifier.load_state_dict(torch.load(args.ckpt, map_location=device))
    classifier.eval()  # dropout OFF

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():  # no gradients, no weight updates
        for features, labels in loader:
            features = features.to(device)
            logits = classifier(features)
            # softmax only here, at inference -- the training loss applied
            # it internally, so the model itself outputs raw logits
            probs = torch.softmax(logits, dim=1)

            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_prob, len(class_names))
    acc = metrics["accuracy"]
    print(f"\nEvaluated {len(y_true)} clips from {args.features}")
    print(f"Accuracy: {acc:.4f}\n")

    # per-class precision/recall/F1 -- with 3 classes this tells you far
    # more than a single accuracy number, especially if classes are imbalanced
    per_class = per_class_prf(y_true, y_pred, class_names)
    print(format_per_class_report(per_class))
    print()

    class_auc = per_class_auc(y_true, y_prob, class_names)
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"F1 (macro):     {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted):  {metrics['f1_weighted']:.4f}")
    print(f"ROC-AUC (macro, one-vs-rest): {metrics['roc_auc_macro']:.4f}")
    print(f"PR-AUC  (macro):              {metrics['pr_auc_macro']:.4f}")
    for name, a in class_auc.items():
        print(f"  {name:<12} ROC-AUC={a['roc_auc']:.4f}  PR-AUC={a['pr_auc']:.4f}")
    print()

    cm = confusion_matrix_torch(y_true, y_pred, len(class_names)).numpy()
    print("Confusion matrix (rows = true, cols = predicted):")
    print(cm)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion matrix -- accuracy {acc:.3f}")
    plt.tight_layout()
    cm_path = args.out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\nsaved -> {cm_path}")

    # ROC curves: one line per class (one-vs-rest). The further a curve bends
    # towards the top-left corner, the better; the diagonal = random guessing.
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for c, name in enumerate(class_names):
        pos = (y_true == c).astype(int)
        if 0 < pos.sum() < len(pos):
            fpr, tpr, _ = roc_curve(pos, y_prob[:, c])
            ax.plot(fpr, tpr, label=f"{name} (AUC {class_auc[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curves -- macro AUC {metrics['roc_auc_macro']:.3f}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = args.out_dir / "roc_curves.png"
    plt.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {roc_path}")

    # all numbers in one file, so you can cite them in the written report
    # without re-running (and re-peeking at) the test set
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps({
        "features": str(args.features),
        "n_clips": int(len(y_true)),
        **metrics,
        "per_class": {name: {**per_class[name], **class_auc[name]} for name in class_names},
        "confusion_matrix": cm.tolist(),
    }, indent=2))
    print(f"saved -> {metrics_path}")

    # per-clip predictions, so you can go look at the clips the model got
    # wrong -- usually far more informative than the aggregate numbers
    pred_csv = args.out_dir / "predictions.csv"
    with open(pred_csv, "w") as f:
        prob_cols = ",".join(f"prob_{n}" for n in class_names)
        f.write(f"feature_path,true_label,predicted_label,confidence,correct,{prob_cols}\n")
        for row, t, p, probs in zip(ds.rows, y_true, y_pred, y_prob):
            prob_vals = ",".join(f"{v:.4f}" for v in probs)
            f.write(f"{row['feature_path']},{idx_to_class[t]},{idx_to_class[p]},"
                    f"{probs[p]:.4f},{int(t == p)},{prob_vals}\n")
    print(f"saved -> {pred_csv}")

    n_wrong = int((y_true != y_pred).sum())
    if n_wrong:
        print(f"\n{n_wrong} misclassified clip(s) -- see {pred_csv} "
              f"(filter on correct=0) and check those clips with preview_clip.py")


if __name__ == "__main__":
    main()
