"""
Shared metric helpers used by train_classifier.py, evaluate.py and
training_report.py, so all three compute F1 and ROC-AUC the same way.

- F1 (macro): harmonic mean of precision and recall, computed per class and
  then averaged with equal weight per class. Unlike accuracy it doesn't hide
  a class that the model keeps getting wrong.
- ROC-AUC (one-vs-rest, macro): for each class, "this class vs. all others",
  how well the softmax probability separates the two -- across ALL possible
  thresholds, not just argmax. 0.5 = random guessing, 1.0 = perfect.
  AUC just means "Area Under the Curve"; ROC-AUC is the area under the ROC
  curve (true-positive rate vs. false-positive rate).
- PR-AUC (average precision): area under the precision-recall curve. A second
  AUC that is stricter than ROC-AUC when a class is rare.

Not run directly.
"""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true, y_pred, y_prob, num_classes: int) -> dict:
    """y_true/y_pred: (N,) int labels. y_prob: (N, C) softmax probabilities."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    labels = list(range(num_classes))

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels,
                                      average="weighted", zero_division=0)),
    }

    # ROC-AUC is undefined for a class that doesn't appear in y_true (no
    # positives to rank). With small val/test splits that can happen, so
    # return NaN instead of crashing the training loop.
    y_bin = label_binarize(y_true, classes=labels)
    present = [c for c in labels if 0 < y_bin[:, c].sum() < len(y_true)]
    if len(present) == num_classes:
        out["roc_auc_macro"] = float(roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro", labels=labels))
        out["pr_auc_macro"] = float(average_precision_score(
            y_bin, y_prob, average="macro"))
    else:
        out["roc_auc_macro"] = float("nan")
        out["pr_auc_macro"] = float("nan")
    return out


def per_class_auc(y_true, y_prob, class_names) -> dict:
    """ROC-AUC and PR-AUC per class (one-vs-rest)."""
    labels = list(range(len(class_names)))
    y_bin = label_binarize(y_true, classes=labels)
    result = {}
    for c, name in enumerate(class_names):
        pos = y_bin[:, c]
        if 0 < pos.sum() < len(pos):
            result[name] = {
                "roc_auc": float(roc_auc_score(pos, y_prob[:, c])),
                "pr_auc": float(average_precision_score(pos, y_prob[:, c])),
            }
        else:
            result[name] = {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return result
