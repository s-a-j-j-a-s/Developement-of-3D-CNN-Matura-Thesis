"""
Shared metric helpers used by train_classifier.py, evaluate.py and
training_report.py, so all three compute precision, recall, F1 and ROC-AUC
the same way.

Accuracy, precision, recall and F1 are computed in pure PyTorch, directly
from the confusion matrix -- same tensor library as the rest of the
pipeline, and the formulas are visible here instead of hidden inside a
library call. Only ROC-AUC and PR-AUC still come from scikit-learn: those
need the curve over all thresholds, which is a lot more code than a
confusion matrix.

Confusion matrix convention: cm[t, p] = number of clips whose true class is
t and that were predicted as p. For a class c:

    TP = cm[c, c]                    correctly recognised clips of class c
    FP = cm[:, c].sum() - TP         other actions wrongly predicted as c
    FN = cm[c, :].sum() - TP         clips of class c predicted as something else

- Precision (macro): TP / (TP + FP) per class, then averaged with equal
  weight per class. "Of all clips the model called spike, how many really
  were spikes."
- Recall (macro): TP / (TP + FN) per class, averaged with equal weight.
  "Of all actual spikes, how many the model found."
- F1 (macro): harmonic mean of precision and recall per class, averaged
  with equal weight. Unlike accuracy it doesn't hide a class that the model
  keeps getting wrong.
- F1 (weighted): the same per-class F1 values, but averaged weighted by how
  often each class actually occurs.
- ROC-AUC (one-vs-rest, macro): for each class, "this class vs. all others",
  how well the softmax probability separates the two -- across ALL possible
  thresholds, not just argmax. 0.5 = random guessing, 1.0 = perfect.
  AUC just means "Area Under the Curve"; ROC-AUC is the area under the ROC
  curve (true-positive rate vs. false-positive rate).
- PR-AUC (average precision): area under the precision-recall curve. A second
  AUC that is stricter than ROC-AUC when a class is rare.

A division by zero (a class that is never predicted, or that never occurs in
this split) yields 0.0 here -- the same convention as scikit-learn's
zero_division=0, so the training loop doesn't crash on a small val split.

Not run directly.
"""
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _one_hot(y_true, num_classes: int) -> np.ndarray:
    """(N,) int labels -> (N, C) 0/1 matrix, one column per class.

    Written out instead of sklearn's label_binarize because that returns a
    single column for a 2-class problem, which breaks the per-class loops
    below. This always gives exactly num_classes columns.
    """
    return np.eye(num_classes, dtype=int)[np.asarray(y_true).reshape(-1)]


def confusion_matrix_torch(y_true, y_pred, num_classes: int) -> torch.Tensor:
    """cm[t, p] = how many clips of true class t were predicted as p.

    Accepts torch tensors, numpy arrays or lists of int labels. Built with a
    single bincount over the flattened index t * num_classes + p, which is
    the usual trick for counting pairs without a Python loop.
    """
    t = torch.as_tensor(np.asarray(y_true), dtype=torch.long).reshape(-1)
    p = torch.as_tensor(np.asarray(y_pred), dtype=torch.long).reshape(-1)
    if t.numel() != p.numel():
        raise ValueError(f"y_true has {t.numel()} entries, y_pred has {p.numel()}")
    flat = torch.bincount(t * num_classes + p, minlength=num_classes ** 2)
    return flat.reshape(num_classes, num_classes)


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Element-wise division that returns 0.0 where the denominator is 0.

    The denominator is replaced by 1 only where it is 0, so the division
    itself never produces NaN. (Do NOT clamp to a minimum of 1 here -- the
    denominator of the F1 formula, precision + recall, is usually smaller
    than 1 and would be silently distorted.)
    """
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return torch.where(denominator > 0,
                       numerator / safe,
                       torch.zeros_like(numerator))


def prf_from_confusion(cm: torch.Tensor):
    """Per-class precision, recall, F1 and support from a confusion matrix."""
    cm = cm.to(torch.float64)
    tp = torch.diagonal(cm)
    support = cm.sum(dim=1)        # row sums: how often each class really occurs
    predicted = cm.sum(dim=0)      # column sums: how often each class was predicted

    precision = _safe_div(tp, predicted)          # TP / (TP + FP)
    recall = _safe_div(tp, support)               # TP / (TP + FN)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1, support


def macro_f1(y_true, y_pred, num_classes: int) -> float:
    """Macro F1 on its own -- used for the rough train-F1 inside the loop."""
    _, _, f1, _ = prf_from_confusion(confusion_matrix_torch(y_true, y_pred, num_classes))
    return float(f1.mean())


def compute_metrics(y_true, y_pred, y_prob, num_classes: int) -> dict:
    """y_true/y_pred: (N,) int labels. y_prob: (N, C) softmax probabilities."""
    cm = confusion_matrix_torch(y_true, y_pred, num_classes)
    precision, recall, f1, support = prf_from_confusion(cm)
    total = support.sum()
    weights = _safe_div(support, total.expand_as(support))

    out = {
        "accuracy": float(_safe_div(torch.diagonal(cm.to(torch.float64)).sum(), total)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "f1_weighted": float((f1 * weights).sum()),
    }

    # ROC-AUC is undefined for a class that doesn't appear in y_true (no
    # positives to rank). With small val/test splits that can happen, so
    # return NaN instead of crashing the training loop.
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    labels = list(range(num_classes))
    y_bin = _one_hot(y_true, num_classes)
    present = [c for c in labels if 0 < y_bin[:, c].sum() < len(y_true)]
    if len(present) == num_classes:
        # With exactly 2 classes sklearn expects the probability of the
        # positive class as a 1d array, not an (N, 2) matrix.
        score = y_prob[:, 1] if num_classes == 2 else y_prob
        out["roc_auc_macro"] = float(roc_auc_score(
            y_true, score, multi_class="ovr", average="macro", labels=labels))
        out["pr_auc_macro"] = float(average_precision_score(
            y_bin, y_prob, average="macro"))
    else:
        out["roc_auc_macro"] = float("nan")
        out["pr_auc_macro"] = float("nan")
    return out


def per_class_prf(y_true, y_pred, class_names) -> dict:
    """Precision, recall, F1 and support per class (the per-class table)."""
    cm = confusion_matrix_torch(y_true, y_pred, len(class_names))
    precision, recall, f1, support = prf_from_confusion(cm)
    return {
        name: {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1-score": float(f1[c]),
            "support": int(support[c]),
        }
        for c, name in enumerate(class_names)
    }


def format_per_class_report(per_class: dict) -> str:
    """Render per_class_prf() as a readable table (like sklearn's report)."""
    width = max([len(n) for n in per_class] + [12])
    lines = [f"{'':<{width}} {'precision':>9} {'recall':>7} {'f1-score':>9} {'support':>8}"]
    for name, m in per_class.items():
        lines.append(f"{name:<{width}} {m['precision']:>9.3f} {m['recall']:>7.3f} "
                     f"{m['f1-score']:>9.3f} {m['support']:>8d}")
    n = sum(m["support"] for m in per_class.values())
    macro_p = sum(m["precision"] for m in per_class.values()) / len(per_class)
    macro_r = sum(m["recall"] for m in per_class.values()) / len(per_class)
    macro_f = sum(m["f1-score"] for m in per_class.values()) / len(per_class)
    lines.append(f"{'macro avg':<{width}} {macro_p:>9.3f} {macro_r:>7.3f} "
                 f"{macro_f:>9.3f} {n:>8d}")
    return "\n".join(lines)


def per_class_auc(y_true, y_prob, class_names) -> dict:
    """ROC-AUC and PR-AUC per class (one-vs-rest). Still scikit-learn."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_bin = _one_hot(y_true, len(class_names))
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
