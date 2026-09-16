"""
Plots train/val loss and accuracy curves from data/training_history.csv.

This is the "can be done after" half of a training performance analysis --
run it any time once training_history.csv exists, including days later, on
a different machine, without re-training anything. The half that CANNOT be
done after the fact is the CSV itself: that only gets written while
train_classifier.py is actually running. If you trained before this
history-logging was added to train_classifier.py, there is no CSV to plot
-- you have to re-run training once to get one. Re-running is cheap here
(seconds, since the backbone is frozen and features are cached), but the
data literally does not exist otherwise.

Run:
    python plot_training_history.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HISTORY_PATH = Path("data/training_history.csv")
SUMMARY_PATH = Path("data/training_summary.json")
OUT_PATH = Path("data/eval/training_curves.png")


def main():
    if not HISTORY_PATH.exists():
        raise SystemExit(
            f"{HISTORY_PATH} not found. This file is only written while "
            f"train_classifier.py is running -- run (or re-run) it first."
        )

    with open(HISTORY_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    train_acc = [float(r["train_acc"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]

    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    best_epoch = summary.get("best_epoch")

    # F1 / ROC-AUC columns only exist in histories written by the newer
    # train_classifier.py -- older CSVs still plot the first two panels
    has_extra = "val_f1_macro" in rows[0]
    if has_extra:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, train_loss, label="train")
    ax1.plot(epochs, val_loss, label="val")
    if best_epoch:
        ax1.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                     label=f"best epoch ({best_epoch})")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, train_acc, label="train")
    ax2.plot(epochs, val_acc, label="val")
    if best_epoch:
        ax2.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(0, 1)
    ax2.set_title("Accuracy")
    ax2.legend()

    if has_extra:
        ax3.plot(epochs, [float(r["train_f1_macro"]) for r in rows], label="train F1")
        ax3.plot(epochs, [float(r["val_f1_macro"]) for r in rows], label="val F1")
        ax3.plot(epochs, [float(r["val_roc_auc_macro"]) for r in rows], label="val ROC-AUC")
        if best_epoch:
            ax3.axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
        ax3.set_xlabel("epoch")
        ax3.set_ylabel("score")
        ax3.set_ylim(0, 1.02)
        ax3.set_title("F1 (macro) & ROC-AUC")
        ax3.legend()

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"saved -> {OUT_PATH}")

    if summary:
        print(f"\n{summary['epochs_run']}/{summary['epochs_planned']} epochs run "
              f"({'stopped early' if summary['stopped_early'] else 'ran to completion'}), "
              f"{summary['wall_clock_seconds']}s on {summary['device']}")
        print(f"best epoch: {best_epoch}  "
              f"(val_loss={summary['best_val_loss']:.4f}, val_acc={summary['best_val_acc']:.3f}"
              + (f", val_f1={summary['best_val_f1_macro']:.3f}, "
                 f"val_roc_auc={summary['best_val_roc_auc_macro']:.3f}"
                 if "best_val_f1_macro" in summary else "") + ")")

    # a quick, honest flag: is train pulling away from val? worth checking by eye
    # in the plot regardless, but this catches the obvious case in text output too
    if train_acc[-1] - val_acc[-1] > 0.25:
        print(f"\nNote: train_acc ({train_acc[-1]:.2f}) is well above val_acc "
              f"({val_acc[-1]:.2f}) by the last epoch -- worth checking the plot "
              f"for overfitting rather than assuming the gap is fine.")


if __name__ == "__main__":
    main()
