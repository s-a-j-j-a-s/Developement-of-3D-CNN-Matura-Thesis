"""
Steps 4-10 in the diagram -- the training loop, repeated for many epochs.

Trains a small classifier head on the cached, frozen R3D-18 features
from extract_features.py. Never touches the backbone or raw video, so
this is cheap to re-run many times while you tune hyperparameters --
you only need to re-run extract_features.py if the trim windows, crop
size, or backbone checkpoint change.

Prerequisite (run once):
    python extract_features.py

Run:
    python train_classifier.py
"""
import csv
import json
import time
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader

from feature_dataset import FeatureDataset
from metrics_utils import compute_metrics, macro_f1

FEATURE_DIM = 512      # r3d_18's penultimate layer output size, do not change unless the backbone does
NUM_EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-3
DROPOUT_P = 0.5 
SEED = 0
WEIGHT_DECAY = 0.0
PATIENCE = 5            # stop if val loss hasn't improved for this many epochs
CKPT_PATH = Path("data/best_classifier.pt")
HISTORY_PATH = Path("data/training_history.csv")     # per-epoch loss/acc/F1/ROC-AUC
SUMMARY_PATH = Path("data/training_summary.json")     # best epoch, wall time, early-stop info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--out_dir", type=Path, default=Path("data"))
    parser.add_argument("--train_features", type=Path, default=Path("data/train_features.csv"))
    parser.add_argument("--val_features", type=Path, default=Path("data/val_features.csv"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "best_classifier.pt"
    history_path = args.out_dir / "training_history.csv"
    summary_path = args.out_dir / "training_summary.json"

    torch.manual_seed(args.seed)
    
    class_to_idx = json.loads(Path("data/class_to_idx.json").read_text())
    num_classes = len(class_to_idx)
    
    train_ds = FeatureDataset(args.train_features, class_to_idx)
    val_ds = FeatureDataset(args.val_features, class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # "Dropout + Linear + Softmax" from the diagram, minus the Softmax:
    # nn.CrossEntropyLoss applies log-softmax internally, so the model
    # must output raw logits here. An explicit Softmax before the loss
    # would double-apply softmax and distort the gradients. If you want
    # actual probabilities (e.g. to show prediction confidence), apply
    # torch.softmax(logits, dim=1) separately at inference time only.
    classifier = torch.nn.Sequential(
        torch.nn.Dropout(DROPOUT_P),
        torch.nn.Linear(FEATURE_DIM, num_classes),
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []  # one dict per completed epoch -- this IS the training performance record
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- one epoch: loop over training batches ("next batch" in the diagram) ----
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_true, train_pred = [], []
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_true.append(labels.cpu())
            train_pred.append(logits.argmax(dim=1).detach().cpu())
        train_loss /= len(train_ds)
        train_acc = train_correct / len(train_ds)

        # ---- validate + early stopping check ----
        classifier.eval()
        val_loss = 0.0
        val_true, val_pred, val_prob = [], [], []
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                logits = classifier(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * features.size(0)
                val_true.append(labels.cpu())
                val_pred.append(logits.argmax(dim=1).cpu())
                val_prob.append(torch.softmax(logits, dim=1).cpu())  # probs for ROC-AUC
        val_loss /= len(val_ds)
        val_m = compute_metrics(torch.cat(val_true).numpy(), torch.cat(val_pred).numpy(),
                                torch.cat(val_prob).numpy(), num_classes)
        val_acc = val_m["accuracy"]
        # train F1 from the argmax predictions collected during the epoch
        # (dropout ON, weights changing mid-epoch -> a rough number, like train_acc)
        train_f1 = macro_f1(torch.cat(train_true), torch.cat(train_pred), num_classes)

        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"train_f1={train_f1:.3f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"val_f1={val_m['f1_macro']:.3f}  val_roc_auc={val_m['roc_auc_macro']:.3f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "train_f1_macro": train_f1,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_precision_macro": val_m["precision_macro"],
            "val_recall_macro": val_m["recall_macro"],
            "val_f1_macro": val_m["f1_macro"],
            "val_roc_auc_macro": val_m["roc_auc_macro"],
            "val_pr_auc_macro": val_m["pr_auc_macro"],
        })
        # rewritten every epoch on purpose: if training crashes or gets killed
        # partway through, you still have every completed epoch on disk, not zero.
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(classifier.state_dict(), ckpt_path)  # save only if val loss improved
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"no improvement for {PATIENCE} epochs, stopping early")
                break
        # ---- loop back to top for "next epoch" ----

    elapsed = time.time() - start_time
    summary_path.write_text(json.dumps({
        "epochs_run": len(history),
        "epochs_planned": NUM_EPOCHS,
        "stopped_early": len(history) < NUM_EPOCHS,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": history[best_epoch - 1]["val_acc"],
        "best_val_f1_macro": history[best_epoch - 1]["val_f1_macro"],
        "best_val_roc_auc_macro": history[best_epoch - 1]["val_roc_auc_macro"],
        "wall_clock_seconds": round(elapsed, 1),
        "device": device,
        "seed": args.seed,
        "weight_decay": args.weight_decay,
    }, indent=2))

    print(f"\nTraining finished in {elapsed:.1f}s ({len(history)} epochs run). "
          f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}. "
          f"Weights saved to {ckpt_path}, history saved to {history_path}")


if __name__ == "__main__":
    main()
