"""
Reports loss, accuracy, macro F1 and ROC-AUC on train, val, AND test using the trained classifier --
the overfitting check. Unlike plot_training_history.py, this can genuinely
be run any time after training: it doesn't need anything that was only
available while the loop was running. It just needs the three cached
*_features.csv files (from extract_features.py) and the saved checkpoint
(from train_classifier.py) -- both already sitting on disk.

Interpretation:
- train_acc >> val_acc/test_acc  -> overfitting. Expected to some degree
  even here, since train_acc is measured with dropout ON during training
  (see train_classifier.py) -- a large gap is still worth reporting as-is.
- train_acc ~= val_acc ~= test_acc -> generalizing about as well on unseen
  data as on seen data. Report this number as your headline result.
- test noticeably worse than val -> you may have tuned hyperparameters
  against val enough times that it stopped being a clean estimate. Say so.

Run:
    python training_report.py
"""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate import build_classifier
from feature_dataset import FeatureDataset
from metrics_utils import compute_metrics

CKPT_PATH = Path("data/best_classifier.pt")


def evaluate_split(features_csv: Path, classifier, criterion, class_to_idx, device):
    ds = FeatureDataset(features_csv, class_to_idx)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    total_loss = 0.0
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = classifier(features)
            total_loss += criterion(logits, labels).item() * features.size(0)
            ys.append(labels.cpu())
            preds.append(logits.argmax(dim=1).cpu())
            probs.append(torch.softmax(logits, dim=1).cpu())
    m = compute_metrics(torch.cat(ys).numpy(), torch.cat(preds).numpy(),
                        torch.cat(probs).numpy(), len(class_to_idx))
    return {"loss": total_loss / len(ds), **m, "n": len(ds)}


def main():
    class_to_idx = json.loads(Path("data/class_to_idx.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = build_classifier(len(class_to_idx)).to(device)
    classifier.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    classifier.eval()  # dropout OFF for all three -- unlike the noisy train_acc
                        # logged during training, this is a clean, comparable number
    criterion = torch.nn.CrossEntropyLoss()

    print(f"{'split':<8} {'n':>5} {'loss':>8} {'accuracy':>9} {'F1':>7} {'ROC-AUC':>8}")
    results = {}
    for split in ["train", "val", "test"]:
        path = Path(f"data/{split}_features.csv")
        if not path.exists():
            print(f"{split:<8} -- {path} not found, skipping")
            continue
        r = evaluate_split(path, classifier, criterion, class_to_idx, device)
        results[split] = r
        print(f"{split:<8} {r['n']:>5} {r['loss']:>8.4f} {r['accuracy']:>9.3f} "
              f"{r['f1_macro']:>7.3f} {r['roc_auc_macro']:>8.3f}")

    Path("data/eval").mkdir(parents=True, exist_ok=True)
    Path("data/eval/split_comparison.json").write_text(json.dumps(results, indent=2))
    print("\nsaved -> data/eval/split_comparison.json")

    if "train" in results and "test" in results:
        gap = results["train"]["accuracy"] - results["test"]["accuracy"]
        if gap > 0.25:
            print(f"\ntrain_acc - test_acc = {gap:.2f} -- large gap, likely overfitting. "
                  f"Report it honestly rather than only citing test_acc.")


if __name__ == "__main__":
    main()
