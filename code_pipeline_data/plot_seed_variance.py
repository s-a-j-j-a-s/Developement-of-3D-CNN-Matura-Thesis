"""
Validation loss of the same configuration trained with three different random
seeds -- the figure behind the statement that the seed matters more than the
hyperparameters.

All three runs use identical settings (weight decay 0, LR 1e-3, dropout 0.5);
only the seed differs, which changes the weight initialisation, the dropout
masks and the order of the batches. The marked point on each curve is the
epoch with the lowest validation loss, i.e. the checkpoint that early stopping
kept.

Reads   data/runs/wd0_s<seed>/training_history.csv
Writes  data/eval/seed_variance.png and .pdf   (use the PDF in LaTeX)

Run:
    python plot_seed_variance.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [0, 1, 2]
RUN_DIR = Path("data/runs")
OUT_STEM = Path("data/eval/seed_variance")

# colour AND line style differ, so the curves stay distinguishable in
# greyscale print and for colour-vision deficiencies
STYLES = [("#4C72A8", "-"), ("#DD8452", "--"), ("#6F4C9B", "-.")]


def read_history(seed: int):
    path = RUN_DIR / f"wd0_s{seed}" / "training_history.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run the seed runs first")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    return epochs, val_loss


def main():
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    for seed, (colour, style) in zip(SEEDS, STYLES):
        epochs, val_loss = read_history(seed)
        best_i = min(range(len(val_loss)), key=val_loss.__getitem__)
        ax.plot(epochs, val_loss, style, color=colour, linewidth=1.8,
                label=f"seed {seed} (best: epoch {epochs[best_i]}, "
                      f"{val_loss[best_i]:.3f})")
        # mark the checkpoint early stopping kept
        ax.plot(epochs[best_i], val_loss[best_i], "o", color=colour,
                markersize=7, markeredgecolor="white", markeredgewidth=1.2,
                zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("Same configuration, three random seeds", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")

    fig.tight_layout()
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=200)
    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved -> {OUT_STEM.with_suffix('.png')} and {OUT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
