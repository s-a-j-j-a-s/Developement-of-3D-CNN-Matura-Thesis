"""
Bar chart of the leave-one-match-out results: one bar per held-out match,
error bars = standard deviation over the seeds, with the random-split test
result drawn in as a reference line.

Reads the metrics.json files that evaluate.py wrote for the folds
(data/folds/<match>/eval_s<seed>/metrics.json) and writes
data/eval/match_folds.png (for viewing) and .pdf (vector, for LaTeX).

Run after the fold loop:
    python plot_match_folds.py
"""
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt

FOLD_GLOB = "data/folds/*/eval_s*/metrics.json"
OUT_STEM = Path("data/eval/match_folds")

# random-split test result (mean +/- SD over seeds 0-2), for the reference line
RANDOM_SPLIT_MEAN = 0.919
RANDOM_SPLIT_SD = 0.034

BAR_COLOR = "#4C72A8"     # one series -> one hue, no legend needed
ERROR_COLOR = "#2B2B2B"
REFERENCE_COLOR = "#8A8A8A"


def collect():
    """{match_id: {"accs": [...], "n": clips}} from the fold metrics files."""
    per_match = defaultdict(lambda: {"accs": [], "n": 0})
    for path in sorted(Path(".").glob(FOLD_GLOB)):
        match_id = re.search(r"folds[\\/](\w+)[\\/]", str(path)).group(1)
        data = json.loads(path.read_text())
        per_match[match_id]["accs"].append(data["accuracy"])
        per_match[match_id]["n"] = data["n_clips"]
    return dict(sorted(per_match.items()))


def main():
    per_match = collect()
    if not per_match:
        raise SystemExit(f"no metrics found under {FOLD_GLOB} -- run the fold loop first")

    labels, means, sds = [], [], []
    for match_id, d in per_match.items():
        labels.append(f"{match_id}\n(n={d['n']})")
        means.append(st.mean(d["accs"]))
        sds.append(st.stdev(d["accs"]) if len(d["accs"]) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    # reference band first, so the bars sit on top of it
    ax.axhspan(RANDOM_SPLIT_MEAN - RANDOM_SPLIT_SD, RANDOM_SPLIT_MEAN + RANDOM_SPLIT_SD,
               color=REFERENCE_COLOR, alpha=0.15, zorder=0)
    ax.axhline(RANDOM_SPLIT_MEAN, color=REFERENCE_COLOR, linestyle="--", linewidth=1.5,
               zorder=1,
               label=f"Random split: {RANDOM_SPLIT_MEAN:.3f} ± {RANDOM_SPLIT_SD:.3f}")

    bars = ax.bar(labels, means, yerr=sds, capsize=4, color=BAR_COLOR, width=0.62,
                  zorder=2, error_kw={"ecolor": ERROR_COLOR, "elinewidth": 1.2},
                  label="Held-out match (mean ± SD, 3 seeds)")

    # value on top of each bar -- only 5 bars, so labelling all of them is fine
    for bar, mean, sd in zip(bars, means, sds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + sd + 0.02, f"{mean:.3f}",
                ha="center", va="bottom", fontsize=9, color="#2B2B2B")

    # headroom above the bars for the legend, so nothing overlaps the data
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Accuracy on the held-out match")
    ax.set_xlabel("Held-out match")
    ax.set_title("Leave-one-match-out: generalisation to unseen matches", fontsize=11)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=9,
              bbox_to_anchor=(0.5, 1.02))
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999999")

    fig.tight_layout()
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=200)
    fig.savefig(OUT_STEM.with_suffix(".pdf"))   # vector version for LaTeX
    plt.close(fig)

    overall = [a for d in per_match.values() for a in d["accs"]]
    print(f"{len(per_match)} folds, {len(overall)} runs, "
          f"overall accuracy {st.mean(overall):.3f} +/- {st.stdev(overall):.3f}")
    print(f"saved -> {OUT_STEM.with_suffix('.png')} and {OUT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
