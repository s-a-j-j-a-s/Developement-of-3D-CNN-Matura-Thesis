"""
Builds leave-one-match-out folds from the already extracted features.

Why: in the random split, clips from the same match end up in train, val AND
test. The model therefore always sees the same hall, camera angle and teams
during training as at test time, so a high test score may partly reflect
match-specific context rather than volleyball technique. This script holds out
one entire match at a time to measure how well the classifier transfers to a
match it has never seen.

Nothing is re-extracted -- this only regroups the rows of the existing
data/{train,val,test}_features.csv files. The backbone never runs here.

Two rules keep the folds honest:
  1. Augmented copies (<group_id>_augN.npy) belong ONLY in training. val and
     test contain the deterministic center crop of each clip, exactly like the
     original val/test splits.
  2. A source clip goes entirely into one split, with all of its copies. If a
     crop of a clip were in train and another crop of the SAME clip in val, the
     scores would be far too optimistic.

Run:
    python make_match_folds.py
"""
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

SPLIT_CSVS = [Path("data/train_features.csv"), Path("data/val_features.csv"),
              Path("data/test_features.csv")]
OUT_ROOT = Path("data/folds")
VAL_FRACTION = 0.15   # share of source clips (per class) used for early stopping
SEED = 0              # fixed, so the folds are reproducible

# 1074 has no reception clips, so a fold held out on it cannot be scored on all
# three classes. It stays in the training data of every other fold.
SKIP_AS_HELDOUT = {"1074"}

AUG_SUFFIX = re.compile(r"_aug\d+$")


def parse_row(feature_path: str):
    """'data\\features\\train\\2026-07-09T20-46-18_3815_1_aug2.npy'
       -> group_id '2026-07-09T20-46-18_3815_1', match '3815', augmented True"""
    stem = Path(feature_path.replace("\\", "/")).stem   # filename without .npy
    augmented = bool(AUG_SUFFIX.search(stem))
    group_id = AUG_SUFFIX.sub("", stem)                 # strip _augN if present
    match_id = group_id.split("_")[1]                   # <timestamp>_<match>_<n>
    return group_id, match_id, augmented


def load_rows():
    rows = []
    for csv_path in SPLIT_CSVS:
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                group_id, match_id, augmented = parse_row(r["feature_path"])
                rows.append({"feature_path": r["feature_path"], "label": r["label"],
                             "group_id": group_id, "match_id": match_id,
                             "augmented": augmented})
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_path", "label"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"feature_path": r["feature_path"], "label": r["label"]})


def main():
    rows = load_rows()
    matches = sorted({r["match_id"] for r in rows})
    print(f"{len(rows)} feature rows, "
          f"{len({r['group_id'] for r in rows})} source clips, "
          f"matches: {', '.join(matches)}\n")

    for heldout in matches:
        if heldout in SKIP_AS_HELDOUT:
            print(f"fold {heldout}: skipped (not all classes present)\n")
            continue

        rng = random.Random(SEED)   # same seed per fold -> reproducible

        test_rows = [r for r in rows if r["match_id"] == heldout and not r["augmented"]]
        rest = [r for r in rows if r["match_id"] != heldout]

        # choose validation clips per class, so every class is represented
        clips_by_class = defaultdict(set)
        for r in rest:
            clips_by_class[r["label"]].add(r["group_id"])

        val_clips = set()
        for label, clips in clips_by_class.items():
            ordered = sorted(clips)                 # sort first -> deterministic
            rng.shuffle(ordered)
            n_val = max(1, round(len(ordered) * VAL_FRACTION))
            val_clips.update(ordered[:n_val])

        val_rows = [r for r in rest if r["group_id"] in val_clips and not r["augmented"]]
        train_rows = [r for r in rest if r["group_id"] not in val_clips]

        # a clip must never appear in two splits
        g_train = {r["group_id"] for r in train_rows}
        g_val = {r["group_id"] for r in val_rows}
        g_test = {r["group_id"] for r in test_rows}
        assert not (g_train & g_val), f"fold {heldout}: clip in train AND val"
        assert not (g_train & g_test), f"fold {heldout}: clip in train AND test"
        assert not (g_val & g_test), f"fold {heldout}: clip in val AND test"
        assert not any(r["augmented"] for r in val_rows + test_rows), \
            f"fold {heldout}: augmented copy in val/test"

        out_dir = OUT_ROOT / heldout
        write_csv(out_dir / "train_features.csv", train_rows)
        write_csv(out_dir / "val_features.csv", val_rows)
        write_csv(out_dir / "test_features.csv", test_rows)

        def per_class(rs):
            counts = defaultdict(int)
            for r in rs:
                counts[r["label"]] += 1
            return "  ".join(f"{k}={counts[k]}" for k in sorted(counts))

        print(f"fold {heldout}  (held-out match {heldout})")
        print(f"  train {len(train_rows):>5} rows / {len(g_train):>3} clips   {per_class(train_rows)}")
        print(f"  val   {len(val_rows):>5} rows / {len(g_val):>3} clips   {per_class(val_rows)}")
        print(f"  test  {len(test_rows):>5} rows / {len(g_test):>3} clips   {per_class(test_rows)}")
        print()

    print(f"written to {OUT_ROOT}/<match>/")


if __name__ == "__main__":
    main()
