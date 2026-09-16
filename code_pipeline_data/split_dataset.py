"""
Stage 2: split the manifest into train / val / test manifests.

Splitting happens by `group_id`, not by row. In trim_and_convert.py,
group_id defaults to the filename stem (so today it's one group per
clip) -- but if you ever have several clips coming from the same
recording session or subject, set group_id upstream to that shared
session/subject id so all of a group's clips land in the same split.
Otherwise you'll leak near-duplicate footage across train and test.

The split is stratified per class: each class's groups are shuffled
(seeded) and cut at the requested fractions, so class balance is
preserved across train/val/test.

Run:
    python split_dataset.py --manifest data/manifest.csv --out_dir data
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def split_groups(group_ids: list[str], train_frac: float, val_frac: float,
                  seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    ids = list(group_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    # whatever's left goes to test, so fractions don't need to sum exactly to 1
    n_val = min(n_val, n - n_train)

    assignment = {}
    for gid in ids[:n_train]:
        assignment[gid] = "train"
    for gid in ids[n_train:n_train + n_val]:
        assignment[gid] = "val"
    for gid in ids[n_train + n_val:]:
        assignment[gid] = "test"
    return assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("data"))
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if not rows:
        raise SystemExit(f"no rows found in {args.manifest}")

    # unique group_ids per class
    groups_by_class: dict[str, set] = defaultdict(set)
    for row in rows:
        groups_by_class[row["label"]].add(row["group_id"])

    # assign every group_id -> split, class by class so each class is
    # split independently (this is what keeps the split stratified)
    group_to_split: dict[str, str] = {}
    for label, group_ids in groups_by_class.items():
        assignment = split_groups(sorted(group_ids), args.train_frac,
                                   args.val_frac, seed=args.seed)
        group_to_split.update(assignment)

    out_rows = {"train": [], "val": [], "test": []}
    for row in rows:
        split = group_to_split[row["group_id"]]
        out_rows[split].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = rows[0].keys()
    for split, split_rows in out_rows.items():
        out_path = args.out_dir / f"{split}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)

    print("Split sizes (clips):")
    for split, split_rows in out_rows.items():
        counts = defaultdict(int)
        for row in split_rows:
            counts[row["label"]] += 1
        counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {split:5s}: {len(split_rows):4d} total  ({counts_str})")


if __name__ == "__main__":
    main()
