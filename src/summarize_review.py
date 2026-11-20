import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from embed import load_raw
from make_review_page import BANDS

VERDICT_NAMES = {"wrong": "label wrong", "correct": "label correct", "ambiguous": "ambiguous"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verdicts", default="results/stage6_review_verdicts.csv")
    p.add_argument("--review", default="results/stage6_cifar100_review.csv")
    p.add_argument("--duplicates", default="results/stage6_cifar100_duplicates.csv")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()
    out = Path(args.out_dir)

    v = pd.read_csv(args.verdicts, dtype={"id": int})
    labels = v[v.kind == "label"].merge(pd.read_csv(args.review).drop(columns="verdict"), left_on="id", right_on="train_idx")
    counts = labels.verdict.value_counts().reindex(list(VERDICT_NAMES), fill_value=0)
    n = int(counts.sum())
    print(f"label review: {n} checked")
    for k, c in counts.items():
        print(f"  {VERDICT_NAMES[k]:14s} {c:3d}  ({c / max(n, 1):.0%})")
    print(f"  precision of flags (wrong only): {counts['wrong'] / max(n, 1):.0%}, "
          f"wrong + ambiguous: {(counts['wrong'] + counts['ambiguous']) / max(n, 1):.0%}")

    pairs = pd.read_csv(args.duplicates)
    d = v[v.kind == "dup"].merge(pairs, left_on="id", right_on="test_idx")
    rows = []
    for lo, hi in BANDS:
        in_band = pairs[(pairs.cos_sim >= lo) & (pairs.cos_sim < hi)]
        checked = d[(d.cos_sim >= lo) & (d.cos_sim < hi)]
        prec = (checked.verdict == "dup").mean() if len(checked) else np.nan
        rows.append({"band": f"{lo}-{min(hi, 1.0)}", "pairs_in_band": len(in_band), "checked": len(checked),
                     "share_duplicates": prec, "estimated_duplicates": len(in_band) * prec,
                     "label_conflicts_in_checked_dups": int(((checked.verdict == "dup") &
                                                             (checked.test_label != checked.train_label)).sum())})
    bands = pd.DataFrame(rows)
    print(bands.round(2).to_string(index=False))
    total = bands.estimated_duplicates.sum()
    print(f"estimated test images with a near-duplicate in train (sim >= {BANDS[-1][0]}): "
          f"{total:.0f} ({total / len(pairs):.1%} of test); lower bound, pairs below the last band are not counted")
    summary = pd.concat([pd.DataFrame([{"part": "labels", **{VERDICT_NAMES[k]: int(c) for k, c in counts.items()}}]),
                         bands.assign(part="duplicates")], ignore_index=True)
    summary.to_csv(out / "stage6_review_summary.csv", index=False)

    x_train, _, _, _, _ = load_raw("cifar100", args.data_dir)
    show = labels.sort_values("verdict", key=lambda s: s.map({"wrong": 0, "ambiguous": 1, "correct": 2}))
    ncols = 10
    nrows = max(1, (len(show) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.9, nrows * 2.6))
    colors = {"wrong": "tab:red", "ambiguous": "tab:orange", "correct": "tab:green"}
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, r in zip(np.atleast_1d(axes).ravel(), show.itertuples()):
        ax.imshow(x_train[r.train_idx], interpolation="bicubic")
        ax.set_title(f"#{r.train_idx}\nlabel: {r.given_name}\nkNN/CLIP: {r.knn_name}/{r.clip_name}\n"
                     f"{VERDICT_NAMES[r.verdict]}", fontsize=6.5, color=colors[r.verdict], loc="left")
    fig.suptitle("CIFAR-100: manually checked top suspects (red = label wrong, orange = ambiguous, green = correct)")
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out / "stage6_review_gallery.png", dpi=110)


if __name__ == "__main__":
    main()
