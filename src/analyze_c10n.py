import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import noise_name

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def collect(dirs):
    rows = []
    for d in dirs:
        cfg_path, sum_path = d / "config.json", d / "summary.json"
        if not (cfg_path.exists() and sum_path.exists()):
            continue
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("method", "ce") != "ce" or cfg.get("filter_from") or cfg.get("noise_type", "none") == "none":
            continue
        s = json.loads(sum_path.read_text())
        rows.append({"noise": noise_name(cfg), "kind": cfg["noise_type"], "seed": cfg["seed"],
                     "actual_noise": cfg["actual_noise_rate"], "final_test": s["final_test_acc"],
                     "es_clean_val": s["es_clean_val_test_acc"], "es_noisy_val": s["es_noisy_val_test_acc"],
                     "memorized": s["final_noisy_memorized"]})
    return pd.DataFrame(rows)


def confusion(labels_path):
    lab = np.load(labels_path)
    m = np.zeros((10, 10))
    np.add.at(m, (lab["y_true"], lab["y_noisy"]), 1)
    return m / m.sum(1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage5", default="results/runs/stage5")
    p.add_argument("--synthetic", default="results/runs")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()
    s5 = Path(args.stage5)

    dirs = list((s5 / "rates").glob("*")) + list((s5 / "methods").glob("ce_*")) + \
        list(Path(args.synthetic).glob("*_seed0")) + list((Path(args.synthetic) / "stage3").glob("ce_*"))
    df = collect(dirs).drop_duplicates(["noise", "seed"])
    table = df.groupby(["kind", "noise"]).agg(
        actual_noise=("actual_noise", "mean"), final_test=("final_test", "mean"),
        es_clean_val=("es_clean_val", "mean"), es_noisy_val=("es_noisy_val", "mean"),
        memorized=("memorized", "mean"), n_seeds=("seed", "count")).sort_values("actual_noise")
    pd.set_option("display.width", 200)
    print((table.drop(columns="n_seeds") * 100).round(2).join(table.n_seeds))
    table.to_csv(Path(args.out_dir) / "stage5_vs_synthetic.csv")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for kind, marker in [("sym", "o"), ("asym", "s"), ("c10n", "*")]:
        t = table.loc[kind] if kind in table.index.get_level_values(0) else None
        if t is None:
            continue
        ax.plot(t.actual_noise * 100, t.final_test * 100, marker=marker, markersize=10 if kind == "c10n" else 6,
                linestyle="-" if kind != "c10n" else "none", label=kind)
        if kind == "c10n":
            for name, r in t.iterrows():
                ax.annotate(name.replace("c10n-", ""), (r.actual_noise * 100, r.final_test * 100),
                            textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("share of wrong training labels, %")
    ax.set_ylabel("test accuracy (CE, last epoch), %")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(args.out_dir) / "stage5_vs_synthetic.png", dpi=120)

    label_runs = {"worse": s5 / "methods" / "ce_c10n-worse_seed0", "aggre": s5 / "rates" / "ce_c10n-aggre_seed0",
                  "random1": s5 / "rates" / "ce_c10n-random1_seed0"}
    label_runs = {k: v for k, v in label_runs.items() if (v / "labels.npz").exists()}
    fig, axes = plt.subplots(1, len(label_runs), figsize=(6 * len(label_runs), 5.2), squeeze=False)
    for ax, (name, d) in zip(axes[0], label_runs.items()):
        m = confusion(d / "labels.npz")
        ax.imshow(m, cmap="viridis", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if m[i, j] >= 0.02:
                    ax.text(j, i, f"{m[i, j] * 100:.0f}", ha="center", va="center", fontsize=7,
                            color="black" if m[i, j] > 0.5 else "white")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASSES, rotation=60, fontsize=8)
        ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel("human label")
        ax.set_ylabel("true class")
        ax.set_title(f"CIFAR-10N {name}: P(label | class), %")
    plt.tight_layout()
    plt.savefig(Path(args.out_dir) / "stage5_confusion.png", dpi=120)


if __name__ == "__main__":
    main()
