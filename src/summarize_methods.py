import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import noise_name

METRICS = {
    "final_test_acc": "last epoch",
    "es_clean_val_test_acc": "early stop: clean val",
    "es_noisy_val_test_acc": "early stop: noisy val",
    "best_test_acc_oracle": "oracle (best test)",
}
METHOD_ORDER = ["ce", "ls", "mixup", "sce"]


def load(runs_dir):
    rows = []
    for d in sorted(Path(runs_dir).iterdir()):
        cfg_path, sum_path = d / "config.json", d / "summary.json"
        if not (cfg_path.exists() and sum_path.exists()):
            continue
        cfg = json.loads(cfg_path.read_text())
        summary = json.loads(sum_path.read_text())
        rows.append({"method": cfg.get("method", "ce"),
                     "noise": noise_name(cfg),
                     "seed": cfg["seed"], **summary})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/runs/stage3")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--prefix", default="stage3")
    args = p.parse_args()

    df = load(args.runs)
    agg = df.groupby(["noise", "method"])[list(METRICS)].agg(["mean", "std"])
    seeds = df.groupby(["noise", "method"]).seed.count().rename("n_seeds")
    pretty = pd.DataFrame({label: [f"{agg.loc[i, (m, 'mean')] * 100:.2f} ± {agg.loc[i, (m, 'std')] * 100:.2f}"
                                   for i in agg.index] for m, label in METRICS.items()}, index=agg.index)
    pretty = pretty.join(seeds)
    pd.set_option("display.width", 200)
    print(pretty)
    pretty.to_csv(Path(args.out_dir) / f"{args.prefix}_table.csv")

    noises = sorted(df.noise.unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(7 * len(noises), 4.5), squeeze=False)
    width = 0.2
    for ax, noise in zip(axes[0], noises):
        methods = [m for m in METHOD_ORDER if m in df[df.noise == noise].method.unique()]
        x = np.arange(len(methods))
        for k, (metric, label) in enumerate(METRICS.items()):
            means = [agg.loc[(noise, m), (metric, "mean")] * 100 for m in methods]
            stds = [agg.loc[(noise, m), (metric, "std")] * 100 for m in methods]
            ax.bar(x + (k - 1.5) * width, means, width, yerr=np.nan_to_num(stds), capsize=3, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(f"noise {noise}")
        ax.set_ylabel("test accuracy, %")
        lo = min(agg.loc[noise][[(m, "mean") for m in METRICS]].min()) * 100
        ax.set_ylim(lo - 5, None)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(args.out_dir) / f"{args.prefix}_methods.png", dpi=120)

    fig, axes = plt.subplots(2, len(noises), figsize=(7 * len(noises), 7), squeeze=False)
    for col, noise in enumerate(noises):
        for method in METHOD_ORDER:
            logs = [pd.read_csv(d / "log.csv") for d in Path(args.runs).glob(f"{method}_{noise}_seed*")]
            if not logs:
                continue
            mean = pd.concat(logs).groupby("epoch").mean(numeric_only=True)
            axes[0][col].plot(mean.index, mean.test_acc * 100, label=method)
            axes[1][col].plot(mean.index, mean.noisy_memorized * 100, label=method)
        axes[0][col].set_title(f"{noise}: test accuracy, % (mean over seeds)")
        axes[1][col].set_title(f"{noise}: noisy train examples fitted to wrong label, %")
        for ax in axes[:, col]:
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(args.out_dir) / f"{args.prefix}_curves.png", dpi=120)


if __name__ == "__main__":
    main()
