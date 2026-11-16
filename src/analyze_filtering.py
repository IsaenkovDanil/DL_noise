import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from filtering import auc, drop_by_loss, load_source, precision_recall


def epoch_curves(src_dir, drop_frac):
    _, losses, is_noisy = load_source(src_dir)
    rows = []
    for e in range(losses.shape[0]):
        dropped = drop_by_loss(losses[e], drop_frac)
        p, r = precision_recall(dropped, is_noisy)
        rows.append({"epoch": e + 1, "precision": p, "recall": r, "auc": auc(losses[e], is_noisy)})
    return pd.DataFrame(rows)


def retrain_table(runs_dir, baseline_dir):
    rows = []
    for d in sorted(Path(runs_dir).glob("filtered_*")):
        cfg = json.loads((d / "config.json").read_text())
        s = json.loads((d / "summary.json").read_text())
        variant = "oracle" if cfg["filter_oracle"] else f"loss@epoch{cfg['filter_epoch']}"
        rows.append({"noise": f"{cfg['noise_type']}0.4", "variant": variant, "seed": cfg["seed"],
                     "precision": cfg["filter"]["precision"], "recall": cfg["filter"]["recall"],
                     "noise_left": cfg["filter"]["noise_left"], "test": s["final_test_acc"]})
    for d in sorted(Path(baseline_dir).glob("ce_*0.4_seed*")):
        cfg = json.loads((d / "config.json").read_text())
        s = json.loads((d / "summary.json").read_text())
        rows.append({"noise": f"{cfg['noise_type']}0.4", "variant": "no filter (stage 3 ce)",
                     "seed": cfg["seed"], "precision": np.nan, "recall": np.nan,
                     "noise_left": cfg["actual_noise_rate"], "test": s["final_test_acc"]})
    df = pd.DataFrame(rows)
    agg = df.groupby(["noise", "variant"]).agg(
        precision=("precision", "mean"), recall=("recall", "mean"), noise_left=("noise_left", "mean"),
        test_mean=("test", "mean"), test_std=("test", "std"), n_seeds=("seed", "count"))
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/runs/stage4")
    p.add_argument("--baseline", default="results/runs/stage3")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()
    runs = Path(args.runs)
    sources = {"sym0.4": (runs / "source_sym0.4_seed0", 0.4), "asym0.4": (runs / "source_asym0.4_seed0", 0.2)}
    sources = {k: v for k, v in sources.items() if (v[0] / "train_losses.npy").exists()}

    fig, axes = plt.subplots(2, len(sources), figsize=(6.5 * len(sources), 8), squeeze=False)
    for col, (name, (src, frac)) in enumerate(sources.items()):
        curves = epoch_curves(src, frac)
        curves.to_csv(Path(args.out_dir) / f"stage4_pr_by_epoch_{name}.csv", index=False)
        ax = axes[0][col]
        ax.plot(curves.epoch, curves.precision, label="precision")
        ax.plot(curves.epoch, curves.recall, label="recall")
        ax.plot(curves.epoch, curves.auc, label="ROC AUC")
        ax.set_title(f"{name}: drop top {int(frac * 100)}% by loss at epoch e")
        ax.set_xlabel("epoch e")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()

        _, losses, is_noisy = load_source(src)
        ax = axes[1][col]
        for e, style in [(min(10, losses.shape[0]), "-"), (losses.shape[0], "--")]:
            l = losses[e - 1]
            bins = np.linspace(0, np.percentile(l, 99.5), 60)
            ax.hist(l[~is_noisy], bins=bins, histtype="step", linestyle=style, density=True,
                    color="tab:blue", label=f"clean, epoch {e}")
            ax.hist(l[is_noisy], bins=bins, histtype="step", linestyle=style, density=True,
                    color="tab:red", label=f"noisy, epoch {e}")
        ax.set_title(f"{name}: per-example loss")
        ax.set_xlabel("cross-entropy w.r.t. given (noisy) label")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        print(name, "best AUC epoch:", int(curves.loc[curves.auc.idxmax(), "epoch"]),
              "AUC@10=%.3f AUC@last=%.3f" % (curves.auc.iloc[min(9, len(curves) - 1)], curves.auc.iloc[-1]))
    plt.tight_layout()
    plt.savefig(Path(args.out_dir) / "stage4_filtering.png", dpi=120)

    if list(runs.glob("filtered_*")):
        agg = retrain_table(runs, args.baseline)
        pd.set_option("display.width", 200)
        print((agg.drop(columns="n_seeds") * 100).round(2).join(agg.n_seeds))
        agg.to_csv(Path(args.out_dir) / "stage4_table.csv")


if __name__ == "__main__":
    main()
