import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_runs(runs_dir, seed):
    rows = []
    for d in sorted(Path(runs_dir).glob(f"*_seed{seed}")):
        cfg_path, log_path = d / "config.json", d / "log.csv"
        if not (cfg_path.exists() and log_path.exists()):
            continue
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("noise_type", "none") == "none" or "noise_rate" not in cfg:
            continue
        rows.append({"name": d.name, "type": cfg["noise_type"], "rate": cfg["noise_rate"],
                     "actual": cfg["actual_noise_rate"], "log": pd.read_csv(log_path)})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    runs = load_runs(args.runs, args.seed)
    table = pd.DataFrame([{
        "run": r["name"], "type": r["type"], "rate": r["rate"], "actual_noise": r["actual"],
        "final_test": r["log"].test_acc.iloc[-1], "best_test": r["log"].test_acc.max(),
        "best_epoch": int(r["log"].test_acc.idxmax()) + 1,
        "final_memorized": r["log"].noisy_memorized.iloc[-1],
    } for r in runs]).sort_values(["type", "rate"])
    print(table.to_string(index=False))
    table.to_csv(Path(args.out_dir) / "stage2_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    for kind, marker in [("sym", "o"), ("asym", "s")]:
        t = table[table.type == kind]
        if len(t):
            ax.plot(t.rate * 100, t.final_test * 100, marker=marker, label=f"{kind}: final epoch")
            ax.plot(t.rate * 100, t.best_test * 100, marker=marker, linestyle="--", label=f"{kind}: best epoch")
    ax.set_xlabel("noise rate, %")
    ax.set_ylabel("test accuracy, %")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(args.out_dir) / "stage2_acc_vs_noise.png", dpi=120)

    noisy_runs = [r for r in runs if r["rate"] > 0]
    fig, axes = plt.subplots(1, len(noisy_runs), figsize=(4 * len(noisy_runs), 3.6), squeeze=False)
    for ax, r in zip(axes[0], noisy_runs):
        log = r["log"]
        ax.plot(log.epoch, log.test_acc, label="test acc")
        ax.plot(log.epoch, log.train_clean_acc, label="clean train: true label")
        ax.plot(log.epoch, log.noisy_memorized, label="noisy train: wrong label")
        ax.plot(log.epoch, log.noisy_true_pred, label="noisy train: true label")
        ax.set_title(f"{r['type']} {int(r['rate'] * 100)}%")
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(Path(args.out_dir) / "stage2_memorization.png", dpi=120)


if __name__ == "__main__":
    main()
