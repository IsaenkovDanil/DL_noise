import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    log = pd.read_csv(Path(args.run) / "log.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(log.epoch, log.train_acc, label="train (augmented)")
    axes[0].plot(log.epoch, log.val_acc, label="validation")
    axes[0].plot(log.epoch, log.test_acc, label="test")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(log.epoch, log.train_loss, label="train")
    axes[1].plot(log.epoch, log.val_loss, label="validation")
    axes[1].plot(log.epoch, log.test_loss, label="test")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(log.epoch, log.lr)
    axes[2].set_title("Learning rate")
    axes[2].set_xlabel("epoch")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)


if __name__ == "__main__":
    main()
