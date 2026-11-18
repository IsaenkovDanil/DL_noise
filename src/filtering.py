from pathlib import Path

import numpy as np


def load_source(run_dir):
    run_dir = Path(run_dir)
    labels = np.load(run_dir / "labels.npz")
    losses = np.load(run_dir / "train_losses.npy").astype(np.float32)
    is_noisy = labels["y_true"] != labels["y_noisy"]
    return labels, losses, is_noisy


def drop_by_loss(losses, drop_frac):
    n_drop = int(round(drop_frac * len(losses)))
    order = np.argsort(-losses)
    dropped = np.zeros(len(losses), dtype=bool)
    dropped[order[:n_drop]] = True
    return dropped


def drop_by_loss_per_class(losses, labels, drop_frac):
    dropped = np.zeros(len(losses), dtype=bool)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        dropped[idx] = drop_by_loss(losses[idx], drop_frac)
    return dropped


def precision_recall(dropped, is_noisy):
    tp = int((dropped & is_noisy).sum())
    precision = tp / max(int(dropped.sum()), 1)
    recall = tp / max(int(is_noisy.sum()), 1)
    return precision, recall


def auc(scores, is_noisy):
    ranks = np.empty(len(scores))
    ranks[np.argsort(scores)] = np.arange(1, len(scores) + 1)
    n_pos = int(is_noisy.sum())
    n_neg = len(scores) - n_pos
    return (ranks[is_noisy].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def keep_mask(run_dir, epoch, drop_frac, oracle=False, per_class=False):
    labels, losses, is_noisy = load_source(run_dir)
    if oracle:
        dropped = is_noisy.copy()
    elif per_class:
        dropped = drop_by_loss_per_class(losses[epoch - 1], labels["y_noisy"], drop_frac)
    else:
        dropped = drop_by_loss(losses[epoch - 1], drop_frac)
    precision, recall = precision_recall(dropped, is_noisy)
    stats = {"n_dropped": int(dropped.sum()), "precision": precision, "recall": recall,
             "noise_left": float(is_noisy[~dropped].mean())}
    return labels["y_noisy"], ~dropped, stats
