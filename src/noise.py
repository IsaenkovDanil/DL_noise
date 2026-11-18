import urllib.request
from pathlib import Path

import numpy as np
import torch

CIFAR10N_URL = "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-10_human.pt"

ASYM_MAP_CIFAR10 = {9: 1, 2: 0, 4: 7, 3: 5, 5: 3}


def symmetric_noise(y, rate, num_classes, rng):
    y_noisy = y.copy()
    n_flip = int(round(rate * len(y)))
    idx = rng.choice(len(y), size=n_flip, replace=False)
    shift = rng.integers(1, num_classes, size=n_flip)
    y_noisy[idx] = (y[idx] + shift) % num_classes
    return y_noisy


def asymmetric_noise(y, rate, rng, mapping=ASYM_MAP_CIFAR10):
    y_noisy = y.copy()
    for src, dst in mapping.items():
        idx = np.where(y == src)[0]
        n_flip = int(round(rate * len(idx)))
        chosen = rng.choice(idx, size=n_flip, replace=False)
        y_noisy[chosen] = dst
    return y_noisy


def make_noisy_labels(y, kind, rate, num_classes, seed):
    y = np.asarray(y)
    if kind == "none" or rate == 0:
        return y.copy()
    rng = np.random.default_rng(seed)
    if kind == "sym":
        return symmetric_noise(y, rate, num_classes, rng)
    if kind == "asym":
        return asymmetric_noise(y, rate, rng)
    raise ValueError(f"unknown noise type: {kind}")


def load_cifar10n(path, url=CIFAR10N_URL):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    raw = torch.load(path, weights_only=False)
    return {k: np.asarray(v).astype(np.int64) for k, v in raw.items()}
