import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
VAL_SIZE = 5000
SPLIT_SEED = 0


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(num_classes):
    model = torchvision.models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def split_indices(n_total):
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n_total)
    return perm[VAL_SIZE:], perm[:VAL_SIZE]


def find_cifar_root(root):
    if (Path(root) / "cifar-10-batches-py").is_dir():
        return root
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        for found in kaggle.rglob("cifar-10-batches-py"):
            if (found / "data_batch_1").exists():
                return str(found.parent)
    return root


def load_cifar10(root, device):
    root = find_cifar_root(root)
    print("CIFAR-10 root:", root)
    train = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    x_all = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous()
    y_all = torch.tensor(train.targets)
    train_idx, val_idx = split_indices(len(y_all))
    data = {
        "train_x": x_all[train_idx],
        "train_y": y_all[train_idx],
        "train_idx": torch.from_numpy(train_idx),
        "val_idx": torch.from_numpy(val_idx),
        "val_x": x_all[val_idx],
        "val_y": y_all[val_idx],
        "test_x": torch.from_numpy(test.data).permute(0, 3, 1, 2).contiguous(),
        "test_y": torch.tensor(test.targets),
        "classes": train.classes,
    }
    for k, v in data.items():
        if torch.is_tensor(v):
            data[k] = v.to(device)
    return data


def normalize(x_uint8):
    x = x_uint8.float() / 255.0
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x_uint8):
    n = x_uint8.size(0)
    dev = x_uint8.device
    x = x_uint8.float()
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip.view(n, 1, 1, 1), x.flip(3), x)
    x = F.pad(x, (4, 4, 4, 4))
    di = torch.randint(0, 9, (n,), device=dev)
    dj = torch.randint(0, 9, (n,), device=dev)
    ar = torch.arange(32, device=dev)
    rows = (di.view(n, 1) + ar.view(1, 32)).view(n, 1, 32, 1)
    cols = (dj.view(n, 1) + ar.view(1, 32)).view(n, 1, 1, 32)
    batch = torch.arange(n, device=dev).view(n, 1, 1, 1)
    chan = torch.arange(3, device=dev).view(1, 3, 1, 1)
    x = x[batch, chan, rows, cols]
    return normalize(x.to(torch.uint8))


@torch.no_grad()
def predict(model, x_uint8, device, batch_size=1000):
    model.eval()
    logits = []
    for i in range(0, x_uint8.size(0), batch_size):
        xb = normalize(x_uint8[i:i + batch_size]).to(memory_format=torch.channels_last)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device != "cpu"):
            logits.append(model(xb).float())
    return torch.cat(logits)


def evaluate(model, x_uint8, y, device):
    logits = predict(model, x_uint8, device)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


def noise_name(cfg):
    if cfg.get("noise_type") == "c10n":
        return "c10n-" + cfg["c10n_label"].replace("_label", "")
    return f"{cfg.get('noise_type', 'none')}{cfg.get('noise_rate', 0.0)}"
