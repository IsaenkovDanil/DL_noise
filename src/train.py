import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from filtering import keep_mask
from common import augment, evaluate, get_device, load_cifar10, make_model, predict, set_seed
from losses import make_loss_fn
from noise import make_noisy_labels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-type", choices=["none", "sym", "asym"], default="none")
    p.add_argument("--noise-rate", type=float, default=0.0)
    p.add_argument("--method", choices=["ce", "ls", "mixup", "sce"], default="ce")
    p.add_argument("--ls-eps", type=float, default=0.1)
    p.add_argument("--mixup-alpha", type=float, default=1.0)
    p.add_argument("--sce-alpha", type=float, default=0.1)
    p.add_argument("--sce-beta", type=float, default=1.0)
    p.add_argument("--save-train-loss", action="store_true")
    p.add_argument("--filter-from", default="")
    p.add_argument("--filter-epoch", type=int, default=10)
    p.add_argument("--drop-frac", type=float, default=0.4)
    p.add_argument("--filter-oracle", action="store_true")
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results/runs/baseline_seed0")
    return p.parse_args()


def add_noise(data, args):
    y_train_true = data["train_y"].cpu().numpy()
    y_val_true = data["val_y"].cpu().numpy()
    y_train_noisy = make_noisy_labels(y_train_true, args.noise_type, args.noise_rate, 10, 1000 + args.seed)
    y_val_noisy = make_noisy_labels(y_val_true, args.noise_type, args.noise_rate, 10, 2000 + args.seed)
    return y_train_true, y_train_noisy, y_val_noisy


def train_fit_stats(model, x, y_true, y_noisy, is_noisy, device):
    logits = predict(model, x, device)
    pred = logits.argmax(1)
    per_sample_loss = F.cross_entropy(logits, y_noisy, reduction="none")
    clean = ~is_noisy
    stats = {"train_clean_acc": (pred[clean] == y_true[clean]).float().mean().item()}
    if is_noisy.any():
        stats["noisy_memorized"] = (pred[is_noisy] == y_noisy[is_noisy]).float().mean().item()
        stats["noisy_true_pred"] = (pred[is_noisy] == y_true[is_noisy]).float().mean().item()
    else:
        stats["noisy_memorized"] = float("nan")
        stats["noisy_true_pred"] = float("nan")
    return stats, per_sample_loss


def main():
    args = parse_args()
    device = get_device()
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_cifar10(args.data_dir, device)
    y_train_true, y_train_noisy, y_val_noisy = add_noise(data, args)
    keep = np.ones(len(y_train_true), dtype=bool)
    filter_stats = {}
    if args.filter_from:
        y_train_noisy, keep, filter_stats = keep_mask(args.filter_from, args.filter_epoch,
                                                      args.drop_frac, args.filter_oracle)
        print("filter:", filter_stats)
    np.savez(out / "labels.npz", train_idx=data["train_idx"].cpu().numpy(),
             y_true=y_train_true, y_noisy=y_train_noisy, keep=keep)
    actual_rate = float((y_train_true != y_train_noisy).mean())
    (out / "config.json").write_text(json.dumps(
        vars(args) | {"device": device, "actual_noise_rate": actual_rate, "filter": filter_stats}, indent=2))

    x_train = data["train_x"]
    y_true = data["train_y"]
    y_train = torch.from_numpy(y_train_noisy).to(device)
    val_noisy = torch.from_numpy(y_val_noisy).to(device)
    keep_t = torch.from_numpy(keep).to(device)
    x_train, y_true, y_train = x_train[keep_t], y_true[keep_t], y_train[keep_t]
    if args.limit_train:
        x_train, y_true, y_train = (t[:args.limit_train] for t in (x_train, y_true, y_train))
    is_noisy = y_train != y_true
    n = x_train.size(0)
    steps_per_epoch = (n + args.batch_size - 1) // args.batch_size
    print(f"device={device} train={n} method={args.method} noise={args.noise_type}:{args.noise_rate} "
          f"actual={is_noisy.float().mean().item():.3f}")

    model = make_model(10).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=True, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=0.15, anneal_strategy="linear")
    use_amp = device != "cpu"
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    loss_fn = make_loss_fn(args)

    log = []
    train_losses = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=device)
        loss_sum, correct = 0.0, 0
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            xb = augment(x_train[idx]).to(memory_format=torch.channels_last)
            yb = y_train[idx]
            if args.method == "mixup":
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                partner = torch.randperm(yb.numel(), device=device)
                xb = lam * xb + (1 - lam) * xb[partner]
                yb_partner = yb[partner]
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
                if args.method == "mixup":
                    loss = lam * F.cross_entropy(logits, yb) + (1 - lam) * F.cross_entropy(logits, yb_partner)
                else:
                    loss = loss_fn(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * yb.numel()
            correct += (logits.argmax(1) == yb).sum().item()

        val_logits = predict(model, data["val_x"], device)
        test_loss, test_acc = evaluate(model, data["test_x"], data["test_y"], device)
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": loss_sum / n,
            "train_acc": correct / n,
            "val_loss": F.cross_entropy(val_logits, data["val_y"]).item(),
            "val_acc": (val_logits.argmax(1) == data["val_y"]).float().mean().item(),
            "val_noisy_loss": F.cross_entropy(val_logits, val_noisy).item(),
            "val_noisy_acc": (val_logits.argmax(1) == val_noisy).float().mean().item(),
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        stats, per_sample_loss = train_fit_stats(model, x_train, y_true, y_train, is_noisy, device)
        row |= stats
        if args.save_train_loss:
            train_losses.append(per_sample_loss.cpu().numpy().astype(np.float16))
            np.save(out / "train_losses.npy", np.stack(train_losses))
        row["sec"] = time.time() - t0
        log.append(row)
        pd.DataFrame(log).to_csv(out / "log.csv", index=False)
        print(f"ep {epoch:3d} | train {row['train_acc']:.3f} | clean-fit {row['train_clean_acc']:.3f} | "
              f"noisy-memo {row['noisy_memorized']:.3f} | val {row['val_acc']:.3f} | "
              f"val-noisy {row['val_noisy_acc']:.3f} | test {test_acc:.3f} | {row['sec']:.0f}s", flush=True)

    df = pd.DataFrame(log)
    best_clean = df.loc[df.val_acc.idxmax()]
    best_noisy = df.loc[df.val_noisy_acc.idxmax()]
    summary = {
        "final_test_acc": float(df.test_acc.iloc[-1]),
        "best_test_acc_oracle": float(df.test_acc.max()),
        "es_clean_val_epoch": int(best_clean.epoch),
        "es_clean_val_test_acc": float(best_clean.test_acc),
        "es_noisy_val_epoch": int(best_noisy.epoch),
        "es_noisy_val_test_acc": float(best_noisy.test_acc),
        "final_noisy_memorized": float(df.noisy_memorized.iloc[-1]),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
