import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from common import augment, evaluate, get_device, load_cifar10, make_model, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results/runs/baseline_seed0")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args) | {"device": device}, indent=2))

    data = load_cifar10(args.data_dir, device)
    x_train, y_train = data["train_x"], data["train_y"]
    if args.limit_train:
        x_train, y_train = x_train[:args.limit_train], y_train[:args.limit_train]
    n = x_train.size(0)
    steps_per_epoch = (n + args.batch_size - 1) // args.batch_size
    print(f"device={device} train={n} val={data['val_y'].numel()} test={data['test_y'].numel()}")

    model = make_model(10).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=True, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=0.15, anneal_strategy="linear")
    use_amp = device != "cpu"
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    log = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=device)
        loss_sum, correct = 0.0, 0
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            xb = augment(x_train[idx]).to(memory_format=torch.channels_last)
            yb = y_train[idx]
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * yb.numel()
            correct += (logits.argmax(1) == yb).sum().item()

        val_loss, val_acc = evaluate(model, data["val_x"], data["val_y"], device)
        test_loss, test_acc = evaluate(model, data["test_x"], data["test_y"], device)
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": loss_sum / n,
            "train_acc": correct / n,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "sec": time.time() - t0,
        }
        log.append(row)
        pd.DataFrame(log).to_csv(out / "log.csv", index=False)
        print(f"ep {epoch:3d} | train {row['train_acc']:.3f} | val {val_acc:.3f} | "
              f"test {test_acc:.3f} | {row['sec']:.0f}s", flush=True)

    summary = {"final_test_acc": log[-1]["test_acc"], "final_val_acc": log[-1]["val_acc"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
