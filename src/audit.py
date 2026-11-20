import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import get_device
from embed import load_raw
from filtering import auc


def knn(query, base, k, same_set, device, chunk=2048):
    q_all = torch.from_numpy(query).to(device).float()
    b = torch.from_numpy(base).to(device).float()
    idx_out, sim_out = [], []
    for i in range(0, len(q_all), chunk):
        sims = q_all[i:i + chunk] @ b.T
        if same_set:
            rows = torch.arange(sims.size(0), device=device)
            sims[rows, rows + i] = -2.0
        s, j = sims.topk(k, dim=1)
        idx_out.append(j.cpu())
        sim_out.append(s.cpu())
    return torch.cat(idx_out).numpy(), torch.cat(sim_out).numpy()


def suspicion_scores(emb, text_emb, labels, k, device):
    nb_idx, _ = knn(emb, emb, k, True, device)
    nb_labels = labels[nb_idx]
    agree = (nb_labels == labels[:, None]).mean(1)
    knn_label = np.array([np.bincount(row, minlength=text_emb.shape[0]).argmax() for row in nb_labels])
    logits = 100.0 * emb.astype(np.float32) @ text_emb.astype(np.float32).T
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    p_given = probs[np.arange(len(labels)), labels]
    return pd.DataFrame({
        "given": labels, "knn_label": knn_label, "knn_agree": agree, "clip_label": probs.argmax(1),
        "clip_p_given": p_given, "score_knn": 1 - agree, "score_clip": 1 - p_given,
        "score": (1 - agree + 1 - p_given) / 2,
    })


def evaluate(df, is_noisy):
    n = int(is_noisy.sum())
    rows = []
    for col in ["score_knn", "score_clip", "score"]:
        top = np.argsort(-df[col].to_numpy())[:n]
        rows.append({"score": col, "auc": auc(df[col].to_numpy(), is_noisy),
                     "precision_at_noise_count": is_noisy[top].mean(),
                     "precision_top100": is_noisy[np.argsort(-df[col].to_numpy())[:100]].mean()})
    return pd.DataFrame(rows)


def gallery(images, rows, classes, path, title, ncols=10, truth=None):
    n = len(rows)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.9, nrows * 2.9))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (i, r) in zip(np.atleast_1d(axes).ravel(), rows.iterrows()):
        ax.imshow(images[i], interpolation="bicubic")
        text = (f"#{int(r.train_idx)}\nlabel: {classes[int(r.given)]}\nkNN: {classes[int(r.knn_label)]} "
                f"({r.knn_agree:.0%})\nCLIP: {classes[int(r.clip_label)]}")
        if truth is not None:
            text += f"\ntrue: {classes[truth[i]]}"
        ax.set_title(text, fontsize=6.5, loc="left")
    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.subplots_adjust(hspace=0.75)
    plt.savefig(path, dpi=110)
    plt.close(fig)


def duplicates(train_emb, test_emb, x_train, x_test, y_train, y_test, classes, out_dir, prefix, device):
    j, s = knn(test_emb, train_emb, 1, False, device)
    j, s = j[:, 0], s[:, 0]
    pix = np.abs(x_test.astype(np.float32) - x_train[j].astype(np.float32)).mean(axis=(1, 2, 3)) / 255
    pairs = pd.DataFrame({"test_idx": np.arange(len(j)), "train_idx": j, "cos_sim": s, "pixel_mae": pix,
                          "test_label": [classes[c] for c in y_test], "train_label": [classes[c] for c in y_train[j]]})
    pairs = pairs.sort_values("cos_sim", ascending=False).reset_index(drop=True)
    pairs.to_csv(out_dir / f"{prefix}_duplicates.csv", index=False)
    summary = pd.DataFrame([{"threshold": t, "test_images_with_close_train": int((pairs.cos_sim >= t).sum()),
                             "share": float((pairs.cos_sim >= t).mean()),
                             "label_conflicts": int(((pairs.cos_sim >= t) & (pairs.test_label != pairs.train_label)).sum())}
                            for t in [0.95, 0.97, 0.975, 0.98, 0.985, 0.99]])
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / f"{prefix}_duplicates_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(pairs.cos_sim, bins=100)
    ax.set_yscale("log")
    ax.set_xlabel("cosine similarity of test image to nearest train image (CLIP)")
    ax.set_ylabel("test images")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_duplicates_hist.png", dpi=120)
    plt.close(fig)

    bands = [(0.99, 1.01), (0.985, 0.99), (0.98, 0.985), (0.975, 0.98)]
    band_pairs = pd.concat([pairs[(pairs.cos_sim >= lo) & (pairs.cos_sim < hi)].sample(6, random_state=0)
                            for lo, hi in bands])
    for name, sel in [("top", pairs.head(24)), ("bands", band_pairs)]:
        fig, axes = plt.subplots(4, 12, figsize=(18, 7.2))
        for ax in axes.ravel():
            ax.axis("off")
        for k, (_, r) in enumerate(sel.iterrows()):
            a, b = axes.ravel()[2 * k], axes.ravel()[2 * k + 1]
            a.imshow(x_test[r.test_idx], interpolation="bicubic")
            b.imshow(x_train[r.train_idx], interpolation="bicubic")
            a.set_title(f"test #{r.test_idx}\n{r.test_label}\nsim {r.cos_sim:.3f}", fontsize=6.5)
            b.set_title(f"train #{r.train_idx}\n{r.train_label}\npix {r.pixel_mae:.3f}", fontsize=6.5)
        fig.suptitle(f"test (left) and nearest train image (right): {name}"
                     + (" - rows: sim 0.99+, 0.985-0.99, 0.98-0.985, 0.975-0.98" if name == "bands" else ""))
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_duplicates_{name}.png", dpi=110)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True)
    p.add_argument("--labels", default="")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--top", type=int, default=60)
    p.add_argument("--duplicates", action="store_true")
    p.add_argument("--prefix", default="stage6")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    device = get_device()

    e = np.load(args.emb)
    classes = list(e["classes"])
    dataset = "cifar10" if len(classes) == 10 else "cifar100"
    x_train, y_train, x_test, y_test, _ = load_raw(dataset, args.data_dir)
    emb, text_emb = e["train_emb"], e["text_emb"]

    truth = None
    if args.labels:
        lab = np.load(args.labels)
        idx = lab["train_idx"]
        emb, images, given, truth = emb[idx], x_train[idx], lab["y_noisy"], lab["y_true"]
    else:
        images, given = x_train, y_train

    df = suspicion_scores(emb, text_emb, given, args.k, device)
    df["train_idx"] = idx if args.labels else np.arange(len(given))
    if truth is not None:
        is_noisy = given != truth
        df["is_noisy"] = is_noisy
        ev = evaluate(df, is_noisy)
        print(ev.round(3).to_string(index=False))
        ev.to_csv(out_dir / f"{args.prefix}_detector_eval.csv", index=False)

    np.save(out_dir / f"{args.prefix}_scores.npy", df.score.to_numpy().astype(np.float32))
    ranked = df.sort_values("score", ascending=False)
    review = ranked.head(200).copy()
    review["given_name"] = [classes[c] for c in review.given]
    review["knn_name"] = [classes[c] for c in review.knn_label]
    review["clip_name"] = [classes[c] for c in review.clip_label]
    review["verdict"] = ""
    review.to_csv(out_dir / f"{args.prefix}_review.csv", index_label="row")
    gallery(images, ranked.head(args.top), classes, out_dir / f"{args.prefix}_suspects.png",
            f"top {args.top} suspected label errors (kNN k={args.k} + CLIP zero-shot)", truth=truth)

    if args.duplicates:
        duplicates(e["train_emb"], e["test_emb"], x_train, x_test, y_train, y_test, classes,
                   out_dir, args.prefix, device)


if __name__ == "__main__":
    main()
