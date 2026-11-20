import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from transformers import CLIPModel, CLIPTokenizer

from common import find_cifar_root, get_device

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_raw(dataset, root):
    if dataset == "cifar10":
        cls = torchvision.datasets.CIFAR10
        root = find_cifar_root(root)
    else:
        cls = torchvision.datasets.CIFAR100
    train = cls(root=root, train=True, download=True)
    test = cls(root=root, train=False, download=True)
    return train.data, np.array(train.targets), test.data, np.array(test.targets), train.classes


def as_tensor(output):
    if torch.is_tensor(output):
        return output
    if getattr(output, "pooler_output", None) is not None:
        return output.pooler_output
    return output[0]


@torch.no_grad()
def image_embeddings(model, images, device, batch_size):
    dtype = torch.float16 if device != "cpu" else torch.float32
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i:i + batch_size]).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False).clamp(0, 1)
        x = ((x - mean) / std).to(dtype)
        e = as_tensor(model.get_image_features(pixel_values=x)).float()
        out.append(F.normalize(e, dim=1).cpu())
        if (i // batch_size) % 20 == 0:
            print(f"  {i + len(x)}/{len(images)}", flush=True)
    return torch.cat(out).numpy()


@torch.no_grad()
def text_embeddings(model, tokenizer, classes, device):
    prompts = [f"a photo of a {c.replace('_', ' ')}." for c in classes]
    tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    e = as_tensor(model.get_text_features(**tokens)).float()
    return F.normalize(e, dim=1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    p.add_argument("--model", default="openai/clip-vit-base-patch16")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="")
    args = p.parse_args()
    out = Path(args.out or f"results/embeddings/{args.dataset}_clip_b16.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    x_train, y_train, x_test, y_test, classes = load_raw(args.dataset, args.data_dir)
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = CLIPModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.model)

    print("train images")
    train_emb = image_embeddings(model, x_train, device, args.batch_size)
    print("test images")
    test_emb = image_embeddings(model, x_test, device, args.batch_size)
    text_emb = text_embeddings(model, tokenizer, classes, device)

    zero_shot = (test_emb @ text_emb.T).argmax(1)
    print(f"CLIP zero-shot test accuracy: {(zero_shot == y_test).mean():.3f}")
    np.savez_compressed(out, train_emb=train_emb.astype(np.float16), test_emb=test_emb.astype(np.float16),
                        train_labels=y_train, test_labels=y_test, text_emb=text_emb.astype(np.float16),
                        classes=np.array(classes))
    print("saved", out)


if __name__ == "__main__":
    main()
