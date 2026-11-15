import torch
import torch.nn.functional as F


def symmetric_cross_entropy(logits, y, alpha=0.1, beta=1.0, num_classes=10):
    logits = logits.float()
    ce = F.cross_entropy(logits, y)
    pred = F.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
    one_hot = F.one_hot(y, num_classes).float().clamp(min=1e-4, max=1.0)
    rce = -(pred * torch.log(one_hot)).sum(dim=1).mean()
    return alpha * ce + beta * rce


def make_loss_fn(args):
    if args.method == "ls":
        return lambda logits, y: F.cross_entropy(logits, y, label_smoothing=args.ls_eps)
    if args.method == "sce":
        return lambda logits, y: symmetric_cross_entropy(logits, y, args.sce_alpha, args.sce_beta)
    return lambda logits, y: F.cross_entropy(logits, y)
