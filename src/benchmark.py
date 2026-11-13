import time
import torch
import torch.nn as nn

from common import get_device, make_model


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    device = get_device()
    model = make_model(10).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 256
    x = torch.randn(batch_size, 3, 32, 32, device=device).to(memory_format=torch.channels_last)
    y = torch.randint(0, 10, (batch_size,), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device != "cpu"):
            loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    for _ in range(5):
        step()
    sync(device)
    n = 30
    t0 = time.time()
    for _ in range(n):
        step()
    sync(device)
    sec = (time.time() - t0) / n
    print(f"device={device} step={sec*1000:.0f}ms epoch~{sec*50000/batch_size:.0f}s")


if __name__ == "__main__":
    main()
