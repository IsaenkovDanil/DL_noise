import numpy as np
import matplotlib.pyplot as plt
import torchvision

train = torchvision.datasets.CIFAR10(root="data", train=True, download=True)
test = torchvision.datasets.CIFAR10(root="data", train=False, download=True)

print("train:", len(train))
print("test:", len(test))
print("classes:", train.classes)
print("train.data shape:", train.data.shape)
print("dtype:", train.data.dtype, "range:", train.data.min(), "...", train.data.max())

labels = np.array(train.targets)
print("first labels:", labels[:10])
print("per class:", np.bincount(labels))

fig, axes = plt.subplots(10, 8, figsize=(8, 10))
for c in range(10):
    idx = np.where(labels == c)[0][:8]
    for j, i in enumerate(idx):
        axes[c, j].imshow(train.data[i])
        axes[c, j].axis("off")
    axes[c, 0].set_title(train.classes[c], fontsize=8, loc="left")
plt.tight_layout()
plt.savefig("results/cifar10_examples.png", dpi=120)
