"""MNIST loading.

Images stay in [0, 1] pixel space (ToTensor only). Normalization happens inside the
model, so attacks perturb real pixels and the epsilon budget means the same thing for
every model (eps = 0.1 -> each pixel moves by at most 10% of its range).

TRAIN split -> training (55k) + validation (5k). TEST split -> only used in test.py.
"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

TO_TENSOR = transforms.ToTensor()


def get_train_val_loaders(batch_size=128, val_size=5000, root="./data", seed=42):
    full = datasets.MNIST(root, train=True, download=True, transform=TO_TENSOR)
    idx = torch.randperm(len(full), generator=torch.Generator().manual_seed(seed))
    train_ds, val_ds = Subset(full, idx[val_size:]), Subset(full, idx[:val_size])
    kw = dict(num_workers=2, pin_memory=torch.cuda.is_available())
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw),
            DataLoader(val_ds, batch_size=512, shuffle=False, **kw))


def get_test_loader(batch_size=500, n=None, root="./data"):
    test_ds = datasets.MNIST(root, train=False, download=True, transform=TO_TENSOR)
    if n is not None and n < len(test_ds):
        test_ds = Subset(test_ds, range(n))  # first n test images, deterministic
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
