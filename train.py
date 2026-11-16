"""Train a model on the MNIST TRAIN split (optionally with PGD adversarial training).

Usage:
    python train.py --model cnn --epochs 10
    python train.py --model cnn --epochs 10 --adv-train      # optional defense
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from attacks import pgd
from data import get_train_val_loaders
from models import build_model, count_params


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "cnn", "vit"], required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--adv-train", action="store_true", help="train on PGD adversarial examples")
    p.add_argument("--adv-eps", type=float, default=0.3)
    p.add_argument("--adv-steps", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    lr = args.lr or {"mlp": 1e-3, "cnn": 2e-3, "vit": 1e-3}[args.model]
    run = args.model + ("_advtrain" if args.adv_train else "")

    train_loader, val_loader = get_train_val_loaders(args.batch_size)
    model = build_model(args.model).to(device)
    print(f"[{run}] device={device} params={count_params(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=args.epochs, steps_per_epoch=len(train_loader))

    history, start = [], time.time()
    for epoch in range(1, args.epochs + 1):
        # adversarial training: ramp eps up over the first half of training, then hold it
        eps = args.adv_eps * min(1.0, epoch / max(1, args.epochs // 2)) if args.adv_train else 0.0
        loss_sum = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if args.adv_train:
                model.eval()                      # craft attacks with fixed BN/dropout
                x = pgd(model, x, y, eps, steps=args.adv_steps)
            model.train()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        val_acc = accuracy(model, val_loader, device)
        history.append(dict(epoch=epoch, train_loss=loss_sum / total, train_acc=correct / total,
                            val_acc=val_acc, adv_eps=eps))
        extra = f" (trained on PGD eps={eps:.2f})" if args.adv_train else ""
        print(f"epoch {epoch:02d} | train loss {loss_sum / total:.4f} acc {correct / total:.4f} "
              f"| clean val acc {val_acc:.4f}{extra}")

    train_time = time.time() - start
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"arch": args.model, "run": run, "state_dict": model.state_dict(),
                "args": vars(args), "val_acc": val_acc}, Path("checkpoints") / f"{run}.pt")
    Path("results").mkdir(exist_ok=True)
    (Path("results") / f"{run}_train_history.json").write_text(json.dumps(
        dict(run=run, params=count_params(model), train_time_sec=train_time,
             args=vars(args), history=history), indent=2))
    print(f"[{run}] done in {train_time:.0f}s, clean val acc {val_acc:.4f}, saved checkpoints/{run}.pt")


if __name__ == "__main__":
    main()
