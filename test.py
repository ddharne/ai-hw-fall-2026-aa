"""Test trained models on the MNIST TEST split: recognition rate before attacks,
then attack success rate (ASR) for FGSM, I-FGSM, PGD and MI-FGSM at several eps.

ASR (untargeted) = among test images the model classifies correctly BEFORE the attack,
the fraction it misclassifies AFTER the attack.
Robust accuracy = accuracy on the attacked test set (all images).

Also measures transferability (black-box): adversarial images crafted on one model,
fed to the other models, at --transfer-eps.

Usage:
    python test.py                              # all checkpoints, full test set
    python test.py --n 2000                     # quicker: first 2000 test images
    python test.py --ckpt checkpoints/cnn.pt --eps 0.1 0.2 0.3
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from attacks import ATTACKS, ATTACK_NAMES
from data import get_test_loader
from models import build_model
from train import get_device, set_seed

OUT = Path("results")


def load_models(paths, device):
    models = {}
    for pth in paths:
        state = torch.load(pth, map_location=device)
        m = build_model(state["arch"]).to(device)
        m.load_state_dict(state["state_dict"])
        m.eval()
        models[state.get("run", pth.stem)] = m
    return models


@torch.no_grad()
def predict(model, x):
    return model(x).argmax(1)


def pct(v):
    return f"{v * 100:.2f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="*", default=None)
    p.add_argument("--attacks", nargs="*", default=list(ATTACKS), choices=list(ATTACKS))
    p.add_argument("--eps", nargs="*", type=float, default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    p.add_argument("--steps", type=int, default=10, help="iterations for I-FGSM / PGD / MI-FGSM")
    p.add_argument("--mu", type=float, default=1.0, help="MI-FGSM momentum decay")
    p.add_argument("--transfer-eps", type=float, default=0.2)
    p.add_argument("--n", type=int, default=None, help="use only the first n test images")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()
    paths = [Path(c) for c in args.ckpt] if args.ckpt else sorted(Path("checkpoints").glob("*.pt"))
    if not paths:
        raise SystemExit("No checkpoints found. Run train.py first.")
    models = load_models(paths, device)
    names = list(models)
    if args.transfer_eps not in args.eps:
        args.eps = sorted(args.eps + [args.transfer_eps])

    # Batches stay on CPU and move to the device one at a time.
    batches = list(get_test_loader(args.batch_size, args.n))
    n_total = sum(y.numel() for _, y in batches)
    OUT.mkdir(exist_ok=True)
    print(f"device={device}  test images={n_total}  models={names}")

    # ---- 1. recognition rate before attacks ------------------------------------------
    clean_correct = {m: [] for m in names}
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        for m in names:
            clean_correct[m].append((predict(models[m], x) == y).cpu())
    clean_correct = {m: torch.cat(v) for m, v in clean_correct.items()}
    clean_acc = {m: clean_correct[m].float().mean().item() for m in names}
    for m in names:
        print(f"{m:12s} recognition rate (clean): {pct(clean_acc[m])}")

    # ---- 2. attacks ------------------------------------------------------------------
    results = {m: {a: {} for a in args.attacks} for m in names}
    transfer = {a: {s: {} for s in names} for a in args.attacks}
    examples = {}  # (model, attack) -> (x_adv, pred) for the first 8 test images at transfer_eps

    for src in names:
        model = models[src]
        for atk in args.attacks:
            fn = ATTACKS[atk]
            for eps in args.eps:
                t0 = time.time()
                adv_correct = []
                tr_correct = {t: [] for t in names if t != src}
                for bi, (x, y) in enumerate(batches):
                    x, y = x.to(device), y.to(device)
                    x_adv = fn(model, x, y, eps, steps=args.steps, mu=args.mu)
                    pred = predict(model, x_adv)
                    adv_correct.append((pred == y).cpu())
                    if eps == args.transfer_eps:
                        for t in tr_correct:
                            tr_correct[t].append((predict(models[t], x_adv) == y).cpu())
                        if bi == 0:
                            examples[(src, atk)] = (x_adv[:8].cpu(), pred[:8].cpu())
                adv_correct = torch.cat(adv_correct)
                base = clean_correct[src]
                fooled = (base & ~adv_correct).sum().item()
                asr = fooled / max(1, base.sum().item())
                rob = adv_correct.float().mean().item()
                linf = eps  # all attacks are clipped to the eps-ball
                results[src][atk][str(eps)] = dict(asr=asr, robust_acc=rob, fooled=fooled,
                                                    linf=linf, seconds=time.time() - t0)
                print(f"{src:12s} {ATTACK_NAMES[atk]:8s} eps={eps:.2f}  ASR {pct(asr):>7s}  "
                      f"robust acc {pct(rob):>7s}  ({time.time() - t0:.1f}s)")
                if eps == args.transfer_eps:
                    for t, v in tr_correct.items():
                        v = torch.cat(v)
                        tb = clean_correct[t]
                        transfer[atk][src][t] = (tb & ~v).sum().item() / max(1, tb.sum().item())

    # ---- 3. save everything ----------------------------------------------------------
    (OUT / "results.json").write_text(json.dumps(dict(
        n_test=n_total, eps=args.eps, steps=args.steps, mu=args.mu,
        transfer_eps=args.transfer_eps, clean_acc=clean_acc, attacks=results,
        transfer_asr=transfer), indent=2))

    write_markdown(names, args, clean_acc, results, transfer, n_total)
    plot_curves(names, args, results)
    plot_examples(names, args, batches, models, examples)
    print(f"\nSaved results to {OUT}/ (results.md, results.json, plots)")
    print((OUT / "results.md").read_text())


def write_markdown(names, args, clean_acc, results, transfer, n_total):
    L = ["# MNIST adversarial attack results", "",
         f"Test images: {n_total} (MNIST test split). Untargeted white-box L-inf attacks, "
         f"{args.steps} iterations for I-FGSM / PGD / MI-FGSM (mu={args.mu}).", "",
         "**ASR** = share of correctly classified test images that the attack makes the model "
         "misclassify. **Robust acc** = accuracy on the attacked test set.", "",
         "## Recognition rate before attacks", "", "| Model | Clean accuracy |", "|---|---:|"]
    L += [f"| {m} | {pct(clean_acc[m])} |" for m in names]
    eps_hdr = " | ".join(f"eps={e:g}" for e in args.eps)
    sep = "|---|" + "---:|" * len(args.eps)
    for title, key in [("Attack success rate (ASR)", "asr"), ("Robust accuracy under attack", "robust_acc")]:
        L += ["", f"## {title}", ""]
        for m in names:
            L += [f"**{m}**", "", f"| Attack | {eps_hdr} |", sep]
            for a in args.attacks:
                L.append(f"| {ATTACK_NAMES[a]} | " +
                         " | ".join(pct(results[m][a][str(e)][key]) for e in args.eps) + " |")
            L.append("")
    if len(names) > 1:
        L += [f"## Transferability (black-box ASR at eps={args.transfer_eps:g})", "",
              "Rows: model the attack was crafted on. Columns: model being fooled.", ""]
        for a in args.attacks:
            L += [f"**{ATTACK_NAMES[a]}**", "", "| crafted on \\ tested on | " + " | ".join(names) + " |",
                  "|---|" + "---:|" * len(names)]
            for s in names:
                cells = [("(white-box) " + pct(results[s][a][str(args.transfer_eps)]["asr"])) if t == s
                         else pct(transfer[a][s][t]) for t in names]
                L.append(f"| {s} | " + " | ".join(cells) + " |")
            L.append("")
    (OUT / "results.md").write_text("\n".join(L) + "\n")


def plot_curves(names, args, results):
    for key, label in [("asr", "Attack success rate"), ("robust_acc", "Robust accuracy")]:
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
        for ax, m in zip(axes[0], names):
            for a in args.attacks:
                ax.plot(args.eps, [results[m][a][str(e)][key] * 100 for e in args.eps],
                        marker="o", label=ATTACK_NAMES[a])
            ax.set_title(m); ax.set_xlabel("epsilon (L-inf)"); ax.set_ylabel(f"{label} (%)")
            ax.set_ylim(0, 102); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(OUT / f"{key}_vs_eps.png", dpi=120); plt.close(fig)


def plot_examples(names, args, batches, models, examples):
    x0, y0 = batches[0][0][:8], batches[0][1][:8]
    for m in names:
        rows = [("clean", x0, predict(models[m], x0.to(next(models[m].parameters()).device)).cpu())]
        rows += [(ATTACK_NAMES[a], *examples[(m, a)]) for a in args.attacks if (m, a) in examples]
        fig, axes = plt.subplots(len(rows), 8, figsize=(12, 1.7 * len(rows)), squeeze=False)
        for r, (label, imgs, preds) in enumerate(rows):
            for c in range(8):
                ax = axes[r][c]
                ax.imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1); ax.axis("off")
                ok = preds[c].item() == y0[c].item()
                ax.set_title(f"pred {preds[c].item()}", fontsize=9, color="green" if ok else "red")
            axes[r][0].text(-12, 14, label, fontsize=10, ha="right", va="center")
        fig.suptitle(f"{m}: true labels {y0.tolist()}  (eps={args.transfer_eps:g}, red = fooled)")
        fig.tight_layout(); fig.savefig(OUT / f"{m}_examples.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
