# ai-hw-fall-2026-aa — Adversarial attacks on MNIST

Train small image classifiers on MNIST, then try to fool them with
**FGSM**, **I-FGSM**, **PGD** and **Momentum I-FGSM (MI-FGSM)**.

**Rule followed:** models are trained only on the official **train** split (55,000 train +
5,000 validation). All metrics are measured on the official **test** split (10,000 images).

## Metrics
- **Recognition rate before attacks** — clean test accuracy.
- **Attack success rate (ASR)** — untargeted: among test images the model classifies
  correctly before the attack, the share it misclassifies after the attack.
- **Robust accuracy** — accuracy on the attacked test set.
- **Transfer ASR** (bonus) — adversarial images made on one model, tested on another (black-box).

## Models (`models.py`)
| Model | Description |
|---|---|
| MLP | 784 → 512 (ReLU, dropout) → 10 |
| CNN | 2 conv blocks (32, 64 channels, BatchNorm, max-pool) → FC 128 → 10 |
| ViT | 7×7 patches → 16 tokens + [CLS], 4 Transformer encoder layers, dim 64 |
| CNN-advtrain (optional) | Same CNN trained on PGD examples (eps 0.3) as a defense |

Each model starts with a normalization layer, so attacks work on raw [0, 1] pixels and
`eps` means the same for every model (eps = 0.1 → every pixel moves by at most 0.1).

## Attacks (`attacks.py`) — white-box, untargeted, L∞
| Attack | Update |
|---|---|
| FGSM | one step: x' = x + eps · sign(∇x loss) |
| I-FGSM | 10 steps of eps/10, projected back into the eps-ball |
| PGD | random start in the eps-ball, 10 steps of 2.5·eps/10, projected |
| MI-FGSM | I-FGSM with momentum (μ = 1.0) on L1-normalized gradients |

Epsilons tested: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30.

## How to run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh                 # train MLP, CNN, ViT, then attack all of them
# individually
python train.py --model cnn --epochs 10
python train.py --model cnn --epochs 10 --adv-train     # optional defense
python test.py --n 2000                                 # quicker test on 2000 images
```

## Results
Full tables: [`results/results.md`](results/results.md) · raw numbers: `results/results.json`
Plots: `results/asr_vs_eps.png`, `results/robust_acc_vs_eps.png`, `results/<model>_examples.png`

### Recognition rate before attacks

| Model | Clean accuracy |
|---|---:|
| cnn | 99.47% |
| mlp | 98.39% |
| vit | 97.79% |

### Attack success rate (ASR)

**cnn**

| Attack | eps=0.05 | eps=0.1 | eps=0.15 | eps=0.2 | eps=0.25 | eps=0.3 |
|---|---:|---:|---:|---:|---:|---:|
| FGSM | 4.19% | 41.42% | 65.05% | 77.60% | 83.46% | 85.33% |
| I-FGSM | 7.64% | 63.88% | 89.71% | 96.60% | 98.66% | 99.60% |
| PGD | 12.62% | 86.57% | 96.23% | 98.13% | 99.62% | 99.99% |
| MI-FGSM | 7.18% | 58.63% | 84.97% | 94.86% | 97.77% | 99.05% |

**mlp**

| Attack | eps=0.05 | eps=0.1 | eps=0.15 | eps=0.2 | eps=0.25 | eps=0.3 |
|---|---:|---:|---:|---:|---:|---:|
| FGSM | 14.19% | 62.18% | 90.22% | 96.45% | 98.65% | 99.48% |
| I-FGSM | 15.24% | 70.85% | 95.57% | 99.47% | 99.98% | 100.00% |
| PGD | 15.75% | 74.56% | 97.31% | 99.87% | 100.00% | 100.00% |
| MI-FGSM | 14.85% | 67.83% | 94.14% | 99.24% | 99.95% | 99.99% |

**vit**

| Attack | eps=0.05 | eps=0.1 | eps=0.15 | eps=0.2 | eps=0.25 | eps=0.3 |
|---|---:|---:|---:|---:|---:|---:|
| FGSM | 46.12% | 83.51% | 92.82% | 95.33% | 96.35% | 96.64% |
| I-FGSM | 66.11% | 98.31% | 99.97% | 100.00% | 100.00% | 100.00% |
| PGD | 77.19% | 99.94% | 100.00% | 100.00% | 100.00% | 100.00% |
| MI-FGSM | 58.70% | 95.27% | 99.65% | 99.98% | 100.00% | 100.00% |

## Repo layout
```
data.py      # MNIST loaders (train/val from train split, test split)
models.py    # MLP, CNN, ViT
attacks.py   # FGSM, I-FGSM, PGD, MI-FGSM
train.py     # training (train split only), optional PGD adversarial training
test.py      # clean accuracy + attacks on the test split, writes results/
run_all.sh   # full experiment
results/     # metrics and plots
```
