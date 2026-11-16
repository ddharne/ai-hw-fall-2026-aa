# MNIST adversarial attack results

Test images: 10000 (MNIST test split). Untargeted white-box L-inf attacks, 10 iterations for I-FGSM / PGD / MI-FGSM (mu=1.0).

**ASR** = share of correctly classified test images that the attack makes the model misclassify. **Robust acc** = accuracy on the attacked test set.

## Recognition rate before attacks

| Model | Clean accuracy |
|---|---:|
| cnn | 99.47% |
| mlp | 98.39% |
| vit | 97.79% |

## Attack success rate (ASR)

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


## Robust accuracy under attack

**cnn**

| Attack | eps=0.05 | eps=0.1 | eps=0.15 | eps=0.2 | eps=0.25 | eps=0.3 |
|---|---:|---:|---:|---:|---:|---:|
| FGSM | 95.30% | 58.27% | 34.77% | 22.30% | 16.48% | 14.62% |
| I-FGSM | 91.87% | 35.93% | 10.24% | 3.38% | 1.33% | 0.40% |
| PGD | 86.92% | 13.36% | 3.75% | 1.86% | 0.38% | 0.01% |
| MI-FGSM | 92.33% | 41.15% | 14.95% | 5.11% | 2.22% | 0.94% |

**mlp**

| Attack | eps=0.05 | eps=0.1 | eps=0.15 | eps=0.2 | eps=0.25 | eps=0.3 |
|---|---:|---:|---:|---:|---:|---:|
| FGSM | 84.43% | 37.21% | 9.62% | 3.49% | 1.33% | 0.51% |
| I-FGSM | 83.40% | 28.68% | 4.36% | 0.52% | 0.02% | 0.00% |
| PGD | 82.89% | 25.03% | 2.65% | 0.13% | 0.00% | 0.00% |
| MI-FGSM | 83.78% | 31.65% | 5.77% | 0.75% | 0.05% | 0.01% |

**vit**

| Attack | eps=0.05 | eps=0.1 | eps=0.15 | eps=0.2 | eps=0.25 | eps=0.3 |
|---|---:|---:|---:|---:|---:|---:|
| FGSM | 52.69% | 16.13% | 7.02% | 4.57% | 3.57% | 3.29% |
| I-FGSM | 33.14% | 1.65% | 0.03% | 0.00% | 0.00% | 0.00% |
| PGD | 22.31% | 0.06% | 0.00% | 0.00% | 0.00% | 0.00% |
| MI-FGSM | 40.39% | 4.63% | 0.34% | 0.02% | 0.00% | 0.00% |

## Transferability (black-box ASR at eps=0.2)

Rows: model the attack was crafted on. Columns: model being fooled.

**FGSM**

| crafted on \ tested on | cnn | mlp | vit |
|---|---:|---:|---:|
| cnn | (white-box) 77.60% | 8.43% | 56.57% |
| mlp | 81.46% | (white-box) 96.45% | 69.92% |
| vit | 72.95% | 13.95% | (white-box) 95.33% |

**I-FGSM**

| crafted on \ tested on | cnn | mlp | vit |
|---|---:|---:|---:|
| cnn | (white-box) 96.60% | 7.42% | 39.34% |
| mlp | 71.55% | (white-box) 99.47% | 67.84% |
| vit | 39.98% | 4.83% | (white-box) 100.00% |

**PGD**

| crafted on \ tested on | cnn | mlp | vit |
|---|---:|---:|---:|
| cnn | (white-box) 98.13% | 4.65% | 69.69% |
| mlp | 79.36% | (white-box) 99.87% | 73.33% |
| vit | 74.85% | 5.29% | (white-box) 100.00% |

**MI-FGSM**

| crafted on \ tested on | cnn | mlp | vit |
|---|---:|---:|---:|
| cnn | (white-box) 94.86% | 11.42% | 54.40% |
| mlp | 78.81% | (white-box) 99.24% | 70.15% |
| vit | 66.62% | 13.16% | (white-box) 99.98% |

