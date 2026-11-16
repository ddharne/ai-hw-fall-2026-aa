#!/usr/bin/env bash
# 1) train the models on the MNIST train split, 2) attack them on the test split.
set -e
for m in mlp cnn vit; do
  python train.py --model $m --epochs 10
done
# Optional defense (slower: ~8x the training time). Uncomment to include it in the results.
# python train.py --model cnn --epochs 10 --adv-train
python test.py
