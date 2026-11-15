#!/usr/bin/env bash
set -e
EPOCHS=${EPOCHS:-30}
SEED=${SEED:-0}
for RATE in 0.0 0.2 0.4 0.6; do
  python src/train.py --epochs $EPOCHS --seed $SEED --noise-type sym --noise-rate $RATE \
    --out results/runs/sym${RATE}_seed${SEED}
done
for RATE in 0.2 0.4; do
  python src/train.py --epochs $EPOCHS --seed $SEED --noise-type asym --noise-rate $RATE \
    --out results/runs/asym${RATE}_seed${SEED}
done
