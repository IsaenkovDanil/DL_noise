#!/usr/bin/env bash
set -e
EPOCHS=${EPOCHS:-30}
SEEDS=${SEEDS:-"0 1 2"}
for NOISE in sym asym; do
  for METHOD in ce ls mixup sce; do
    for SEED in $SEEDS; do
      python src/train.py --epochs $EPOCHS --seed $SEED --noise-type $NOISE --noise-rate 0.4 \
        --method $METHOD --out results/runs/stage3/${METHOD}_${NOISE}0.4_seed${SEED}
    done
  done
done
