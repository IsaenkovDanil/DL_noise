#!/usr/bin/env bash
set -e
EPOCHS=${EPOCHS:-100}
python src/train.py --epochs $EPOCHS --seed 0 --noise-type sym --noise-rate 0.4 \
  --out results/runs/long/sym0.4_e${EPOCHS}_seed0
