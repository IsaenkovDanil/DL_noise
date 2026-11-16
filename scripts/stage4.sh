#!/usr/bin/env bash
set -e
EPOCHS=${EPOCHS:-30}
SEEDS=${SEEDS:-"0 1 2"}
FILTER_EPOCH=${FILTER_EPOCH:-10}
R=results/runs/stage4
for NOISE in sym asym; do
  if [ $NOISE = sym ]; then DROP=0.4; else DROP=0.2; fi
  SRC=$R/source_${NOISE}0.4_seed0
  python src/train.py --epochs $EPOCHS --seed 0 --noise-type $NOISE --noise-rate 0.4 \
    --save-train-loss --out $SRC
  for SEED in $SEEDS; do
    python src/train.py --epochs $EPOCHS --seed $SEED --noise-type $NOISE --noise-rate 0.4 \
      --filter-from $SRC --filter-epoch $FILTER_EPOCH --drop-frac $DROP \
      --out $R/filtered_e${FILTER_EPOCH}_${NOISE}0.4_seed${SEED}
  done
  python src/train.py --epochs $EPOCHS --seed 0 --noise-type $NOISE --noise-rate 0.4 \
    --filter-from $SRC --filter-epoch $EPOCHS --drop-frac $DROP \
    --out $R/filtered_e${EPOCHS}_${NOISE}0.4_seed0
  python src/train.py --epochs $EPOCHS --seed 0 --noise-type $NOISE --noise-rate 0.4 \
    --filter-from $SRC --filter-oracle \
    --out $R/filtered_oracle_${NOISE}0.4_seed0
done
