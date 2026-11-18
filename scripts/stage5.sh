#!/usr/bin/env bash
set -e
EPOCHS=${EPOCHS:-30}
SEEDS=${SEEDS:-"0 1 2"}
R=results/runs/stage5
C10N="--noise-type c10n"

for LABEL in aggre random1; do
  python src/train.py --epochs $EPOCHS --seed 0 $C10N --c10n-label ${LABEL}_label \
    --out $R/rates/ce_c10n-${LABEL}_seed0
done

for METHOD in ce mixup sce; do
  for SEED in $SEEDS; do
    python src/train.py --epochs $EPOCHS --seed $SEED $C10N --c10n-label worse_label \
      --method $METHOD --out $R/methods/${METHOD}_c10n-worse_seed${SEED}
  done
done

SRC=$R/filter/source_c10n-worse_seed0
python src/train.py --epochs $EPOCHS --seed 0 $C10N --c10n-label worse_label --save-train-loss --out $SRC
python src/train.py --epochs $EPOCHS --seed 0 $C10N --c10n-label worse_label \
  --filter-from $SRC --filter-epoch 10 --drop-frac 0.4 --out $R/filter/filtered_e10_c10n-worse_seed0
python src/train.py --epochs $EPOCHS --seed 0 $C10N --c10n-label worse_label \
  --filter-from $SRC --filter-epoch 10 --drop-frac 0.4 --filter-per-class \
  --out $R/filter/filtered_e10pc_c10n-worse_seed0
python src/train.py --epochs $EPOCHS --seed 0 $C10N --c10n-label worse_label \
  --filter-from $SRC --filter-oracle --out $R/filter/filtered_oracle_c10n-worse_seed0

for SEED in $SEEDS; do
  python src/train.py --epochs $EPOCHS --seed $SEED --noise-type sym --noise-rate 0.4 \
    --filter-from results/runs/stage4/source_sym0.4_seed0 --filter-epoch 10 --drop-frac 0.4 \
    --filter-per-class --out results/runs/stage4/filtered_e10pc_sym0.4_seed${SEED}
done
python src/train.py --epochs $EPOCHS --seed 0 --noise-type asym --noise-rate 0.4 \
  --filter-from results/runs/stage4/source_asym0.4_seed0 --filter-epoch 10 --drop-frac 0.2 \
  --filter-per-class --out results/runs/stage4/filtered_e10pc_asym0.4_seed0
