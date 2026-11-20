#!/usr/bin/env bash
set -e
EPOCHS=${EPOCHS:-30}
SEEDS=${SEEDS:-"0 1 2"}
R=results/runs/stage6

run() {
  local out="${@: -1}"
  if [ -f "$out/summary.json" ]; then
    echo "skip $out (already done)"
  else
    python src/train.py "$@"
  fi
}

W_SRC=results/runs/stage5/filter/source_c10n-worse_seed0
W_SCORES=results/stage6_cifar10_c10n-worse_scores.npy
for SEED in $SEEDS; do
  run --epochs $EPOCHS --seed $SEED --noise-type c10n --c10n-label worse_label \
    --filter-from $W_SRC --filter-scores $W_SCORES --drop-frac 0.4 \
    --out $R/clipfilter_c10n-worse_seed${SEED}
done
run --epochs $EPOCHS --seed 0 --noise-type c10n --c10n-label worse_label --method mixup \
  --filter-from $W_SRC --filter-scores $W_SCORES --drop-frac 0.4 \
  --out $R/clipfilter_mixup_c10n-worse_seed0
run --epochs $EPOCHS --seed 0 --noise-type sym --noise-rate 0.4 \
  --filter-from results/runs/stage4/source_sym0.4_seed0 \
  --filter-scores results/stage6_cifar10_sym0.4_scores.npy --drop-frac 0.4 \
  --out $R/clipfilter_sym0.4_seed0
run --epochs $EPOCHS --seed 0 --noise-type asym --noise-rate 0.4 \
  --filter-from results/runs/stage4/source_asym0.4_seed0 \
  --filter-scores results/stage6_cifar10_asym0.4_scores.npy --drop-frac 0.2 \
  --out $R/clipfilter_asym0.4_seed0
