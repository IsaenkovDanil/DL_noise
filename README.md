# DL_noise: label noise and dataset audit on CIFAR

Course project (CVDL, Project 5): how noisy labels hurt a ResNet-18 on CIFAR-10, what helps, and how to find mislabeled images.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CIFAR-10 is downloaded automatically to `data/` on the first run.

## Experiments

| Stage | Command | Output |
|---|---|---|
| 1. Clean baseline | `python src/train.py --epochs 30 --seed 0 --out results/runs/baseline_seed0` | `log.csv`, `summary.json` |

Split: 45 000 train / 5 000 validation (fixed, `SPLIT_SEED=0` in `src/common.py`) / 10 000 test.
Recipe: ResNet-18 (3x3 stem, no max-pool), SGD momentum 0.9 Nesterov, lr 0.1 one-cycle, weight decay 5e-4, batch 256, random crop + horizontal flip, mixed precision.

## Running on Kaggle

Notebook settings: Accelerator `GPU T4` or `GPU P100`, Internet `On`.

```
!git clone https://github.com/IsaenkovDanil/DL_noise.git
%cd DL_noise
!python src/train.py --epochs 30 --seed 0 --out results/runs/baseline_seed0
!zip -r /kaggle/working/runs.zip results/runs
```

## Layout

```
src/common.py        data loading, split, augmentation, model, evaluation
src/train.py         training script, logs every epoch to CSV
src/look_at_data.py  dataset overview figure
src/benchmark.py     speed check
results/             figures and run logs
```
