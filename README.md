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
| 1. Plot | `python src/plot_curves.py results/runs/baseline_seed0 --out results/stage1_baseline_curves.png` | figure |
| 2. Noise sweep (sym 0/20/40/60%, asym 20/40%) | `bash scripts/stage2.sh` | `results/runs/{sym,asym}*_seed0/` |
| 2. Plots and table | `python src/plot_noise.py` | `results/stage2_*.png`, `results/stage2_table.csv` |
| 3. Mitigation at 40% noise: CE, label smoothing, mixup, SCE × sym/asym × seeds 0,1,2 | `bash scripts/stage3.sh` | `results/runs/stage3/` |
| 3. Long run, CE sym 40%, 100 epochs | `bash scripts/stage3_long.sh` | `results/runs/long/` |
| 3. Table and figure | `python src/summarize_methods.py` | `results/stage3_table.csv`, `results/stage3_methods.png` |

Methods (hyperparameters from the original papers, not tuned): label smoothing ε=0.1; mixup α=1.0; symmetric cross-entropy α=0.1, β=1.0 (log 0 clipped at log 1e-4). Early stopping is evaluated post hoc from per-epoch logs: the epoch with the best clean-label validation accuracy and the epoch with the best noisy-label validation accuracy.

Noise: symmetric flips exactly `rate` of training labels to a uniformly random other class; asymmetric flips `rate` of labels within classes truck→automobile, bird→airplane, deer→horse, cat↔dog. Validation labels get the same kind of noise (stored as `val_noisy_*` columns); test labels are always clean.
Each run logs per epoch: accuracy on clean training examples, share of noisy examples predicted as their wrong label (`noisy_memorized`) or as their true label (`noisy_true_pred`), clean and noisy validation, test.

Split: 45 000 train / 5 000 validation (fixed, `SPLIT_SEED=0` in `src/common.py`) / 10 000 test.
Recipe: ResNet-18 (3x3 stem, no max-pool), SGD momentum 0.9 Nesterov, lr 0.1 one-cycle, weight decay 5e-4, batch 256, random crop + horizontal flip, mixed precision.

## Running on Kaggle

Notebook settings: Accelerator `GPU T4` or `GPU P100`, Internet `On`. Optionally add a CIFAR-10 python-format dataset as Input: any folder named `cifar-10-batches-py` under `/kaggle/input` is found automatically, otherwise the data is downloaded.

```
%cd /kaggle/working
!rm -rf DL_noise
!git clone https://github.com/IsaenkovDanil/DL_noise.git
%cd DL_noise
!bash scripts/stage3.sh
!bash scripts/stage3_long.sh
!zip -r /kaggle/working/stage3_runs.zip results/runs/stage3 results/runs/long
```

Run with `Save Version → Save & Run All (Commit)` so that results in `/kaggle/working` are kept in the version Output.

## Layout

```
src/common.py        data loading, split, augmentation, model, evaluation
src/train.py         training script, logs every epoch to CSV
src/noise.py         symmetric and class-dependent label noise
src/plot_curves.py   curves of one run
src/plot_noise.py    stage 2 table and figures
src/losses.py        symmetric cross-entropy and loss selection
src/summarize_methods.py  stage 3 table (mean ± std over seeds) and figure
scripts/stage2.sh    stage 2 noise sweep
scripts/stage3.sh    stage 3 methods sweep
scripts/stage3_long.sh  stage 3 long run
src/look_at_data.py  dataset overview figure
src/benchmark.py     speed check
results/             figures and run logs
```
