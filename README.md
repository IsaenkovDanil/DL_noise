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

| 4. Loss-based filtering: source run saving per-example loss, retrain after dropping top-loss examples (epoch 10: seeds 0,1,2; last epoch; oracle) | `bash scripts/stage4.sh` | `results/runs/stage4/` |
| 4. Analysis | `python src/analyze_filtering.py` | `results/stage4_filtering.png`, `results/stage4_table.csv`, `results/stage4_pr_by_epoch_*.csv` |

| 5. CIFAR-10N human labels (aggre, random1, worse); CE/mixup/SCE on worse × seeds 0,1,2; loss filtering on worse; per-class filtering for stage 4 | `bash scripts/stage5.sh` | `results/runs/stage5/`, `results/runs/stage4/filtered_e10pc_*` |
| 5. Human vs synthetic noise, confusion matrices | `python src/analyze_c10n.py` | `results/stage5_vs_synthetic.*`, `results/stage5_confusion.png` |
| 5. Methods on worse | `python src/summarize_methods.py --runs results/runs/stage5/methods --prefix stage5` | `results/stage5_table.csv`, `results/stage5_methods.png`, `results/stage5_curves.png` |
| 5. Filtering on worse | `python src/analyze_filtering.py --runs results/runs/stage5/filter --baseline results/runs/stage5/methods --prefix stage5` | `results/stage5_filtering.png`, `results/stage5_table.csv` |

CIFAR-10N labels are downloaded from the authors' repository (`UCSC-REAL/cifar-10-100n`, `CIFAR-10_human.pt`) into `data/` on first use; their `clean_label` is checked against torchvision labels. Validation uses the same human label set as training for the noisy-validation columns.

Filtering drops the known noise fraction (40% for sym, 20% for asym) with the largest cross-entropy w.r.t. the given label at the chosen epoch of a CE source run; retraining reuses the source run's noisy labels, only the model seed changes. Precision/recall of dropped examples and ROC AUC of the loss as a noise detector are computed against the true noise mask.

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
!bash scripts/stage5.sh
!zip -r /kaggle/working/stage5_runs.zip results/runs/stage5 results/runs/stage4/filtered_e10pc_*
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
src/filtering.py     loss-based selection, precision/recall, ROC AUC
src/analyze_filtering.py  stage 4 analysis and figure
scripts/stage4.sh    stage 4 filtering and retraining
src/analyze_c10n.py  stage 5 human vs synthetic noise
scripts/stage5.sh    stage 5 CIFAR-10N runs
src/look_at_data.py  dataset overview figure
src/benchmark.py     speed check
results/             figures and run logs
```
