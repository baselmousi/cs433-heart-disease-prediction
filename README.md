# Heart Disease Prediction

Predicting coronary heart disease (MICHD) from the BRFSS 2015 health survey - EPFL
CS-433 Machine Learning, Project 1 (Fall 2026).

Final model: regularized logistic regression on 77 cleaned survey features
(200 columns), with the probability cutoff tuned for F1. 5-fold cross-validated
F1 0.4274 ± 0.0028; AIcrowd F1 0.436.

## Setup

Python 3.9 or later with NumPy (matplotlib only for plots in the notebooks); no
other library is used. Put the course files `x_train.csv`, `y_train.csv`, `x_test.csv`
and `sample_submission.csv` in `data/` (or pass `--data-dir` to any script).

## Reproduce the submission

```bash
python run.py
```

This fits the final model on all 328,135 training rows and writes
`submissions/submission.csv` (about one minute on a laptop). It first runs the same
5-fold cross-validation as the experiments to pick the F1 cutoff (0.20);
`python run.py --threshold 0.20` skips that step and gives the same file. Before
finishing, it reads the file back and checks the header, the -1/+1 values and that
the IDs match `x_test.csv` and `sample_submission.csv`.

## Reproduce the experiments

Every number in the report comes from `run_cv.py`, which runs named experiments and
appends one row each to `results/results.csv`:

```bash
python run_cv.py --list                      # every experiment and its settings
python run_cv.py E1-logistic E1-ridge        # run some of them
python run_cv.py E1-logistic --seed 43       # same experiment on other folds
```

All experiments share one protocol: 5 stratified folds (seed 42), the whole
preprocessing pipeline refit inside each fold, and the F1 cutoff chosen on a grid
from 0.05 to 0.60 using the pooled out-of-fold probabilities. Each run also saves
those probabilities in `results/oof/`, so runs can be compared afterwards without
refitting:

```bash
python compare_cv.py E1-logistic E3-indicators E9-leaky
```

`compare_cv.py` compares every run with the first one fold by fold (a paired
comparison with the Nadeau-Bengio corrected t-test) and gives the verdict that
Table I of the report uses. The experiment names follow the report: E0 baselines,
E1 the six required methods, E2-E5 preprocessing and feature choices, E6 class
imbalance, E7 hyperparameters, E8 non-linear features, E9 post-diagnosis questions.
Running all of them takes about 40 minutes.

## Code layout

| Path | Contents |
|---|---|
| `implementations.py` | The six required methods (least squares, ridge, MSE GD / SGD, logistic and regularized logistic regression by GD) |
| `run.py` | Fits the final model and writes the AIcrowd submission |
| `run_cv.py` | The experiment definitions and the cross-validation runner |
| `compare_cv.py` | Paired fold-by-fold comparison of saved runs |
| `src/feature_spec.py` | Per-feature decisions for all 321 columns: type, valid codes, skip-pattern fills, or why it is dropped |
| `src/preprocessing.py` | CSV loading and the preprocessing pipeline (decoding, imputation, encoding, scaling), fitted on training rows only |
| `src/cross_validation.py` | Stratified folds, cross-validation and the paired t-test |
| `src/metrics.py` | Precision, recall, F1, cutoff search, stable sigmoid |
| `helpers.py` | Course-provided CSV helpers |
| `train_baseline.py` | The first five-feature logistic baseline (`results/baseline/`) |
| `notebooks/` | Data exploration and the feature-by-feature decisions behind `feature_spec.py`; start with its README |
| `tests/` | Unit tests |

## Preprocessing in short

Each of the 321 columns is either used (77 features), dropped (survey metadata,
optional modules that are at least 90 % missing, raw items that CDC already
summarises in a calculated column) or excluded as leaky (questions only asked after
a heart-disease diagnosis). Used features are decoded from survey codes: "don't
know" / "refused" become missing and skip patterns are filled with their implied
answer. Binary, ordinal and continuous features are kept as numbers, with
continuous ones clipped to their 0.5-99.5 % training quantiles; missing values get
the training median, and every column is standardized. Nominal features are one-hot
encoded with "missing" as a level of its own and the most frequent level as the
reference. All statistics come from the training rows of each fold.

[`notebooks/README.md`](notebooks/README.md) explains every step: what the data
looks like, where each of the 321 columns went and why, the evidence for each
decision in the notebooks, and the experiments that tested the uncertain ones.

## Tests

```bash
python -m unittest discover -s tests
```

The tests use small synthetic tables, so they need no data and run in a few seconds.
