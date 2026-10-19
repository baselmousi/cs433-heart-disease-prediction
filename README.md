# Heart Disease Prediction
Predicting coronary heart disease from BRFSS health survey data - EPFL CS-433 Machine Learning, Project 1 (Fall 2026).

## Five-feature logistic baseline

Place the course files `x_train.csv` and `y_train.csv` in `data/`. Use a Python
environment with NumPy installed, then run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 python train_baseline.py
```

The thread settings avoid excessive BLAS parallelism on this small feature matrix;
the script also works as `python train_baseline.py`. No notebook execution is needed.
Options include `--data-dir`, `--output-dir`, `--max-iters`, and `--tolerance`.

The script uses `GENHLTH`, `PHYSHLTH`, `_SMOKER3`, `_BMI5`, and `_AGE80`, with
the same stratified 80/20 split (seed 42) as the exploration notebook. Preprocessing
is fitted on training rows only. Categories use reference-category indicators;
numeric features use training-median replacement and training-based scaling.
Missing indicators are included where they vary in training. BMI is already in
decimal units in these CSVs. The resulting matrix includes an intercept.

Training calls `implementations.logistic_regression`: unregularized full-batch
gradient descent, starting at zero. The learning rate is derived from a bound on
training-loss curvature. Every 250 updates, the script checks the largest absolute
training-gradient component, stopping at `1e-5` or at 10,000 updates. It records
whether the convergence criterion was reached. Neither stopping nor the initial
probability cutoff of 0.5 is selected using validation performance.

Outputs in `results/baseline/`:

- `metrics.json`: settings, preprocessing statistics, coefficients, optimization
  history, training/validation scores, and majority-class comparison.
- `logistic_model.npz`: weights, preprocessing statistics, feature/column order,
  split settings, and probability cutoff. Binary model files are ignored by Git.

The model is fitted only on the training portion. Validation predictions use the
same preprocessing statistics. Confusion-matrix rows are actual classes `[0, 1]`
and columns are predicted classes `[0, 1]`: `[[TN, FP], [FN, TP]]`. Positive class 1
corresponds to course label +1. Undefined precision/recall/F1 are reported as zero.
Earlier exploration viewed all labeled data; this split does not undo that influence.

The initial run converged after 5,500 gradient updates, using 262,508 training
examples and 65,627 validation examples. At cutoff 0.5:

| Validation model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Always negative | 91.17% | 0.00% | 0.00% | 0.0000 |
| Logistic regression | 91.09% | 44.72% | 3.80% | 0.0700 |

The logistic model detects 220 of 5,795 positive validation examples, with 272
false positives. This is a weak detection baseline at the default cutoff, despite
its high accuracy. Training and validation log losses are 0.242721 and 0.242675.
No threshold search or class weighting has been performed. See `metrics.json`
for the full settings, convergence history, and confusion matrix.

## Evaluation with the JD preprocessing matrix

To evaluate logistic regression using the preprocessing pipeline in
`src/preprocessing_jd.py`, run from the repository root:

```bash
python train_preprocessed_jd.py
```

The script evaluates every completed matrix returned by the preprocessing
function, prints accuracy, precision, recall, and F1 on the stratified
validation split, and writes
`results/preprocessed_jd/preprocessed_jd_metrics.json`. It also reports the
mean and standard deviation across the completed matrices. Because the current
preprocessing function runs before the holdout split, these validation metrics
may be optimistic: its imputation and category encoding can see validation
rows.

To reuse the saved model, load it with `np.load(..., allow_pickle=False)`, pass the
five raw columns in its recorded order to `src.preprocessing.transform_baseline_features`
with the saved `medians`, `means`, `scales`, and `keep_missing`, then apply the
weights and sigmoid. The training script verifies that reloading reproduces the
validation scores. `run.py` remains reserved for the eventual competition submission.

Run focused checks with:

```bash
python -m unittest discover -s tests -v
```
