"""Run named cross-validation experiments and log one row each in results/results.csv.

Every experiment uses the same protocol: 5 stratified folds with seed 42, the full
preprocessing pipeline refit inside each fold, and the F1 cutoff tuned on the pooled
out-of-fold probabilities. Run from the repository root:

    python run_cv.py --list                  # show the experiments
    python run_cv.py E1-logistic E9-leaky    # run some of them
    python run_cv.py E1-logistic --seed 43   # other folds, logged as E1-logistic-seed43
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import time

import numpy as np

from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
)
from src.cross_validation import cross_validate
from src.metrics import classification_metrics, probabilities_from_logits
from src.preprocessing import load_csv, read_feature_names

ROOT = Path(__file__).resolve().parent
COLUMNS = [
    "date",
    "owner",
    "experiment",
    "git_commit",
    "model",
    "params",
    "preprocessing",
    "columns",
    "threshold",
    "f1_mean",
    "f1_std",
    "f1_pooled",
    "precision",
    "recall",
    "accuracy",
    "seconds",
    "aicrowd_f1",  # filled in by hand after a submission
]


# --- Models: each returns fit_predict(train_matrix, train_labels, held_out_matrix) --


def majority():
    """Always predict -1: the floor every model must beat (F1 = 0)."""
    return lambda matrix, labels, held_out: np.zeros(len(held_out))


def logistic(lambda_, gamma, max_iters):
    """Regularized logistic regression by gradient descent, starting from zero."""

    def fit_predict(matrix, labels, held_out):
        w0 = np.zeros(matrix.shape[1], dtype=matrix.dtype)
        w, _ = reg_logistic_regression(labels, matrix, lambda_, w0, max_iters, gamma)
        return probabilities_from_logits(held_out @ w)

    return fit_predict


def plain_logistic(gamma, max_iters):
    """Unregularized logistic regression by gradient descent, starting from zero."""

    def fit_predict(matrix, labels, held_out):
        w0 = np.zeros(matrix.shape[1], dtype=matrix.dtype)
        w, _ = logistic_regression(labels, matrix, w0, max_iters, gamma)
        return probabilities_from_logits(held_out @ w)

    return fit_predict


def mse_gd(gamma, max_iters):
    """Linear regression on 0/1 labels by full-batch gradient descent, from zero."""

    def fit_predict(matrix, labels, held_out):
        w0 = np.zeros(matrix.shape[1], dtype=matrix.dtype)
        w, _ = mean_squared_error_gd(labels, matrix, w0, max_iters, gamma)
        return held_out @ w

    return fit_predict


def mse_sgd(gamma, max_iters, seed=0):
    """Linear regression on 0/1 labels by SGD, one random row per step, from zero.
    mean_squared_error_sgd draws rows from np.random, seeded here so runs repeat."""

    def fit_predict(matrix, labels, held_out):
        np.random.seed(seed)
        w0 = np.zeros(matrix.shape[1], dtype=matrix.dtype)
        w, _ = mean_squared_error_sgd(labels, matrix, w0, max_iters, gamma)
        return held_out @ w

    return fit_predict


def weighted_logistic(lambda_, gamma, max_iters, positive_weight):
    """Logistic regression whose loss counts every positive positive_weight times
    ("balanced": as many times as there are negatives per positive).

    Same gradient descent as reg_logistic_regression, with the weights rescaled to
    mean 1 so the step size means the same. The weighting multiplies the odds by
    positive_weight; the returned probabilities divide it back out, which keeps
    their ranking but puts the tuned cutoff inside the usual grid.
    """

    def fit_predict(matrix, labels, held_out):
        weight = positive_weight
        if weight == "balanced":
            weight = np.sum(labels != 1) / np.sum(labels == 1)
        weights = np.where(labels == 1, weight, 1.0).astype(matrix.dtype)
        weights /= weights.mean()
        w = np.zeros(matrix.shape[1], dtype=matrix.dtype)
        for _ in range(max_iters):
            error = weights * (probabilities_from_logits(matrix @ w) - labels)
            w = w - gamma * (matrix.T @ error / len(labels) + 2 * lambda_ * w)
        return probabilities_from_logits(held_out @ w - np.log(weight))

    return fit_predict


def undersampled_logistic(lambda_, gamma, max_iters, negatives_per_positive, seed=0):
    """Regularized logistic regression trained on all positives and a random subset
    of negatives_per_positive negatives per positive, drawn anew in every fold.

    Keeping a fraction f of the negatives multiplies the odds by 1 / f; the returned
    probabilities undo that, as in weighted_logistic.
    """
    rng = np.random.default_rng(seed)

    def fit_predict(matrix, labels, held_out):
        positives = np.flatnonzero(labels == 1)
        negatives = np.flatnonzero(labels != 1)
        n_kept = min(len(negatives), negatives_per_positive * len(positives))
        keep = np.r_[positives, rng.choice(negatives, n_kept, replace=False)]
        w0 = np.zeros(matrix.shape[1], dtype=matrix.dtype)
        w, _ = reg_logistic_regression(
            labels[keep], matrix[keep], lambda_, w0, max_iters, gamma
        )
        kept_fraction = n_kept / len(negatives)
        return probabilities_from_logits(held_out @ w + np.log(kept_fraction))

    return fit_predict


def subsampled_logistic(lambda_, gamma, max_iters, fraction, seed=0):
    """Regularized logistic regression trained on a random fraction of the training
    rows, the same fraction of each class, drawn anew in every fold (E7 learning
    curve). Preprocessing statistics still come from the whole training fold; the
    held-out fold is the same as for the other experiments.
    """
    rng = np.random.default_rng(seed)

    def fit_predict(matrix, labels, held_out):
        keep = np.concatenate(
            [
                rng.choice(rows, max(1, round(fraction * len(rows))), replace=False)
                for rows in (np.flatnonzero(labels == 1), np.flatnonzero(labels != 1))
            ]
        )
        w0 = np.zeros(matrix.shape[1], dtype=matrix.dtype)
        w, _ = reg_logistic_regression(
            labels[keep], matrix[keep], lambda_, w0, max_iters, gamma
        )
        return probabilities_from_logits(held_out @ w)

    return fit_predict


def linear(lambda_=None):
    """Least squares (or ridge if lambda_ is given) on 0/1 labels; scores are not
    probabilities, but the tuned cutoff makes them usable as such."""

    def fit_predict(matrix, labels, held_out):
        matrix, labels = matrix.astype(np.float64), labels.astype(np.float64)
        if lambda_ is None:
            w, _ = least_squares(labels, matrix)
        else:
            w, _ = ridge_regression(labels, matrix, lambda_)
        return held_out.astype(np.float64) @ w

    return fit_predict


MODELS = {
    "majority": majority,
    "logistic": logistic,
    "plain-logistic": plain_logistic,
    "mse-gd": mse_gd,
    "mse-sgd": mse_sgd,
    "weighted-logistic": weighted_logistic,
    "undersampled-logistic": undersampled_logistic,
    "subsampled-logistic": subsampled_logistic,
    "linear": linear,
}
BASE = {"lambda_": 1e-4, "gamma": 1.0, "max_iters": 200}

# name -> (question it answers, model, model params, fit_preprocessor options)
EXPERIMENTS = {
    "E0-majority": ("Always predict -1", "majority", {}, {}),
    "E1-logistic": ("Main pipeline, regularized logistic", "logistic", BASE, {}),
    "E1-least-squares": ("Main pipeline, least squares", "linear", {}, {}),
    "E1-ridge": ("Main pipeline, ridge", "linear", {"lambda_": 1e-4}, {}),
    # gamma below 2 / 8.5, the largest eigenvalue of X.T @ X / n on the 200 columns.
    "E1-mse-gd": (
        "Main pipeline, linear regression by GD",
        "mse-gd",
        {"gamma": 0.1, "max_iters": 1000},
        {},
    ),
    # gamma * |x_i|^2 < 2 for every row (largest |x_i|^2 is about 7400); about 4 epochs.
    "E1-mse-sgd": (
        "Main pipeline, linear regression by SGD",
        "mse-sgd",
        {"gamma": 1e-4, "max_iters": 1_000_000},
        {},
    ),
    "E1-logistic-plain": (
        "Main pipeline, logistic regression without penalty",
        "plain-logistic",
        {"gamma": 1.0, "max_iters": 200},
        {},
    ),
    "E0-raw-least-squares": (
        "Every column except LEAKY, raw codes, mean-imputed, least squares",
        "linear",
        {},
        {"raw": "all"},
    ),
    "E0-raw-logistic": (
        "Every column except LEAKY, raw codes, mean-imputed, logistic",
        "logistic",
        BASE,
        {"raw": "all"},
    ),
    "E2-raw-codes": (
        "The 77 SPEC features with raw codes, mean-imputed, logistic",
        "logistic",
        BASE,
        {"raw": "spec"},
    ),
    "E3-indicators": (
        "Add missing indicators",
        "logistic",
        BASE,
        {"indicators": True},
    ),
    "E3-max-nan-0.5": (
        "Drop features over 50 % missing",
        "logistic",
        BASE,
        {"max_nan": 0.5},
    ),
    "E4-ordinal-onehot": (
        "One-hot instead of numeric ordinals",
        "logistic",
        BASE,
        {"ordinal_onehot": True},
    ),
    "E9-leaky": (
        "Add post-diagnosis questions (HAREHAB1, aspirin)",
        "logistic",
        BASE,
        {"leaky": True},
    ),
}

# E7: tune logistic regression one setting at a time, each against E1-logistic.
for iters in (500, 1000, 2000):
    EXPERIMENTS[f"E7-iters-{iters}"] = (
        f"{iters} gradient steps instead of 200",
        "logistic",
        {**BASE, "max_iters": iters},
        {},
    )
# 200 steps already reach the F1 plateau, but stop short of the optimum for a given
# lambda; 1000 steps let small lambdas show their effect.
for lambda_ in (1e-6, 1e-5, 1e-3, 1e-2, 1e-1):
    EXPERIMENTS[f"E7-lambda-{lambda_:g}"] = (
        f"lambda {lambda_:g} instead of 1e-4, 1000 steps",
        "logistic",
        {**BASE, "lambda_": lambda_, "max_iters": 1000},
        {},
    )
for gamma in (0.5, 2.0):
    EXPERIMENTS[f"E7-gamma-{gamma:g}"] = (
        f"step size {gamma:g} instead of 1",
        "logistic",
        {**BASE, "gamma": gamma},
        {},
    )
# Learning curve: the same held-out folds, a growing share of the training rows
# (1.0 is E1-logistic).
for fraction in (0.01, 0.02, 0.05, 0.1, 0.25, 0.5):
    EXPERIMENTS[f"E7-fraction-{fraction:g}"] = (
        f"train on {fraction:.0%} of the training rows",
        "subsampled-logistic",
        {**BASE, "fraction": fraction},
        {},
    )

# E6: handle the 1:10 imbalance in training, not only with the tuned cutoff.
for weight in (3, "balanced"):
    EXPERIMENTS[f"E6-weight-{weight}"] = (
        f"positives weighted {weight} in the loss",
        "weighted-logistic",
        {**BASE, "positive_weight": weight},
        {},
    )
for ratio in (1, 3):
    EXPERIMENTS[f"E6-undersample-{ratio}"] = (
        f"train on {ratio} random negative(s) per positive",
        "undersampled-logistic",
        {**BASE, "negatives_per_positive": ratio},
        {},
    )

# E8: non-linear features, one group at a time, then all three together.
E8_GROUPS = {
    "squares": "squares of the continuous features",
    "bins": "age and BMI bins",
    "interactions": "age x sex, BMI, smoking, diabetes, BP, cholesterol",
}
for group, question in E8_GROUPS.items():
    EXPERIMENTS[f"E8-{group}"] = (f"Add {question}", "logistic", BASE, {group: True})
EXPERIMENTS["E8-all"] = (
    "Add squares, bins and age interactions",
    "logistic",
    BASE,
    dict.fromkeys(E8_GROUPS, True),
)

# E5: features that may partly follow from a heart-disease diagnosis rather than
# predict it (vaccines recommended to cardiac patients, blood-pressure medication).
E5_DROPS = {
    "vaccines": ("FLUSHOT6", "PNEUVAC3"),
    "bpmeds": ("BPMEDS",),
    "borderline": ("FLUSHOT6", "PNEUVAC3", "BPMEDS"),
}
for group, features in E5_DROPS.items():
    EXPERIMENTS[f"E5-no-{group}"] = (
        f"Drop {', '.join(features)}",
        "logistic",
        BASE,
        {"drop": features},
    )


def git_commit():
    """Short hash of HEAD, marked -dirty if tracked files have uncommitted changes."""

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        dirty = git("status", "--porcelain", "--untracked-files=no")
        return git("rev-parse", "--short", "HEAD") + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_name(name, seed):
    """Name in the log and the out-of-fold file; seed 42 keeps the plain name."""
    return name if seed == 42 else f"{name}-seed{seed}"


def run_experiment(name, x, labels, feature_names, seed=42):
    """Cross-validate one named experiment and return its results.csv row and
    out-of-fold probabilities."""
    _, model, params, options = EXPERIMENTS[name]
    start = time.perf_counter()
    result = cross_validate(
        x, labels, feature_names, MODELS[model](**params), seed=seed, **options
    )
    y = (np.asarray(labels) == 1).astype(float)
    pooled = classification_metrics(y, result["probabilities"] >= result["threshold"])
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "experiment": run_name(name, seed),
        "model": model,
        "params": repr(params),
        "preprocessing": repr(options),
        "columns": result["columns_per_fold"][0],
        "threshold": result["threshold"],
        "f1_mean": round(result["f1_mean"], 4),
        "f1_std": round(result["f1_std"], 4),
        "f1_pooled": round(result["f1_pooled"], 4),
        "precision": round(pooled["precision"], 4),
        "recall": round(pooled["recall"], 4),
        "accuracy": round(pooled["accuracy"], 4),
        "seconds": round(time.perf_counter() - start),
        "aicrowd_f1": "",
    }
    return row, result["probabilities"]


def append_row(path, row):
    """Append one row to the results log, writing the header if the file is new."""
    path = Path(path)
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    """Command-line entry point; paths default to this repository."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiments", nargs="*", help="names from --list")
    parser.add_argument("--list", action="store_true", help="show the experiments")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--log", type=Path, default=ROOT / "results" / "results.csv")
    parser.add_argument("--owner", default=None, help="defaults to git user.name")
    parser.add_argument("--seed", type=int, default=42, help="seed of the folds")
    args = parser.parse_args()

    if args.list or not args.experiments:
        for name, (question, model, params, options) in EXPERIMENTS.items():
            print(f"{name:<20} {question}  [{model} {params} {options}]")
        return
    unknown = [name for name in args.experiments if name not in EXPERIMENTS]
    if unknown:
        parser.error(f"unknown experiments: {unknown}; see --list")

    print("Loading training data...", flush=True)
    feature_names = read_feature_names(args.data_dir / "x_train.csv")
    x = load_csv(args.data_dir / "x_train.csv")
    y = load_csv(args.data_dir / "y_train.csv")
    if not np.array_equal(x[:, 0], y[:, 0]):
        raise ValueError("x_train and y_train IDs differ.")
    x, labels = x[:, 1:], y[:, 1]
    owner = (
        args.owner
        or subprocess.run(
            ["git", "config", "user.name"], capture_output=True, text=True
        ).stdout.strip()
    )
    commit = git_commit()

    for name in args.experiments:
        print(f"Running {run_name(name, args.seed)}...", flush=True)
        row, probabilities = run_experiment(name, x, labels, feature_names, args.seed)
        row.update(owner=owner, git_commit=commit)
        append_row(args.log, row)
        # Out-of-fold probabilities for error analysis and threshold plots (E11).
        # compare_cv.py rebuilds the folds from the labels and the seed.
        oof = args.log.parent / "oof" / f"{row['experiment']}.npz"
        oof.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            oof, probabilities=probabilities, labels=labels, seed=args.seed
        )
        print(
            f"  F1 {row['f1_mean']:.4f} ± {row['f1_std']:.4f} at cutoff "
            f"{row['threshold']:.2f}, {row['columns']} columns, {row['seconds']} s",
            flush=True,
        )
    print("Logged to", args.log)


if __name__ == "__main__":
    main()
