"""Run named cross-validation experiments and log one row each in results/results.csv.

Every experiment uses the same protocol: 5 stratified folds with seed 42, the full
preprocessing pipeline refit inside each fold, and the F1 cutoff tuned on the pooled
out-of-fold probabilities. Run from the repository root:

    python run_cv.py --list                  # show the experiments
    python run_cv.py E1-logistic E9-leaky    # run some of them
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import subprocess
import time

import numpy as np

from implementations import least_squares, reg_logistic_regression, ridge_regression
from src.cross_validation import cross_validate
from src.metrics import classification_metrics
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
        return 1 / (1 + np.exp(-(held_out @ w)))

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


MODELS = {"majority": majority, "logistic": logistic, "linear": linear}
BASE = {"lambda_": 1e-4, "gamma": 1.0, "max_iters": 200}

# name -> (question it answers, model, model params, fit_preprocessor options)
EXPERIMENTS = {
    "E0-majority": ("Always predict -1", "majority", {}, {}),
    "E1-logistic": ("Main pipeline, regularized logistic", "logistic", BASE, {}),
    "E1-least-squares": ("Main pipeline, least squares", "linear", {}, {}),
    "E1-ridge": ("Main pipeline, ridge", "linear", {"lambda_": 1e-4}, {}),
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


def run_experiment(name, x, labels, feature_names):
    """Cross-validate one named experiment and return its results.csv row and
    out-of-fold probabilities."""
    _, model, params, options = EXPERIMENTS[name]
    start = time.perf_counter()
    result = cross_validate(
        x, labels, feature_names, MODELS[model](**params), **options
    )
    y = (np.asarray(labels) == 1).astype(float)
    pooled = classification_metrics(y, result["probabilities"] >= result["threshold"])
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "experiment": name,
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
        print(f"Running {name}...", flush=True)
        row, probabilities = run_experiment(name, x, labels, feature_names)
        row.update(owner=owner, git_commit=commit)
        append_row(args.log, row)
        # Out-of-fold probabilities for error analysis and threshold plots (E11).
        oof = args.log.parent / "oof" / f"{name}.npz"
        oof.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(oof, probabilities=probabilities, labels=labels)
        print(
            f"  F1 {row['f1_mean']:.4f} ± {row['f1_std']:.4f} at cutoff "
            f"{row['threshold']:.2f}, {row['columns']} columns, {row['seconds']} s",
            flush=True,
        )
    print("Logged to", args.log)


if __name__ == "__main__":
    main()
