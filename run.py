"""Reproduce our best AIcrowd submission for CS-433 Project 1.

Fits one experiment from run_cv.py (E1-logistic by default) on every training row
and writes +1/-1 predictions for x_test.csv. The F1 cutoff is chosen by the same
5-fold CV as in run_cv.py unless --threshold is given. Run from the repository root:

    python run.py                    # writes submissions/submission.csv
    python run.py --threshold 0.20   # skip the CV and use a fixed cutoff
"""

import argparse
import csv
from pathlib import Path
import time

import numpy as np

from helpers import create_csv_submission
from run_cv import EXPERIMENTS, MODELS
from src.cross_validation import cross_validate
from src.preprocessing import (
    fit_preprocessor,
    load_csv,
    read_feature_names,
    transform_features,
)

ROOT = Path(__file__).resolve().parent


def load_data(data_dir):
    """Return training features, -1/+1 labels, test features, test IDs and the
    feature names, after checking that the three files line up."""
    data_dir = Path(data_dir)
    feature_names = read_feature_names(data_dir / "x_train.csv")
    if read_feature_names(data_dir / "x_test.csv") != feature_names:
        raise ValueError("x_train and x_test have different columns.")
    x_train = load_csv(data_dir / "x_train.csv")
    y_train = load_csv(data_dir / "y_train.csv")
    x_test = load_csv(data_dir / "x_test.csv")
    if not np.array_equal(x_train[:, 0], y_train[:, 0]):
        raise ValueError("x_train and y_train IDs differ.")
    test_ids = x_test[:, 0].astype(int)
    return x_train[:, 1:], y_train[:, 1], x_test[:, 1:], test_ids, feature_names


def predict(name, x_train, labels, x_test, feature_names, threshold=None):
    """Fit a run_cv.py experiment on all training rows and label the test rows.

    Without a threshold, the cutoff is tuned by the experiment's 5-fold CV first.
    Returns the -1/+1 predictions, the test probabilities and the cutoff used.
    """
    _, model, params, options = EXPERIMENTS[name]
    fit_predict = MODELS[model](**params)
    if threshold is None:
        cv = cross_validate(x_train, labels, feature_names, fit_predict, **options)
        threshold = cv["threshold"]
        print(f"  CV F1 {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}", flush=True)
    y = (np.asarray(labels) == 1).astype(float)
    state = fit_preprocessor(x_train, feature_names, **options)
    matrix = transform_features(x_train, feature_names, state)
    probabilities = fit_predict(
        matrix,
        y.astype(matrix.dtype),
        transform_features(x_test, feature_names, state),
    )
    # A NaN probability would silently become -1 below, so stop instead.
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Some test probabilities are not finite.")
    return np.where(probabilities >= threshold, 1, -1), probabilities, threshold


def check_submission(path, test_ids, sample_path=None):
    """Read the written file back and check that it is what AIcrowd expects: the
    header Id,Prediction and one -1/+1 row per test ID, in x_test order, and the
    same IDs as the course's sample_submission.csv when that file is given."""
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if rows[0] != ["Id", "Prediction"]:
        raise ValueError(f"Unexpected header {rows[0]} in {path}.")
    ids = [int(r[0]) for r in rows[1:]]
    if ids != list(test_ids):
        raise ValueError(f"The IDs in {path} are not the x_test IDs in order.")
    if {r[1] for r in rows[1:]} - {"-1", "1"}:
        raise ValueError(f"{path} has predictions other than -1 and 1.")
    if sample_path is not None and Path(sample_path).exists():
        with open(sample_path, newline="") as handle:
            sample_ids = [int(r[0]) for r in list(csv.reader(handle))[1:]]
        if ids != sample_ids:
            raise ValueError(f"The IDs in {path} differ from {sample_path}.")


def main(argv=None):
    """Command-line entry point; paths default to this repository."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment", default="E1-logistic", choices=EXPERIMENTS)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "submissions" / "submission.csv"
    )
    args = parser.parse_args(argv)

    start = time.perf_counter()
    print("Loading data...", flush=True)
    x_train, labels, x_test, test_ids, feature_names = load_data(args.data_dir)
    print(f"Fitting {args.experiment} on {len(x_train)} rows...", flush=True)
    predictions, _, threshold = predict(
        args.experiment, x_train, labels, x_test, feature_names, args.threshold
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(test_ids, predictions, args.output)
    check_submission(args.output, test_ids, args.data_dir / "sample_submission.csv")
    # The training positive rate is about 9 %; the cutoff trades precision for
    # recall, so a few points more positives than that is expected.
    print(
        f"  cutoff {threshold:.2f}: {np.mean(predictions == 1):.1%} of "
        f"{len(predictions)} test rows predicted +1 "
        f"(training rate {np.mean(labels == 1):.1%})",
        flush=True,
    )
    print(f"Wrote {args.output} in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
