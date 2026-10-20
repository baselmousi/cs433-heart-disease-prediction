"""Evaluate logistic regression with the preprocessing pipeline in preprocessing_jd.py."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from implementations import logistic_regression
from src.metrics import (
    binary_log_loss,
    classification_metrics,
    probabilities_from_logits,
)
from src.preprocessing_jd import (
    cat_variables_selected,
    name_code_to_replace_map,
    name_missing_code_map,
    num_variables_selected,
    preprocessing,
)
from train_baseline import fit_logistic_baseline


ROOT = Path(__file__).resolve().parent

def fit_logistic_baseline(matrix, y, max_iters=10000, tolerance=1e-5, check_every=250):
    """Train via the course GD implementation, checking the training gradient.

    Logistic loss has Hessian bounded by X.T @ X / (4*n). Taking gamma=1/L,
    with L the largest eigenvalue of that bound, gives a training-derived step
    size. Validation data is not used for this choice or for stopping.
    """
    if max_iters <= 0 or tolerance <= 0 or check_every <= 0:
        raise ValueError("Iterations, tolerance, and check interval must be positive.")
    lipschitz = float(np.linalg.eigvalsh(matrix.T @ matrix / (4 * len(y)))[-1])
    gamma = 2 / lipschitz
    weights = np.zeros(matrix.shape[1])
    history = []
    iterations = 0
    while True:
        logits = matrix @ weights
        gradient = matrix.T @ (probabilities_from_logits(logits) - y) / len(y)
        gradient_norm = float(np.max(np.abs(gradient)))
        loss = binary_log_loss(y, logits)
        if not np.isfinite(loss) or not np.isfinite(gradient_norm):
            raise FloatingPointError("Training produced nonfinite values.")
        if history and loss > history[-1]["loss"] + 1e-10:
            raise RuntimeError(
                "Training loss increased despite the conservative step size."
            )
        history.append(
            {"iteration": iterations, "loss": loss, "gradient_max_abs": gradient_norm}
        )
        print(
            f"iteration {iterations:>5}: loss={loss:.8f}, max|gradient|={gradient_norm:.3g}",
            flush=True,
        )
        if gradient_norm <= tolerance or iterations >= max_iters:
            break
        steps = min(check_every, max_iters - iterations)
        # Chunking preserves the course function's full-batch gradient updates.
        weights, returned_loss = logistic_regression(y, matrix, weights, steps, gamma)
        if not np.isfinite(returned_loss):
            raise FloatingPointError(
                "The course implementation returned a nonfinite loss."
            )
        iterations += steps
    return weights, {
        "algorithm": "unregularized logistic regression, full-batch gradient descent",
        "initial_weights": "zeros",
        "gamma": gamma,
        "max_iterations": max_iters,
        "iterations": iterations,
        "gradient_tolerance": tolerance,
        "check_every": check_every,
        "converged": gradient_norm <= tolerance,
        "history": history,
    }


def evaluate_matrix(training_matrix, validation_matrix, y_train, y_validation, args):
    """Fit and score one completed matrix using the baseline model."""
    training = np.asarray(training_matrix, dtype=float)
    validation = np.asarray(validation_matrix, dtype=float)
    if training.ndim != 2 or validation.ndim != 2:
        raise ValueError("Preprocessing matrices must be 2D.")
    if training.shape[1] != validation.shape[1]:
        raise ValueError("Training and validation matrices need identical columns.")
    if training.shape[0] != len(y_train) or validation.shape[0] != len(y_validation):
        raise ValueError(
            "Each preprocessing matrix must have one row per corresponding label."
        )
    if not np.isfinite(training).all() or not np.isfinite(validation).all():
        raise ValueError("The preprocessing matrix contains nonfinite values.")

    y_train = np.asarray(y_train, dtype=float)
    y_validation = np.asarray(y_validation, dtype=float)

    weights, optimization = fit_logistic_baseline(
        training,
        y_train,
        max_iters=args.max_iters,
        tolerance=args.tolerance,
    )
    train_logits = training @ weights
    validation_logits = validation @ weights
    threshold = args.threshold
    train_predictions = probabilities_from_logits(train_logits) >= threshold
    validation_predictions = probabilities_from_logits(validation_logits) >= threshold

    training_metrics = classification_metrics(y_train, train_predictions)
    validation_metrics = classification_metrics(y_validation, validation_predictions)
    training_metrics["log_loss"] = binary_log_loss(y_train, train_logits)
    validation_metrics["log_loss"] = binary_log_loss(y_validation, validation_logits)

    majority_class = int(np.mean(y_train) >= 0.5)
    majority_metrics = classification_metrics(
        y_validation, np.full(len(y_validation), majority_class)
    )
    return {
        "matrix_shape": list(training_matrix.shape),
        "optimization": optimization,
        "training": training_metrics,
        "validation": validation_metrics,
        "majority_class": majority_class,
        "majority_validation": majority_metrics,
    }


def summarize_validation(results):
    """Return mean and standard deviation for scalar validation metrics."""
    metric_names = ("accuracy", "precision", "recall", "f1", "log_loss")
    return {
        name: {
            "mean": float(np.mean([result["validation"][name] for result in results])),
            "std": float(np.std([result["validation"][name] for result in results])),
        }
        for name in metric_names
    }


def run(args):
    """Preprocess the training data, fit models, print metrics, and save a report."""
    start = time.perf_counter()
    training_matrices, validation_matrices, y_train, y_validation, column_names = preprocessing(
        args.data_dir,
        num_variables_selected,
        cat_variables_selected,
        name_missing_code_map,
        name_code_to_replace_map,
        validation_fraction=args.validation_fraction,
        validation_seed=args.seed,
    )

    results = [
        evaluate_matrix(training, validation, y_train, y_validation, args)
        for training, validation in zip(training_matrices, validation_matrices)
    ]
    report = {
        "preprocessing": "src.preprocessing_jd.preprocessing",
        "data_dir": str(args.data_dir),
        "matrix_count": len(training_matrices),
        "model_columns": len(column_names),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "training_rows": len(y_train),
        "validation_rows": len(y_validation),
        "threshold": args.threshold,
        "results": results,
        "validation_summary": summarize_validation(results),
        "metric_convention": (
            "0/1 labels; confusion rows actual [0,1], columns predicted [0,1]; "
            "undefined precision/recall/F1=0"
        ),
        "evaluation_note": (
            "The holdout split is made before MICE and category levels are "
            "fitted, so validation feature rows are transformed without fitting "
            "the preprocessing models on them."
        ),
        "elapsed_seconds": time.perf_counter() - start,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "preprocessed_jd_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print("\nValidation results (positive class = 1; cutoff = %.2f):" % args.threshold)
    print(f"{'Matrix':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for index, result in enumerate(results, start=1):
        metrics = result["validation"]
        print(
            f"{index:<12}"
            + "".join(
                f"{metrics[name]:>10.4f}"
                for name in ("accuracy", "precision", "recall", "f1")
            )
        )
    print("Mean/std:", report["validation_summary"])
    print("Saved:", report_path)
    return report


def main():
    """Command-line entry point; paths default to this repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "preprocessed_jd"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
