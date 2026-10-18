"""Fit and evaluate the five-feature logistic baseline; no notebook required."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from implementations import logistic_regression
from src.cross_validation import stratified_split_indices
from src.metrics import (
    binary_log_loss,
    classification_metrics,
    probabilities_from_logits,
)
from src.preprocessing import (
    BASELINE_FEATURES,
    baseline_column_names,
    fit_baseline_preprocessor,
    load_baseline_training_data,
    transform_baseline_features,
)

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
    gamma = 1.0 / lipschitz
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


def run(args):
    """Load, split, fit preprocessing/model, evaluate, and save reproducible artifacts."""
    start = time.perf_counter()
    print("Loading five features and checking label alignment...", flush=True)
    raw, labels, _ = load_baseline_training_data(args.data_dir)
    train_idx, val_idx = stratified_split_indices(labels, 0.20, 42)
    state = fit_baseline_preprocessor(raw[train_idx])
    training = transform_baseline_features(raw[train_idx], state)
    validation = transform_baseline_features(raw[val_idx], state)
    y_train = (labels[train_idx] == 1).astype(float)
    y_val = (labels[val_idx] == 1).astype(float)
    print(f"Training: {training.shape}; validation: {validation.shape}", flush=True)
    weights, optimization = fit_logistic_baseline(
        training, y_train, args.max_iters, args.tolerance
    )
    train_logits, val_logits = training @ weights, validation @ weights
    threshold = 0.5
    train_metrics = classification_metrics(
        y_train, probabilities_from_logits(train_logits) >= threshold
    )
    val_metrics = classification_metrics(
        y_val, probabilities_from_logits(val_logits) >= threshold
    )
    train_metrics["log_loss"] = binary_log_loss(y_train, train_logits)
    val_metrics["log_loss"] = binary_log_loss(y_val, val_logits)
    majority_class = int(np.mean(y_train) >= 0.5)
    majority_metrics = classification_metrics(
        y_val, np.full(len(y_val), majority_class)
    )
    columns = baseline_column_names(state)
    report = {
        "numpy_version": np.__version__,
        "features": list(BASELINE_FEATURES),
        "model_columns": columns,
        "seed": 42,
        "validation_fraction": 0.20,
        "training_rows": len(train_idx),
        "validation_rows": len(val_idx),
        "training_positive_count": int(y_train.sum()),
        "validation_positive_count": int(y_val.sum()),
        "threshold": threshold,
        "optimization": optimization,
        "preprocessing": {k: v.tolist() for k, v in state.items()},
        "weights": weights.tolist(),
        "training": train_metrics,
        "validation": val_metrics,
        "majority_class": majority_class,
        "majority_validation": majority_metrics,
        "metric_convention": "0/1 labels; confusion rows actual [0,1], columns predicted [0,1]; undefined precision/recall/F1=0",
        "evaluation_note": "Fixed initial baseline; no validation threshold tuning. Earlier exploration had viewed the full labeled data.",
        "elapsed_seconds": time.perf_counter() - start,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "logistic_model.npz"
    np.savez_compressed(
        model_path,
        weights=weights,
        threshold=threshold,
        seed=42,
        validation_fraction=0.20,
        feature_names=np.array(BASELINE_FEATURES),
        column_names=np.array(columns),
        **state,
    )
    # Ensure the persisted model reproduces probabilities, not just its array shapes.
    with np.load(model_path, allow_pickle=False) as saved:
        saved_state = {k: saved[k] for k in state}
        restored = (
            transform_baseline_features(raw[val_idx], saved_state) @ saved["weights"]
        )
        if not np.array_equal(restored, val_logits):
            raise RuntimeError("Saved model did not reproduce validation scores.")
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("\nValidation results (positive class = 1; cutoff = 0.5):")
    print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for name, metrics in (
        ("Majority class", majority_metrics),
        ("Logistic regression", val_metrics),
    ):
        print(
            f"{name:<22}"
            + "".join(
                f"{metrics[k]:>10.4f}"
                for k in ("accuracy", "precision", "recall", "f1")
            )
        )
    print("Confusion matrix [[TN, FP], [FN, TP]]:", val_metrics["confusion_matrix"])
    print(
        f"Converged: {optimization['converged']}; training loss: {train_metrics['log_loss']:.6f}; validation loss: {val_metrics['log_loss']:.6f}"
    )
    print("Saved:", model_path, "and", report_path)
    return report


def main():
    """Command-line entry point; paths default to this repository, not the launch cwd."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "baseline"
    )
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
