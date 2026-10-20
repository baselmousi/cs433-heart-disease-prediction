"""Implementation of cross-validation for CS-433 Project 1"""



import numpy as np
from src.metrics import THRESHOLDS, best_threshold, classification_metrics
from src.preprocessing import fit_preprocessor, transform_features


def stratified_split_indices(labels, validation_fraction=0.20, seed=42):
    """Return the same class-stratified row split used in the exploration notebook."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not 0 < validation_fraction < 1:
        raise ValueError(
            "Expected 1D labels and a validation fraction between 0 and 1."
        )
    rng = np.random.default_rng(seed)
    training_parts, validation_parts = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        n_validation = int(round(validation_fraction * len(indices)))
        if not 0 < n_validation < len(indices):
            raise ValueError("Each class needs examples in both portions.")
        validation_parts.append(indices[:n_validation])
        training_parts.append(indices[n_validation:])
    if not training_parts:
        raise ValueError("Cannot split empty labels.")
    return (
        rng.permutation(np.concatenate(training_parts)),
        rng.permutation(np.concatenate(validation_parts)),
    )


def stratified_kfold_indices(labels, k=5, seed=42):
    """Assign every row to one of k folds, with the same class ratio in each fold.

    Returns an integer array: row i is held out in fold folds[i].
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or k < 2:
        raise ValueError("Expected nonempty 1D labels and at least 2 folds.")
    rng = np.random.default_rng(seed)
    folds = np.empty(len(labels), dtype=int)
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < k:
            raise ValueError("Each class needs at least k examples.")
        folds[indices] = np.arange(len(indices)) % k
    return folds


def cross_validate(
    x,
    labels,
    feature_names,
    fit_predict,
    k=5,
    seed=42,
    thresholds=THRESHOLDS,
    **options
):
    """Stratified k-fold CV of the full pipeline, scored by positive-class F1.

    Preprocessing is refit on the training folds every time; options are passed to
    fit_preprocessor. fit_predict(train_matrix, train_labels, held_out_matrix) must
    return held-out probabilities, and receives 0/1 labels. The F1 cutoff is tuned
    once on the pooled out-of-fold probabilities, then each fold is scored with it.
    """
    y = (np.asarray(labels) == 1).astype(float)  # accepts -1/+1 or 0/1 labels
    folds = stratified_kfold_indices(y, k, seed)
    probabilities = np.empty(len(y))
    columns = []
    for fold in range(k):
        train, held_out = folds != fold, folds == fold
        state = fit_preprocessor(x[train], feature_names, **options)
        matrix = transform_features(x[train], feature_names, state)
        probabilities[held_out] = fit_predict(
            matrix,
            y[train].astype(matrix.dtype),
            transform_features(x[held_out], feature_names, state),
        )
        columns.append(len(state["columns"]))
    threshold, pooled_f1 = best_threshold(y, probabilities, thresholds)
    predictions = probabilities >= threshold
    per_fold = [
        classification_metrics(y[folds == f], predictions[folds == f])["f1"]
        for f in range(k)
    ]
    return {
        "threshold": threshold,
        "f1_mean": float(np.mean(per_fold)),
        "f1_std": float(np.std(per_fold)),
        "f1_per_fold": per_fold,
        "f1_pooled": pooled_f1,
        "columns_per_fold": columns,
        "probabilities": probabilities,
        "folds": folds,
    }

