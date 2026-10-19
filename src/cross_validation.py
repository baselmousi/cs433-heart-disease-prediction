"""Implementation of cross-validation for CS-433 Project 1"""

# The current baseline uses one reproducible stratified holdout split.

import numpy as np


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
