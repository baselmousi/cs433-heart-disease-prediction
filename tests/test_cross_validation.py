"""Checks for stratified k-fold CV and the F1 threshold search."""

import unittest

import numpy as np

from src import feature_spec as fs
from src.cross_validation import cross_validate, stratified_kfold_indices
from src.metrics import best_threshold


def synthetic_table(n, seed):
    """A table with every SPEC feature, filled with valid survey codes."""
    rng = np.random.default_rng(seed)
    names = list(fs.SPEC)
    x = rng.integers(1, 5, size=(n, len(names))).astype(float)  # codes 1-4
    for j, name in enumerate(names):
        if fs.SPEC[name][0] == "binary":
            x[:, j] = rng.integers(1, 3, size=n)  # 1 yes, 2 no
        if fs.SPEC[name][0] == "continuous":
            x[:, j] = rng.normal(50, 10, size=n)  # makes every row unique
    return x, names


def least_squares_probabilities(matrix, labels, held_out):
    """A fast stand-in model: least squares scores clipped to [0, 1]."""
    w = np.linalg.lstsq(matrix, labels, rcond=None)[0]
    return np.clip(held_out @ w, 0, 1)


class KFoldTests(unittest.TestCase):
    def test_folds_are_stratified_and_repeatable(self):
        labels = np.r_[np.full(83, -1), np.ones(17)]
        folds = stratified_kfold_indices(labels, k=5, seed=42)
        np.testing.assert_array_equal(folds, stratified_kfold_indices(labels, 5, 42))
        self.assertFalse(np.array_equal(folds, stratified_kfold_indices(labels, 5, 0)))
        for fold in range(5):
            held_out = labels[folds == fold]
            self.assertIn(np.count_nonzero(held_out == -1), (16, 17))
            self.assertIn(np.count_nonzero(held_out == 1), (3, 4))
        self.assertEqual(set(folds), set(range(5)))

    def test_invalid_inputs_are_rejected(self):
        labels = np.r_[np.zeros(20), np.ones(3)]
        with self.assertRaisesRegex(ValueError, "at least k"):
            stratified_kfold_indices(labels, k=5)
        for bad in (labels.reshape(1, -1), np.array([])):
            with self.assertRaises(ValueError):
                stratified_kfold_indices(bad)
        with self.assertRaises(ValueError):
            stratified_kfold_indices(labels, k=1)


class ThresholdTests(unittest.TestCase):
    def test_best_threshold_maximizes_f1_and_prefers_the_lowest_tie(self):
        y = np.array([0, 0, 1, 1, 1])
        p = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
        # 0.3: TP 3, FP 1, FN 0 -> F1 6/7.  0.5: TP 2, FP 0, FN 1 -> F1 4/5.
        threshold, f1 = best_threshold(y, p, [0.3, 0.5])
        self.assertEqual(threshold, 0.3)
        self.assertAlmostEqual(f1, 6 / 7)
        self.assertEqual(best_threshold(y, p, [0.2, 0.3])[0], 0.2)
        with self.assertRaises(ValueError):
            best_threshold(y, p, [])


class CrossValidateTests(unittest.TestCase):
    def setUp(self):
        self.x, self.names = synthetic_table(1000, seed=0)

    def test_no_row_is_scored_by_a_model_that_saw_it(self):
        labels = np.random.default_rng(1).integers(0, 2, size=len(self.x))
        training_sizes = []

        def memorize(matrix, train_labels, held_out):
            # Scores 0.5 unless the held-out row also appeared in training.
            seen = {row.tobytes(): label for row, label in zip(matrix, train_labels)}
            training_sizes.append(len(matrix))
            return np.array([seen.get(row.tobytes(), 0.5) for row in held_out])

        result = cross_validate(self.x, labels, self.names, memorize, k=4)
        np.testing.assert_array_equal(result["probabilities"], 0.5)
        folds = stratified_kfold_indices(labels.astype(float), 4, 42)
        np.testing.assert_array_equal(result["folds"], folds)
        self.assertEqual(training_sizes, [np.sum(folds != f) for f in range(4)])

    def test_a_planted_signal_is_recovered_with_minus_one_plus_one_labels(self):
        medcost = self.x[:, self.names.index("MEDCOST")]
        labels = np.where(medcost == 1, 1, -1)  # the label is one binary answer
        result = cross_validate(self.x, labels, self.names, least_squares_probabilities)
        self.assertEqual(len(result["f1_per_fold"]), 5)
        self.assertGreater(result["f1_mean"], 0.95)
        self.assertAlmostEqual(
            result["f1_mean"], np.mean(result["f1_per_fold"]), places=12
        )

    def test_preprocessing_options_reach_every_fold(self):
        labels = np.random.default_rng(2).integers(0, 2, size=len(self.x))
        self.x[np.random.default_rng(3).random(self.x.shape) < 0.05] = np.nan
        base = cross_validate(self.x, labels, self.names, least_squares_probabilities)
        more = cross_validate(
            self.x, labels, self.names, least_squares_probabilities, indicators=True
        )
        for a, b in zip(base["columns_per_fold"], more["columns_per_fold"]):
            self.assertGreater(b, a)


if __name__ == "__main__":
    unittest.main()