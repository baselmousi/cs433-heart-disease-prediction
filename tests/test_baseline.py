"""Small checks for leakage, survey decoding, and evaluation semantics."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

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


class BaselineTests(unittest.TestCase):
    def test_fixed_training_statistics_and_missing_zero_distinction(self):
        training = np.array(
            [
                [1, 88, 4, 20, 20],
                [2, 88, 3, 25, 40],
                [3, 10, 2, 30, 60],
                [7, 77, 9, np.nan, 80],
            ],
            dtype=float,
        )
        original = training.copy()
        state = fit_baseline_preprocessor(training)
        snapshot = {key: value.copy() for key, value in state.items()}
        # A genuine zero and an unknown imputed to zero must remain distinguishable.
        prepared = transform_baseline_features(training, state)
        flag = baseline_column_names(state).index("PHYSHLTH_days_missing")
        self.assertEqual(prepared[0, flag], 0)
        self.assertEqual(prepared[3, flag], 1)
        # Training median BMI=25, filled mean=25, std=sqrt(12.5).
        held_out = np.array([[5, 30, 1, 40, 80]], dtype=float)
        held_out_matrix = transform_baseline_features(held_out, state)
        self.assertAlmostEqual(held_out_matrix[0, 11], 15 / np.sqrt(12.5))
        for key in state:
            np.testing.assert_array_equal(state[key], snapshot[key])
        np.testing.assert_array_equal(training, original)
        # Double-dividing an already decoded BMI must fail rather than silently fit.
        bad_units = held_out.copy()
        bad_units[0, 3] /= 100
        with self.assertRaisesRegex(ValueError, "BMI"):
            transform_baseline_features(bad_units, state)

    def test_label_alignment_checked_even_when_counts_match(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            with (data / "y_train.csv").open("w", newline="") as handle:
                csv.writer(handle).writerows([["Id", "_MICHD"], [10, -1], [20, 1]])
            with (data / "x_train.csv").open("w", newline="") as handle:
                csv.writer(handle).writerows(
                    [
                        ["Id", *BASELINE_FEATURES],
                        [20, 1, 88, 4, 25, 40],
                        [10, 5, 30, 1, 35, 70],
                    ]
                )
            with self.assertRaisesRegex(ValueError, "IDs differ"):
                load_baseline_training_data(data)

    def test_stratified_split_is_disjoint_complete_and_repeatable(self):
        labels = np.r_[np.full(80, -1), np.ones(20)]
        train, val = stratified_split_indices(labels)
        train_again, val_again = stratified_split_indices(labels)
        np.testing.assert_array_equal(train, train_again)
        np.testing.assert_array_equal(val, val_again)
        np.testing.assert_array_equal(np.sort(np.r_[train, val]), np.arange(100))
        self.assertEqual(np.intersect1d(train, val).size, 0)
        self.assertEqual(np.count_nonzero(labels[val] == 1), 4)

    def test_confusion_matrix_and_majority_metrics(self):
        actual = np.array([0, 0, 0, 1, 1, 1, 1])
        predictions = np.array([0, 0, 1, 0, 0, 1, 1])
        result = classification_metrics(actual, predictions)
        self.assertEqual(result["confusion_matrix"], [[2, 1], [2, 2]])
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 1 / 2)
        self.assertAlmostEqual(result["f1"], 4 / 7)
        majority = classification_metrics(actual, np.zeros(len(actual)))
        self.assertEqual(majority["precision"], 0)
        self.assertEqual(majority["recall"], 0)
        self.assertEqual(majority["f1"], 0)

    def test_extreme_scores_do_not_overflow_evaluation(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            np.testing.assert_allclose(
                probabilities_from_logits([-1000, 0, 1000]), [0, 0.5, 1]
            )
            self.assertEqual(
                binary_log_loss(np.array([1, 0]), np.array([-1000, 1000])), 1000
            )


if __name__ == "__main__":
    unittest.main()
