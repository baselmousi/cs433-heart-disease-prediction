"""Checks for the paired fold-by-fold comparison and the --seed option."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare_cv import compare, load_run
from run_cv import run_name
from src.cross_validation import (
    cross_validate,
    paired_comparison,
    per_fold_f1,
    stratified_kfold_indices,
)
from test_cross_validation import least_squares_probabilities, synthetic_table


class PerFoldTests(unittest.TestCase):
    def test_saved_probabilities_reproduce_the_cross_validate_scores(self):
        x, names = synthetic_table(600, seed=1)
        labels = np.where(x[:, names.index("MEDCOST")] == 1, 1, -1)
        for seed in (42, 7):
            result = cross_validate(
                x, labels, names, least_squares_probabilities, seed=seed
            )
            folds = stratified_kfold_indices((labels == 1).astype(float), seed=seed)
            np.testing.assert_array_equal(folds, result["folds"])
            threshold, scores = per_fold_f1(labels, result["probabilities"], folds)
            self.assertEqual(threshold, result["threshold"])
            np.testing.assert_allclose(scores, result["f1_per_fold"])


class PairedComparisonTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.labels = np.r_[np.ones(200), -np.ones(1800)]
        self.folds = stratified_kfold_indices(self.labels, k=5, seed=42)
        # Independent noise, so the two runs rank the rows differently.
        positive = (self.labels == 1).astype(float)
        self.weak = np.clip(positive + rng.normal(0, 0.8, len(positive)), 0, 1)
        self.strong = np.clip(positive + rng.normal(0, 0.3, len(positive)), 0, 1)

    def test_a_run_against_itself_has_no_difference(self):
        result = paired_comparison(self.labels, self.weak, self.weak, self.folds)
        np.testing.assert_array_equal(result["differences"], 0)
        self.assertEqual(result["t"], 0)
        self.assertEqual(result["verdict"], "no clear difference")

    def test_a_clearly_better_run_wins_and_the_sign_flips_when_swapped(self):
        better = paired_comparison(self.labels, self.weak, self.strong, self.folds)
        worse = paired_comparison(self.labels, self.strong, self.weak, self.folds)
        self.assertEqual(better["verdict"], "better")
        self.assertEqual(worse["verdict"], "worse")
        self.assertAlmostEqual(better["mean"], -worse["mean"])
        self.assertEqual(better["folds_better"], 5)

    def test_corrected_standard_error_is_wider_than_the_naive_one(self):
        result = paired_comparison(self.labels, self.weak, self.strong, self.folds)
        naive = np.std(result["differences"], ddof=1) / np.sqrt(5)
        self.assertAlmostEqual(result["se"], naive * np.sqrt(0.45 / 0.2))


class SavedRunTests(unittest.TestCase):
    def test_seed_42_keeps_the_plain_name(self):
        self.assertEqual(run_name("E1-logistic", 42), "E1-logistic")
        self.assertEqual(run_name("E1-logistic", 43), "E1-logistic-seed43")

    def test_files_without_a_seed_count_as_42_and_mismatches_are_refused(self):
        labels = np.r_[np.ones(10), -np.ones(40)]
        probabilities = np.linspace(0, 1, 50)
        with tempfile.TemporaryDirectory() as directory:
            np.savez(
                Path(directory) / "A.npz", probabilities=probabilities, labels=labels
            )
            np.savez(
                Path(directory) / "A-seed43.npz",
                probabilities=probabilities,
                labels=labels,
                seed=42,
            )
            loaded, _ = load_run(directory, "A", 42)
            np.testing.assert_array_equal(loaded, probabilities)
            with self.assertRaisesRegex(ValueError, "seed 42, not 43"):
                load_run(directory, "A", 43)
            with self.assertRaises(FileNotFoundError):
                load_run(directory, "B", 42)
            np.savez(
                Path(directory) / "B.npz", probabilities=probabilities, labels=labels
            )
            means = compare(directory, ["A", "B"], 42)
        self.assertEqual(means["A"], means["B"])


if __name__ == "__main__":
    unittest.main()
