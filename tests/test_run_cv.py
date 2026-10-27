"""Checks for the experiment runner in run_cv.py and the CSV loader."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from run_cv import (
    BASE,
    COLUMNS,
    EXPERIMENTS,
    append_row,
    logistic,
    run_experiment,
    subsampled_logistic,
    undersampled_logistic,
    weighted_logistic,
)
from src import feature_spec as fs
from src.preprocessing import load_csv


def synthetic_table(n, seed):
    """A table with every SPEC and LEAKY feature, filled with valid survey codes."""
    rng = np.random.default_rng(seed)
    names = list(fs.SPEC) + list(fs.LEAKY)
    x = rng.integers(1, 5, size=(n, len(names))).astype(float)  # codes 1-4
    for j, name in enumerate(names):
        if name in fs.SPEC and fs.SPEC[name][0] == "binary":
            x[:, j] = rng.integers(1, 3, size=n)  # 1 yes, 2 no
        if name in fs.SPEC and fs.SPEC[name][0] == "continuous":
            x[:, j] = rng.normal(50, 10, size=n)
    return x, names


def short(experiment, max_iters=20_000):
    """The same experiment with at most max_iters steps, so tests stay fast."""
    question, model, params, options = experiment
    if params.get("max_iters", 0) > max_iters:
        params = {**params, "max_iters": max_iters}
    return question, model, params, options


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.x, self.names = synthetic_table(1000, seed=0)
        medcost = self.x[:, self.names.index("MEDCOST")]
        self.labels = np.where(medcost == 1, 1, -1)  # the label is one binary answer

    def test_every_experiment_runs_and_fills_its_row(self):
        for name in EXPERIMENTS:
            with mock.patch.dict(EXPERIMENTS, {name: short(EXPERIMENTS[name])}):
                row, probabilities = run_experiment(
                    name, self.x, self.labels, self.names
                )
            self.assertEqual(row["experiment"], name)
            self.assertEqual(len(probabilities), len(self.x))
            missing = set(COLUMNS) - set(row) - {"owner"}
            self.assertFalse(missing, f"{name} row lacks {missing}")

    def test_models_recover_a_planted_signal_and_majority_scores_zero(self):
        names = ("E1-logistic", "E1-least-squares", "E1-ridge", "E1-mse-gd")
        names += ("E1-mse-sgd", "E1-logistic-plain", "E0-raw-logistic", "E2-raw-codes")
        for name in names:
            with mock.patch.dict(EXPERIMENTS, {name: short(EXPERIMENTS[name])}):
                row, _ = run_experiment(name, self.x, self.labels, self.names)
            self.assertGreater(row["f1_mean"], 0.95, name)
        row, _ = run_experiment("E0-majority", self.x, self.labels, self.names)
        self.assertEqual(row["f1_mean"], 0)

    def test_imbalance_and_subsampled_models_reduce_to_plain_logistic(self):
        rng = np.random.default_rng(1)
        matrix = np.c_[np.ones(300), rng.normal(size=(300, 4))]
        labels = (matrix[:, 1] + rng.normal(size=300) > 1.2).astype(float)
        held_out = np.c_[np.ones(50), rng.normal(size=(50, 4))]
        plain = logistic(**BASE)(matrix, labels, held_out)
        np.testing.assert_allclose(
            weighted_logistic(**BASE, positive_weight=1)(matrix, labels, held_out),
            plain,
        )
        everything = 10 * int(np.sum(labels == 0))  # more than there are negatives
        np.testing.assert_allclose(
            undersampled_logistic(**BASE, negatives_per_positive=everything)(
                matrix, labels, held_out
            ),
            plain,
        )
        np.testing.assert_allclose(
            subsampled_logistic(**BASE, fraction=1.0)(matrix, labels, held_out), plain
        )

    def test_odds_correction_keeps_probabilities_on_the_natural_scale(self):
        rng = np.random.default_rng(2)
        matrix = np.c_[np.ones(2000), rng.normal(size=(2000, 3))]
        labels = (rng.random(2000) < 0.1).astype(float)  # no signal, 10 % positive
        for model in (
            weighted_logistic(**BASE, positive_weight="balanced"),
            undersampled_logistic(**BASE, negatives_per_positive=1),
        ):
            probabilities = model(matrix, labels, matrix)
            self.assertAlmostEqual(probabilities.mean(), labels.mean(), delta=0.02)

    def test_log_gets_one_header_and_rows_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results" / "results.csv"
            append_row(path, {"experiment": "A", "f1_mean": 0.1})
            append_row(path, {"experiment": "B", "f1_mean": 0.2})
            with path.open(newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0], COLUMNS)
        experiment = COLUMNS.index("experiment")
        self.assertEqual([r[experiment] for r in rows[1:]], ["A", "B"])


class LoadCsvTests(unittest.TestCase):
    def test_empty_fields_become_nan_like_genfromtxt(self):
        text = "Id,a,b,c\n1,,2,\n2,,,3\n3,4.5,,\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.csv"
            path.write_text(text)
            expected = np.genfromtxt(path, delimiter=",", skip_header=1)
            loaded = load_csv(path, chunk_rows=2)  # also crosses a chunk boundary
        np.testing.assert_array_equal(loaded, expected)
        self.assertEqual(np.isnan(loaded).sum(), 6)


if __name__ == "__main__":
    unittest.main()
