"""Checks for the experiment runner in run_cv.py and the CSV loader."""

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_cv import COLUMNS, EXPERIMENTS, append_row, run_experiment
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


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.x, self.names = synthetic_table(1000, seed=0)
        medcost = self.x[:, self.names.index("MEDCOST")]
        self.labels = np.where(medcost == 1, 1, -1)  # the label is one binary answer

    def test_every_experiment_runs_and_fills_its_row(self):
        for name in EXPERIMENTS:
            row, probabilities = run_experiment(name, self.x, self.labels, self.names)
            self.assertEqual(row["experiment"], name)
            self.assertEqual(len(probabilities), len(self.x))
            missing = set(COLUMNS) - set(row) - {"owner", "git_commit"}
            self.assertFalse(missing, f"{name} row lacks {missing}")

    def test_models_recover_a_planted_signal_and_majority_scores_zero(self):
        for name in ("E1-logistic", "E1-least-squares", "E1-ridge"):
            row, _ = run_experiment(name, self.x, self.labels, self.names)
            self.assertGreater(row["f1_mean"], 0.95, name)
        row, _ = run_experiment("E0-majority", self.x, self.labels, self.names)
        self.assertEqual(row["f1_mean"], 0)

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