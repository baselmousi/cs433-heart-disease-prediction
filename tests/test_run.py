"""Checks for run.py, the script that writes the AIcrowd submission."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from run import check_submission, main, predict
from src import feature_spec as fs


def synthetic_table(n, seed):
    """A table with every SPEC feature, filled with valid survey codes."""
    rng = np.random.default_rng(seed)
    names = list(fs.SPEC)
    x = rng.integers(1, 5, size=(n, len(names))).astype(float)  # codes 1-4
    for j, name in enumerate(names):
        if fs.SPEC[name][0] == "binary":
            x[:, j] = rng.integers(1, 3, size=n)  # 1 yes, 2 no
        if fs.SPEC[name][0] == "continuous":
            x[:, j] = rng.normal(50, 10, size=n)
    return x, names


def planted_labels(x, names):
    """The label is one binary answer, so a working pipeline gets it right."""
    return np.where(x[:, names.index("MEDCOST")] == 1, 1, -1)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class PredictTests(unittest.TestCase):
    def setUp(self):
        self.x_train, self.names = synthetic_table(1000, seed=0)
        self.x_test, _ = synthetic_table(300, seed=1)
        self.labels = planted_labels(self.x_train, self.names)

    def test_predictions_are_minus_one_plus_one_and_recover_the_signal(self):
        predictions, probabilities, threshold = predict(
            "E1-logistic", self.x_train, self.labels, self.x_test, self.names
        )
        self.assertEqual(set(np.unique(predictions)), {-1, 1})
        self.assertEqual(len(probabilities), len(self.x_test))
        np.testing.assert_array_equal(predictions == 1, probabilities >= threshold)
        expected = planted_labels(self.x_test, self.names)
        self.assertGreater(np.mean(predictions == expected), 0.95)

    def test_a_fixed_threshold_is_used_as_given(self):
        _, probabilities, threshold = predict(
            "E1-logistic", self.x_train, self.labels, self.x_test, self.names, 0.37
        )
        self.assertEqual(threshold, 0.37)
        np.testing.assert_array_equal(
            predict(
                "E1-logistic",
                self.x_train,
                self.labels,
                self.x_test,
                self.names,
                0.37,
            )[1],
            probabilities,
        )  # same inputs, same probabilities: the submission is reproducible


class MainTests(unittest.TestCase):
    def test_writes_a_submission_with_every_test_id(self):
        x_train, names = synthetic_table(600, seed=2)
        x_test, _ = synthetic_table(200, seed=3)
        x_test[::7, 0] = np.nan  # empty fields in the test file must load as NaN
        train_ids = np.arange(len(x_train))
        test_ids = np.arange(len(x_train), len(x_train) + len(x_test))
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            header = ["Id", *names]
            write_csv(
                data / "x_train.csv", header, np.column_stack([train_ids, x_train])
            )
            rows = [
                [i, *("" if np.isnan(v) else v for v in row)]
                for i, row in zip(test_ids, x_test)
            ]
            write_csv(data / "x_test.csv", header, rows)
            labels = planted_labels(x_train, names)
            write_csv(data / "y_train.csv", ["Id", "_MICHD"], zip(train_ids, labels))
            output = data / "out" / "submission.csv"
            main(["--data-dir", str(data), "--output", str(output)])
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([int(r["Id"]) for r in rows], list(test_ids))
        self.assertTrue({r["Prediction"] for r in rows} <= {"-1", "1"})

    def test_mismatched_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            write_csv(data / "x_train.csv", ["Id", "A", "B"], [[0, 1, 2]])
            write_csv(data / "x_test.csv", ["Id", "B", "A"], [[1, 2, 1]])
            write_csv(data / "y_train.csv", ["Id", "_MICHD"], [[0, 1]])
            with self.assertRaisesRegex(ValueError, "different columns"):
                main(["--data-dir", str(data), "--threshold", "0.2"])

    def test_nan_probabilities_stop_the_run(self):
        x_train, names = synthetic_table(300, seed=4)
        labels = planted_labels(x_train, names)
        nan_model = lambda matrix, labels, held_out: np.full(len(held_out), np.nan)
        with mock.patch.dict("run.MODELS", {"logistic": lambda **_: nan_model}):
            with self.assertRaisesRegex(ValueError, "not finite"):
                predict("E1-logistic", x_train, labels, x_train, names, 0.2)


class CheckSubmissionTests(unittest.TestCase):
    def check(self, rows, test_ids=(5, 6), sample_ids=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            write_csv(path, rows[0], rows[1:])
            sample = None
            if sample_ids is not None:
                sample = Path(directory) / "sample_submission.csv"
                write_csv(sample, ["Id", "Prediction"], [[i, -1] for i in sample_ids])
            check_submission(path, test_ids, sample)

    def test_a_valid_file_passes(self):
        self.check([["Id", "Prediction"], [5, 1], [6, -1]], sample_ids=[5, 6])

    def test_wrong_header_ids_values_or_sample_ids_are_rejected(self):
        cases = {
            "header": [["id", "pred"], [5, 1], [6, -1]],
            "x_test IDs": [["Id", "Prediction"], [6, 1], [5, -1]],
            "other than": [["Id", "Prediction"], [5, 0], [6, -1]],
        }
        for message, rows in cases.items():
            with self.assertRaisesRegex(ValueError, message):
                self.check(rows)
        with self.assertRaisesRegex(ValueError, "differ from"):
            self.check([["Id", "Prediction"], [5, 1], [6, -1]], sample_ids=[5, 7])


if __name__ == "__main__":
    unittest.main()
