"""Checks that the logistic helpers in implementations.py stay finite when saturated."""

import unittest
import warnings

import numpy as np

from implementations import _logistic_loss, _sigmoid, reg_logistic_regression


class StableLogisticTests(unittest.TestCase):
    def test_sigmoid_matches_the_textbook_formula_and_never_overflows(self):
        t = np.linspace(-30, 30, 61)
        np.testing.assert_allclose(_sigmoid(t), 1 / (1 + np.exp(-t)), rtol=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # an overflow warning fails the test
            np.testing.assert_array_equal(_sigmoid(np.array([-1000.0, 1000.0])), [0, 1])

    def test_loss_matches_the_textbook_formula_and_stays_finite_when_saturated(self):
        rng = np.random.default_rng(0)
        tx, y, w = rng.normal(size=(50, 3)), rng.integers(0, 2, 50), rng.normal(size=3)
        p = 1 / (1 + np.exp(-(tx @ w)))
        expected = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        self.assertAlmostEqual(_logistic_loss(y, tx, w), expected, places=12)
        # One confidently wrong row: the loss is its margin (1000 / 2 rows), not inf.
        loss = _logistic_loss(
            np.array([0, 1]), np.array([[1.0], [1.0]]), np.array([1000.0])
        )
        self.assertAlmostEqual(loss, 500.0)

    def test_large_steps_do_not_produce_nan_weights(self):
        rng = np.random.default_rng(1)
        tx = np.c_[np.ones(100), 50 * rng.normal(size=(100, 2))]  # unscaled features
        y = (tx[:, 1] > 0).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            w, loss = reg_logistic_regression(y, tx, 0.0, np.zeros(3), 100, 1.0)
        self.assertTrue(np.all(np.isfinite(w)) and np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
