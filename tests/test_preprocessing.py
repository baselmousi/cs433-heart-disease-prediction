"""Checks for the full pipeline in src/preprocessing.py, on small synthetic tables."""

import unittest

import numpy as np

from src import feature_spec as fs
from src.preprocessing import clean_columns, fit_preprocessor, transform_features


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
    x[rng.random(x.shape) < 0.05] = np.nan
    return x, names


class FullPipelineTests(unittest.TestCase):
    def setUp(self):
        self.x, self.names = synthetic_table(2000, seed=0)
        self.index = {name: j for j, name in enumerate(self.names)}

    def column(self, x, name):
        return x[:, self.index[name]]

    def test_codes_remaps_and_skip_fills(self):
        x = self.x[:4].copy()
        x[:, self.index["PHYSHLTH"]] = [88, 77, 5, 88]
        x[:, self.index["MENTHLTH"]] = [88, 88, 88, 3]
        x[:, self.index["POORHLTH"]] = np.nan
        original = x.copy()
        cleaned = clean_columns(x, self.names, ["PHYSHLTH", "POORHLTH"])
        np.testing.assert_array_equal(cleaned["PHYSHLTH"], [0, np.nan, 5, 0])
        # POORHLTH is skipped when both day counts are 0, so only row 0 becomes 0.
        np.testing.assert_array_equal(cleaned["POORHLTH"], [0, np.nan, np.nan, np.nan])
        np.testing.assert_array_equal(x, original)

    def test_statistics_come_from_training_rows_only(self):
        state = fit_preprocessor(self.x, self.names)
        snapshot = {"mean": state["mean"].copy(), "std": state["std"].copy()}
        held_out, _ = synthetic_table(300, seed=1)
        held_out[:, self.index["_STATE"]] = 99  # a level never seen in training
        prepared = transform_features(held_out, self.names, state)
        np.testing.assert_array_equal(state["mean"], snapshot["mean"])
        np.testing.assert_array_equal(state["std"], snapshot["std"])
        self.assertEqual(prepared.shape, (300, len(state["columns"])))
        self.assertTrue(np.isfinite(prepared).all())
        # The unseen level is encoded like the reference: all _STATE columns are 0.
        for name in state["columns"]:
            if name.startswith("_STATE="):
                j = state["columns"].index(name)
                np.testing.assert_allclose(
                    prepared[:, j], -state["mean"][j - 1] / state["std"][j - 1], 1e-5
                )

    def test_training_matrix_is_standardized_with_full_column_rank(self):
        state = fit_preprocessor(self.x, self.names)
        prepared = transform_features(self.x, self.names, state, dtype=np.float64)
        np.testing.assert_array_equal(prepared[:, 0], 1)
        np.testing.assert_allclose(prepared[:, 1:].mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(prepared[:, 1:].std(axis=0), 1, atol=1e-10)
        # One reference level per nominal feature keeps X.T @ X invertible.
        self.assertEqual(np.linalg.matrix_rank(prepared), prepared.shape[1])

    def test_skip_pattern_dependency_is_removed(self):
        # As in the real data: only former smokers (_SMOKER3 = 3) answer LASTSMK2.
        x = self.x.copy()
        former = self.column(x, "_SMOKER3") == 3
        x[~former, self.index["LASTSMK2"]] = np.nan
        x[former, self.index["LASTSMK2"]] = 1 + np.arange(former.sum()) % 3
        state = fit_preprocessor(x, self.names)
        prepared = transform_features(x, self.names, state, dtype=np.float64)
        self.assertEqual(len(state["dependent"]), 1)
        self.assertTrue(state["dependent"][0].startswith("LASTSMK2="))
        self.assertEqual(np.linalg.matrix_rank(prepared), prepared.shape[1])

    def test_options_add_the_expected_columns(self):
        base = fit_preprocessor(self.x, self.names)["columns"]
        with_indicators = fit_preprocessor(self.x, self.names, indicators=True)
        with_leaky = fit_preprocessor(self.x, self.names, leaky=True)
        self.assertFalse(any(name.endswith("_missing") for name in base))
        self.assertTrue(
            any(name.endswith("_missing") for name in with_indicators["columns"])
        )
        self.assertFalse(any(name.split("=")[0] in fs.LEAKY for name in base))
        self.assertIn("HAREHAB1", with_leaky["features"])

    def test_nonlinear_options_add_squares_bins_and_age_products(self):
        x = self.x.copy()
        x[:, self.index["_AGE80"]] = np.random.default_rng(3).integers(18, 81, len(x))
        x[:, self.index["_BMI5"]] = np.random.default_rng(4).uniform(16, 45, len(x))
        base = fit_preprocessor(x, self.names)["columns"]
        squares = fit_preprocessor(x, self.names, squares=True)["columns"]
        bins = fit_preprocessor(x, self.names, bins=True)["columns"]
        products = fit_preprocessor(x, self.names, interactions=True)["columns"]
        continuous = [n for n in fs.SPEC if fs.SPEC[n][0] == "continuous"]
        self.assertEqual(
            set(squares) - set(base), {f"{n}^2" for n in continuous if n in base}
        )
        self.assertEqual(
            set(bins) - set(base),
            {"_AGE80>=25", "_AGE80>=35", "_AGE80>=45", "_AGE80>=55", "_AGE80>=65"}
            | {"_AGE80>=75", "_BMI5>=18.5", "_BMI5>=25", "_BMI5>=30", "_BMI5>=35"}
            | {"_BMI5>=40"},
        )
        added = set(products) - set(base)
        self.assertIn("_AGE80*SEX", added)
        self.assertIn("_AGE80*_BMI5", added)
        self.assertTrue(any(n.startswith("_AGE80*_SMOKER3=") for n in added))
        self.assertFalse(any(n.startswith("_AGE80*_AGE80") for n in added))

    def test_nonlinear_columns_are_computed_from_training_statistics(self):
        state = fit_preprocessor(self.x, self.names, squares=True, interactions=True)
        matrix = transform_features(self.x[:5], self.names, state, dtype=np.float64)
        age = np.clip(self.column(self.x[:5], "_AGE80"), *state["bounds"]["_AGE80"])
        age = np.where(np.isnan(age), state["median"]["_AGE80"], age)
        centred = age - state["center"]["_AGE80"]
        sex = (self.column(self.x[:5], "SEX") == 1) * 1.0
        for name, raw in (("_AGE80^2", centred**2), ("_AGE80*SEX", centred * sex)):
            j = state["columns"].index(name)
            mean, std = state["mean"][j - 1], state["std"][j - 1]  # no intercept
            np.testing.assert_allclose(matrix[:, j], (raw - mean) / std)

    def test_drop_leaves_out_exactly_the_named_features(self):
        base = fit_preprocessor(self.x, self.names)["features"]
        state = fit_preprocessor(self.x, self.names, drop=("FLUSHOT6", "BPMEDS"))
        self.assertEqual(set(base) - set(state["features"]), {"FLUSHOT6", "BPMEDS"})
        with self.assertRaises(ValueError):
            fit_preprocessor(self.x, self.names, drop=("NOT_A_FEATURE",))

    def test_raw_baselines_keep_codes_as_numbers(self):
        x = self.x.copy()
        genhlth = self.column(x, "GENHLTH")
        genhlth[:10] = 9  # "refused": a missing code, which raw mode keeps as 9
        x[:, self.index["GENHLTH"]] = genhlth
        state = fit_preprocessor(x, self.names, raw="spec")
        self.assertEqual(state["features"], list(fs.SPEC))
        matrix = transform_features(x, self.names, state, dtype=np.float64)
        j = state["columns"].index("GENHLTH")
        filled = np.where(np.isnan(genhlth), np.nanmean(genhlth), genhlth)
        expected = (filled - filled.mean()) / filled.std()
        np.testing.assert_allclose(matrix[:, j], expected)
        every = fit_preprocessor(x, self.names, raw="all")["features"]
        self.assertFalse(set(every) & set(fs.LEAKY))
        with self.assertRaises(ValueError):
            fit_preprocessor(x, self.names, raw="decoded")

    def test_mostly_missing_features_are_dropped(self):
        x = self.x.copy()
        x[:1900, self.index["DIABAGE2"]] = np.nan  # 95 % missing > max_nan
        self.assertNotIn("DIABAGE2", fit_preprocessor(x, self.names)["features"])
        self.assertIn(
            "DIABAGE2", fit_preprocessor(x, self.names, max_nan=1.0)["features"]
        )


if __name__ == "__main__":
    unittest.main()
