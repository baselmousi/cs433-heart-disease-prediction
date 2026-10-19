"""Code for preprocessing the data for CS-433 Project 1"""

# Five-feature baseline preprocessing is fitted on training rows only.
# Mappings follow the 2015 BRFSS codebook and the checks documented in
# notebooks/02_data_exploration_step_by_step.ipynb. In the course CSV, BMI is
# already expressed in kg/m²; it must not be divided by 100 again.

import csv
from pathlib import Path

import numpy as np


BASELINE_FEATURES = ("GENHLTH", "PHYSHLTH", "_SMOKER3", "_BMI5", "_AGE80")
NUMERIC_NAMES = ("PHYSHLTH_days", "BMI", "AGE80")
CATEGORICAL_NAMES = (
    "GENHLTH_very_good",
    "GENHLTH_good",
    "GENHLTH_fair",
    "GENHLTH_poor",
    "GENHLTH_no_rating",
    "SMOKER_daily",
    "SMOKER_some_days",
    "SMOKER_former",
    "SMOKER_unknown",
)


def load_baseline_training_data(data_dir):
    """Load the five selected columns and verify feature/label IDs row by row."""
    data_dir = Path(data_dir)
    with (data_dir / "y_train.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        if next(reader) != ["Id", "_MICHD"]:
            raise ValueError("Unexpected label header.")
        labels = []
        for row in reader:
            if len(row) != 2:
                raise ValueError("Malformed label row.")
            labels.append((int(row[0]), int(row[1])))
    records = np.asarray(labels, dtype=np.int64)
    if records.ndim != 2 or not len(records):
        raise ValueError("No label records.")
    ids, y = records[:, 0].copy(), records[:, 1].copy()
    if len(np.unique(ids)) != len(ids) or not np.array_equal(np.unique(y), [-1, 1]):
        raise ValueError("Expected unique IDs and labels -1/+1.")
    raw = np.empty((len(y), len(BASELINE_FEATURES)), dtype=float)
    with (data_dir / "x_train.csv").open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[0] != "Id" or len(set(header)) != len(header):
            raise ValueError("Invalid feature header.")
        columns = [header.index(name) for name in BASELINE_FEATURES]
        count = 0
        for i, row in enumerate(reader):
            if len(row) != len(header) or i >= len(y):
                raise ValueError(
                    "Feature row count/width disagrees with labels/header."
                )
            if int(row[0]) != ids[i]:
                raise ValueError(f"Feature and label IDs differ at data row {i + 1}.")
            raw[i] = [float(row[j]) if row[j].strip() else np.nan for j in columns]
            count += 1
        if count != len(y):
            raise ValueError("Feature and label row counts differ.")
    return raw, y, ids


def decode_baseline_features(raw):
    """Return categorical indicators and decoded numeric columns, preserving raw."""
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(BASELINE_FEATURES):
        raise ValueError("Expected columns in BASELINE_FEATURES order.")
    health, physical, smoking, bmi, age = raw.T
    if not np.all(np.isnan(health) | np.isin(health, [1, 2, 3, 4, 5, 7, 9])):
        raise ValueError("Unexpected GENHLTH code.")
    health_indicators = np.column_stack(
        [
            health == 2,
            health == 3,
            health == 4,
            health == 5,
            np.isin(health, [7, 9]) | np.isnan(health),
        ]
    )
    if not np.all(
        np.isnan(physical) | np.isin(physical, np.r_[np.arange(1, 31), [77, 88, 99]])
    ):
        raise ValueError("Unexpected PHYSHLTH code.")
    days = np.where(physical == 88, 0, physical)
    days = np.where(np.isin(physical, [77, 99]), np.nan, days)
    if not np.all(np.isnan(smoking) | np.isin(smoking, [1, 2, 3, 4, 9])):
        raise ValueError("Unexpected _SMOKER3 code.")
    smoking_indicators = np.column_stack(
        [
            smoking == 1,
            smoking == 2,
            smoking == 3,
            (smoking == 9) | np.isnan(smoking),
        ]
    )
    if not np.all(np.isnan(bmi) | ((bmi >= 12) & (bmi < 100))):
        raise ValueError("Unexpected BMI scale/range; expected CSV values in kg/m².")
    if not np.all(np.isnan(age) | ((age >= 18) & (age <= 80) & (age == np.floor(age)))):
        raise ValueError("Unexpected _AGE80 value.")
    categories = np.column_stack([health_indicators, smoking_indicators]).astype(float)
    return categories, np.column_stack([days, bmi, age])


def fit_baseline_preprocessor(raw_training):
    """Learn median replacements and numeric scaling from training rows only."""
    _, numeric = decode_baseline_features(raw_training)
    if not len(numeric) or np.any(np.isnan(numeric).all(axis=0)):
        raise ValueError("Each numeric feature needs observed training values.")
    missing = np.isnan(numeric)
    medians = np.nanmedian(numeric, axis=0)
    filled = np.where(missing, medians, numeric)
    scales = filled.std(axis=0)
    return {
        "medians": medians,
        "means": filled.mean(axis=0),
        "scales": np.where(scales == 0, 1.0, scales),
        "keep_missing": missing.any(axis=0) & ~missing.all(axis=0),
    }


def transform_baseline_features(raw, state):
    """Apply fixed training statistics and add an intercept; never refit here."""
    categories, numeric = decode_baseline_features(raw)
    missing = np.isnan(numeric)
    filled = np.where(missing, state["medians"], numeric)
    scaled = (filled - state["means"]) / state["scales"]
    matrix = np.column_stack(
        [
            np.ones(len(raw)),
            categories,
            scaled,
            missing[:, state["keep_missing"]].astype(float),
        ]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Prepared features contain nonfinite entries.")
    return matrix


def baseline_column_names(state):
    """Return model columns in exactly the order produced by transform."""
    return (
        ["intercept", *CATEGORICAL_NAMES]
        + [n + "_scaled" for n in NUMERIC_NAMES]
        + [
            name + "_missing"
            for name, keep in zip(NUMERIC_NAMES, state["keep_missing"])
            if keep
        ]
    )
