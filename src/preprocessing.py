"""Code for preprocessing the data for CS-433 Project 1"""

# Two pipelines, both fitted on training rows only:
#
# 1. Five-feature baseline (E0/E1). Mappings follow the 2015 BRFSS codebook and the
#    checks documented in notebooks/02_data_exploration_step_by_step.ipynb. In the
#    course CSV, BMI is already expressed in kg/m²; it must not be divided by 100 again.
# 2. Full pipeline: the 77 features of src/feature_spec.py, with the decisions and
#    evidence in notebooks/02_feature_selection.ipynb.

import csv
import itertools
from pathlib import Path

import numpy as np

from src import feature_spec as fs

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


# --- Full pipeline: the 77 features of src/feature_spec.py --------------------


def read_feature_names(path):
    """Return the feature names of a course CSV header, without the Id column."""
    with open(path, newline="") as handle:
        header = next(csv.reader(handle))
    if header[0] != "Id":
        raise ValueError("Expected Id as the first column.")
    return header[1:]


def load_csv(path, chunk_rows=20_000):
    """Load a course CSV without its header as float64; empty fields become NaN.

    Gives the same values as np.genfromtxt, about 5 times faster (4 s for x_train.csv).
    """
    with open(path) as handle:
        handle.readline()
        n_rows = sum(1 for _ in handle)
    with open(path) as handle:
        n_columns = len(handle.readline().split(","))
        out = np.empty((n_rows, n_columns))
        start = 0
        while lines := list(itertools.islice(handle, chunk_rows)):
            # np.loadtxt cannot read empty fields, so spell them out as nan.
            # Two passes, since ",,," only gets every other gap in one.
            text = "\n".join(line.rstrip("\r\n") for line in lines) + "\n"
            text = text.replace(",,", ",nan,").replace(",,", ",nan,")
            text = text.replace(",\n", ",nan\n")
            out[start : start + len(lines)] = np.loadtxt(
                text.splitlines(), delimiter=",", ndmin=2
            )
            start += len(lines)
    return out


def clean_columns(x, feature_names, features):
    """Return cleaned float64 copies of the requested features, as {name: column}.

    Missing codes become NaN, remaps and SCALE are applied, then FILLS replace NaN
    where the skip pattern implies the answer. Features outside SPEC (the leaky ones)
    are copied unchanged. x itself is never modified.
    """
    if x.shape[1] != len(feature_names):
        raise ValueError("x and feature_names have different numbers of columns.")
    index = {name: j for j, name in enumerate(feature_names)}
    cache = {}

    def get(name):
        if name not in cache:
            if name not in index:
                raise ValueError(f"Feature {name} is not in feature_names.")
            _, codes, remap = fs.SPEC.get(name, ("nominal", [], {}))
            column = x[:, index[name]].astype(np.float64)  # astype copies
            column[np.isin(column, codes)] = np.nan
            for old, new in remap.items():
                column[column == old] = new
            cache[name] = column * fs.SCALE.get(name, 1.0)
        return cache[name]

    columns = {name: get(name) for name in features}
    for name, value, condition in fs.FILLS:
        if name in columns:
            column = columns[name]
            column[np.isnan(column) & condition(get)] = value
    return columns


# E8 non-linear features. Each bin adds a 0/1 step "value >= edge" next to the linear
# term, so its weight is the jump in log-odds at that edge.
BIN_EDGES = {
    "_AGE80": [25, 35, 45, 55, 65, 75],  # ten-year age groups
    "_BMI5": [18.5, 25, 30, 35, 40],  # WHO underweight ... obesity class III
}
AGE_INTERACTIONS = ("SEX", "_BMI5", "_SMOKER3", "DIABETE3", "BPHIGH4", "TOLDHI2")


def fit_preprocessor(
    x,
    feature_names,
    max_nan=0.9,
    indicators=False,
    ordinal_onehot=False,
    leaky=False,
    squares=False,
    bins=False,
    interactions=False,
    drop=(),
    raw=None,
    clip=0.995,
    min_count=30,
):
    """Learn every preprocessing statistic from the training rows x only.

    Defaults are the choices made in notebooks/02_feature_selection.ipynb.
    - max_nan: drop SPEC features with a larger NaN fraction after cleaning.
    - indicators: add a 0/1 missing indicator to numeric features with >= 1 % NaN.
    - ordinal_onehot: one-hot encode ordinal features instead of keeping them numeric.
    - leaky: add the LEAKY features (experiment E9), one-hot encoded.
    - squares: add (value - training mean)² for every continuous feature (E8).
    - bins: add the BIN_EDGES steps for age and BMI (E8).
    - interactions: add age × each column of the AGE_INTERACTIONS features (E8).
      Age and continuous partners are centred first, which keeps the products
      from being nearly copies of the main terms.
    - drop: SPEC features to leave out (experiment E5).
    - raw: skip all cleaning, for the baselines; see _fit_raw. "spec" keeps the SPEC
      features (E2), "all" every column except LEAKY (E0). Other options are ignored.
    - clip: clip continuous features to their [1 - clip, clip] training quantiles.
    - min_count: nominal levels with fewer training rows share the reference row.

    Nominal features are one-hot encoded with NaN as a level of its own. The most
    frequent level is the reference and gets no column: with a column for every
    level, they would sum to the intercept column and X.T @ X would be singular.
    Columns that are exact combinations of earlier ones are dropped as well and
    listed in state["dependent"].
    """
    if raw:
        return _fit_raw(x, feature_names, raw)
    unknown = set(drop) - set(fs.SPEC)
    if unknown:
        raise ValueError(f"Cannot drop features outside SPEC: {sorted(unknown)}")
    candidates = [name for name in fs.SPEC if name not in drop]
    candidates += list(fs.LEAKY) if leaky else []
    columns = clean_columns(x, feature_names, candidates)
    state = {
        "features": [],
        "kind": {},
        "bounds": {},
        "median": {},
        "levels": {},
        "reference": {},
        "indicator": [],
        "center": {},
        "squares": squares,
        "bins": BIN_EDGES if bins else {},
        "interactions": interactions,
    }
    for name in candidates:
        column = columns[name]
        nan = np.isnan(column)
        if nan.all() or (name in fs.SPEC and nan.mean() > max_nan):
            continue
        kind = fs.SPEC[name][0] if name in fs.SPEC else "nominal"
        if ordinal_onehot and kind == "ordinal":
            kind = "nominal"
        state["features"].append(name)
        state["kind"][name] = kind
        if kind == "nominal":
            values, counts = np.unique(column[~nan], return_counts=True)
            values = np.append(values, np.nan)
            counts = np.append(counts, nan.sum())
            reference = np.argmax(counts)
            keep = (counts >= min_count) & (np.arange(len(values)) != reference)
            state["reference"][name] = float(values[reference])
            state["levels"][name] = [float(v) for v in values[keep]]
            continue
        if kind == "binary":
            column = np.where(nan, np.nan, column == 1)
        if kind == "continuous":
            bounds = np.nanquantile(column, [1 - clip, clip])
            state["bounds"][name] = [float(b) for b in bounds]
            column = np.clip(column, *bounds)
        state["median"][name] = float(np.nanmedian(column))
        if kind == "continuous":
            filled = np.where(nan, state["median"][name], column)
            state["center"][name] = float(filled.mean())
        if indicators and nan.mean() >= 0.01:
            state["indicator"].append(name)
    matrix, names = _encode(columns, state)
    mean, std = matrix.mean(axis=0), matrix.std(axis=0)
    varying = np.flatnonzero(std > 0)  # a constant column carries no information
    keep = varying[_independent(matrix[:, varying], mean[varying], std[varying])]
    state["keep"], state["mean"], state["std"] = keep, mean[keep], std[keep]
    state["columns"] = ["intercept"] + [names[j] for j in keep]
    state["dependent"] = [names[j] for j in np.setdiff1d(varying, keep)]
    return state


def _raw_matrix(x, feature_names, features):
    """The requested columns of x as float64, in the order of features."""
    index = {name: j for j, name in enumerate(feature_names)}
    return x[:, [index[name] for name in features]].astype(np.float64, copy=False)


def _fit_raw(x, feature_names, raw):
    """Baseline preprocessing: survey codes used as plain numbers (77, 88, 99 included),
    NaN replaced by the training mean, then the usual standardization."""
    if raw not in ("spec", "all"):
        raise ValueError(f'raw must be "spec" or "all", not {raw!r}.')
    features = list(fs.SPEC)
    if raw == "all":
        features = [name for name in feature_names if name not in fs.LEAKY]
    matrix = _raw_matrix(x, feature_names, features)
    observed = ~np.isnan(matrix).all(axis=0)
    features, matrix = [n for n, o in zip(features, observed) if o], matrix[:, observed]
    fill = np.nanmean(matrix, axis=0)
    matrix = np.where(np.isnan(matrix), fill, matrix)
    mean, std = matrix.mean(axis=0), matrix.std(axis=0)
    varying = np.flatnonzero(std > 0)
    keep = varying[_independent(matrix[:, varying], mean[varying], std[varying])]
    return {
        "raw": raw,
        "features": features,
        "fill": fill,
        "keep": keep,
        "mean": mean[keep],
        "std": std[keep],
        "columns": ["intercept"] + [features[j] for j in keep],
        "dependent": [features[j] for j in np.setdiff1d(varying, keep)],
    }


def _independent(matrix, mean, std, tol=1e-9, chunk_rows=50_000):
    """Positions of the columns to keep so that none is a combination of the others.

    Skip patterns create exact dependencies: LASTSMK2 is answered exactly by former
    smokers (_SMOKER3 = 3), and the three _LMT* features share their "no arthritis"
    rows. A column is dropped when the earlier kept columns (and the intercept)
    explain all but tol of its variance. It loses no information, since the model
    can still represent it, but it makes X.T @ X singular for least squares.
    """
    corr = np.zeros((matrix.shape[1], matrix.shape[1]))
    for start in range(0, len(matrix), chunk_rows):
        block = (matrix[start : start + chunk_rows] - mean) / std
        corr += block.T @ block
    corr /= len(matrix)
    keep = []
    for j in range(len(corr)):
        residual = corr[j, j]
        if keep:
            fit = np.linalg.solve(corr[np.ix_(keep, keep)], corr[keep, j])
            residual -= corr[j, keep] @ fit
        if residual > tol:
            keep.append(j)
    return np.array(keep, dtype=int)


def _encode(columns, state):
    """Turn cleaned columns into an unscaled matrix, and name each matrix column."""
    blocks, names = [], []
    for name in state["features"]:
        column, kind = columns[name], state["kind"][name]
        nan = np.isnan(column)
        if kind == "nominal":
            for level in state["levels"][name]:
                blocks.append(nan if np.isnan(level) else column == level)
                names.append(f"{name}={level:g}")
            continue
        if kind == "binary":
            column = (column == 1) * 1.0
        if kind == "continuous":
            column = np.clip(column, *state["bounds"][name])
        filled = np.where(nan, state["median"][name], column)
        blocks.append(filled)
        names.append(name)
        if name in state["indicator"]:
            blocks.append(nan)
            names.append(f"{name}_missing")
        if kind != "continuous":
            continue
        if state["squares"]:
            blocks.append((filled - state["center"][name]) ** 2)
            names.append(f"{name}^2")
        for edge in state["bins"].get(name, []):
            blocks.append(filled >= edge)
            names.append(f"{name}>={edge:g}")
    if state["interactions"] and "_AGE80" in state["features"]:
        age = blocks[names.index("_AGE80")] - state["center"]["_AGE80"]
        for j, column_name in enumerate(list(names)):
            # "SEX", "_BMI5" or a one-hot level such as "_SMOKER3=1"; not "_BMI5^2".
            if column_name.split("=")[0] in AGE_INTERACTIONS:
                partner = blocks[j] - state["center"].get(column_name, 0.0)
                blocks.append(age * partner)
                names.append(f"_AGE80*{column_name}")
    return np.column_stack(blocks).astype(np.float64), names


def transform_features(x, feature_names, state, dtype=np.float32):
    """Apply fitted statistics, standardize and add an intercept; never refit here.

    The columns are named in state["columns"].
    """
    if state.get("raw"):
        matrix = _raw_matrix(x, feature_names, state["features"])
        matrix = np.where(np.isnan(matrix), state["fill"], matrix)
    else:
        columns = clean_columns(x, feature_names, state["features"])
        matrix, _ = _encode(columns, state)
    matrix = (matrix[:, state["keep"]] - state["mean"]) / state["std"]
    matrix = np.column_stack([np.ones(len(matrix)), matrix]).astype(dtype)
    if not np.isfinite(matrix).all():
        raise ValueError("Prepared features contain nonfinite entries.")
    return matrix
