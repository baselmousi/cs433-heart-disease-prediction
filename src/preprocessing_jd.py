"""Code for preprocessing the data for CS-433 Project 1"""
import sys
from pathlib import Path 
import numpy as np
from helpers import load_csv_data
import os
import timeit
from numba import njit

ROOT = Path.cwd().parent
sys.path.append(str(ROOT))
data_dir = f"{ROOT}/data"

num_variables_selected = ["_AGE80", "FC60_", "PHYSHLTH", "_DRNKWEK", "_BMI5", "PAVIG11_", "GRENDAY_", "POORHLTH", "ORNGDAY_"]

cat_variables_selected = ["SEX", "GENHLTH", "EMPLOY1", "BPHIGH4", "DIFFWALK", "CVDSTRK3", "_HCVU651", "USEEQUIP", "CHCCOPD1", "_LMTSCL1",
                            "QLACTLM2", "PNEUVAC3", "DIABETE3", "_LMTACT1", "_LMTWRK1", "_DRDXAR1", "_RFCHOL", "DIFFALON", "CHCKIDNY", 
                            "MARITAL", "VETERAN3", "DIFFDRES", "INCOME2", "_SMOKER3", "_CHLDCNT" ]

# These variables may need to be excluded given that they may be correlated with the consequences of the disease, 
# so not predictive of the disease itself. For example, DIFFWALK and CVDSTRK3 may be consequences of heart disease rather than predictors.
var_to_exclude_mb = ["DIFFWALK", "CVDSTRK3", "USEEQUIP", "CHCCOPD1", "_LMTSCL1", "QLACTLM2", "PNEUVAC3", "_LMTACT1", "_LMTWRK1", "_DRDXAR1", 
                    , "DIFFALON", "CHCKIDNY", "DIFFDRES"]

#Only necessary for numerical variables as categorical variables will simply have additional class and the fact that the missingness is probably not at random
name_missing_code_map = {"_AGE80": [None], "FC60_" : [99900], "PHYSHLTH":[77, 99], "_DRNKWEK":[99900], "_BMI5":[None], "PAVIG11_":[None],  "GRENDAY_":[None],
                        "POORHLTH":[77,99], "ORNGDAY_":[None]}
name_code_to_replace_map = { "PHYSHLTH": {88: 0}, "POORHLTH": {88:0}}


#--------------------------------
# Loading the data and getting the mapping from variable names to column indices
#--------------------------------

def load_data(data_dir):
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_dir)

    # getting the feature names since load_csv_data drops the header 
    with open(os.path.join(data_dir, "x_train.csv")) as train_data_reader: 
        header = train_data_reader.readline().strip().split(",")

    feature_names = np.array(header[1:])
    name_column_index_mapping = {name: j for j, name in enumerate(feature_names)}
    return x_train, x_test, y_train, name_column_index_mapping

#--------------------------------
# Extracting the selected features from the data and recoding the missing values and the values to be replaced
#--------------------------------

def sub_set_columns(X, name_col_map, num_names, cat_names):
    """
    Subsample the columns of X given the names of the numeric and categorical variables. Returns the subsampled X, a mapping of names to indices, a mapping of indices to names and the numeric and categorical indices. 
    """
    cat_name_idx_map = {}
    num_name_idx_map = {}

    num_idx0 = []
    cat_idx0 = []
    k=0
    for num_n in num_names:
        num_idx0.append(name_col_map[num_n])
        num_name_idx_map[num_n] = k
        k+=1
    for cat_n in cat_names:
        cat_idx0.append(name_col_map[cat_n])
        cat_name_idx_map[cat_n] = k
        k+=1
    
    num_idx=list(num_name_idx_map.values())
    cat_idx=list(cat_name_idx_map.values())

    name_idx_map = {**num_name_idx_map, **cat_name_idx_map}
    idx_name_map = {v: k for k, v in name_idx_map.items()}
    return X[:, num_idx0 + cat_idx0], name_idx_map, idx_name_map, num_idx, cat_idx, num_idx0, cat_idx0, cat_name_idx_map, num_name_idx_map

def replace_missing_codes(X, name_missing_code_map, name_col_map):
    for name,codes in name_missing_code_map.items():
        col_idx = name_col_map[name]
        for code in codes:
            if code is not None:
                X[X[:, col_idx] == code, col_idx] = np.nan
    return X

def replace_codes(X, name_code_to_replace_map, name_col_map):
    for name, code_map in name_code_to_replace_map.items():
        col_idx = name_col_map[name]
        for old_code, new_code in code_map.items():
            X[X[:, col_idx] == old_code, col_idx] = new_code
    return X

#--------------------------------
# Missing value imputation with MICE for the training and test data using only the training data to fit the imputation models
#--------------------------------

def fit_codes(Xraw, cat_idx):
    X = np.asarray(Xraw, dtype=float).copy()
    levels, K = {}, [0] * X.shape[1]
    for j in cat_idx:
        obs = ~np.isnan(X[:, j])
        lv = np.unique(X[obs, j])
        X[obs, j] = np.searchsorted(lv, X[obs, j])
        levels[j], K[j] = lv, len(lv)
    return X, levels, K

def transform_codes(Xraw, levels, K):
    X = np.asarray(Xraw, dtype=float).copy()

    for j, values in levels.items():
        observed = ~np.isnan(X[:, j])
        raw_values = X[observed, j]

        positions = np.searchsorted(values, raw_values)
        valid = (
            (positions < values.size)
            & (values[np.minimum(positions, values.size - 1)] == raw_values)
        )

        observed_rows = np.flatnonzero(observed)

        # Unseen test categories become missing.
        X[observed_rows[~valid], j] = np.nan
        X[observed_rows[valid], j] = positions[valid]

    return X

def from_codes(Xi, levels):
    out = Xi.copy()
    for j, lv in levels.items():
        out[:, j] = lv[Xi[:, j].astype(int)]
    return out

# 2. Imputation models (all observed rows are used)

def _impute_num(A_fit, y_fit, A_miss, rng, ridge=1e-4):
    n, p = A_fit.shape

    G = A_fit.T @ A_fit
    Xt_y = A_fit.T @ y_fit

    scale = ridge * np.trace(G) / p
    G.flat[::p + 1] += scale
    G[0, 0] -= scale

    L = np.linalg.cholesky(G)

    beta_hat = np.linalg.solve(
        L.T,
        np.linalg.solve(L, Xt_y),
    )

    residuals = y_fit - A_fit @ beta_hat
    sigma2 = (residuals @ residuals) / rng.chisquare(max(n - p, 1))

    beta = beta_hat + np.sqrt(sigma2) * np.linalg.solve(
        L.T,
        rng.standard_normal(p),
    )

    return (
        A_miss @ beta
        + np.sqrt(sigma2) * rng.standard_normal(A_miss.shape[0])
    )


def _impute_cat(A_fit, y_fit, A_miss, K, rng, lam=1.0):
    n_obs = A_fit.shape[0]

    bootstrap_indices = rng.integers(
        0,
        n_obs,
        size=n_obs,
    )
    counts = np.bincount(
        bootstrap_indices,
        minlength=n_obs,
    )

    selected = counts > 0
    As = A_fit[selected]
    labels = y_fit[selected].astype(np.int64)
    weights = counts[selected].astype(A_fit.dtype, copy=False)

    weighted_As = As * weights[:, None]
    XtX = As.T @ weighted_As

    XtY = np.zeros(
        (As.shape[0], K),
        dtype=A_fit.dtype,
    )
    XtY[np.arange(As.shape[0]), labels] = weights
    XtY = As.T @ XtY

    p = XtX.shape[0]
    XtX.flat[::p + 1] += lam
    XtX[0, 0] -= lam

    W = np.linalg.solve(XtX, XtY)

    P = np.clip(A_miss @ W, 1e-3, None)
    P /= P.sum(axis=1, keepdims=True)

    u = rng.random(A_miss.shape[0])
    draw = np.sum(P.cumsum(axis=1) < u[:, None], axis=1)

    return np.minimum(draw, K - 1)

@njit(cache=True)
def fill_numba(D, Xi, j, is_num, start, k, mu, sd):
    n = Xi.shape[0]

    if is_num:
        for i in range(n):
            D[i, start] = (Xi[i, j] - mu[j]) / sd[j]
    else:
        for i in range(n):
            value = int(Xi[i, j])

            for c in range(1, k):
                D[i, start + c - 1] = 1.0 if value == c else 0.0

# 3. One chain = one completed dataset

def _chain_train_test(
    X_train,
    X_test,
    types,
    K,
    n_iter,
    seed,
    max_fit_rows,
    n_train,
):
    print(f"Imputing missing values in the test and training set")
    t0 = timeit.default_timer()
    rng = np.random.default_rng(seed)

    X = np.vstack((X_train, X_test)).astype(float, copy=True)

    n_total, d = X.shape
    train_rows = np.arange(n_train)
    test_rows = np.arange(n_train, n_total)

    missing = np.isnan(X)

    order = [
        j
        for j in np.argsort(
            missing.sum(axis=0),
            kind="stable",
        )
        if missing[:, j].any()
    ]

    slices = []
    start = 1

    for variable_type, k in zip(types, K):
        width = 1 if variable_type == "num" else max(k - 1, 0)
        slices.append(slice(start, start + width))
        start += width

    keep = [
        np.concatenate(
            [[0]]
            + [
                np.arange(s.start, s.stop)
                for i, s in enumerate(slices)
                if i != j
            ]
        )
        for j in range(d)
    ]

    # These statistics must be learned from training rows only.
    mu = np.array([
        np.nanmean(X_train[:, j])
        if types[j] == "num"
        else 0.0
        for j in range(d)
    ])

    sd = np.array([
        np.nanstd(X_train[:, j]) + 1e-12
        if types[j] == "num"
        else 1.0
        for j in range(d)
    ])

    is_num = np.array(
        [variable_type == "num" for variable_type in types],
        dtype=np.bool_,
    )

    slice_start = np.array(
        [s.start for s in slices],
        dtype=np.int64,
    )

    K_array = np.asarray(K, dtype=np.int64)

    # Initialize every missing value from training observed values only.
    for j in order:
        train_observed = train_rows[~missing[:n_train, j]]

        if train_observed.size == 0:
            raise ValueError(
                f"Column {j} has no observed training values."
            )

        train_missing = np.flatnonzero(missing[:, j])

        X[train_missing, j] = rng.choice(
            X[train_observed, j],
            size=train_missing.size,
        )

    D = np.ones((n_total, start), dtype=float)

    for j in range(d):
        fill_numba(
            D,
            X,
            j,
            is_num[j],
            slice_start[j],
            K_array[j],
            mu,
            sd,
        )

    for _ in range(n_iter):
        for j in order:
            train_observed = train_rows[~missing[:n_train, j]]

            if train_observed.size > max_fit_rows:
                fit_rows = rng.choice(
                    train_observed,
                    size=max_fit_rows,
                    replace=False,
                )
            else:
                fit_rows = train_observed

            # Only training rows enter the fitted model.
            A_fit = D[np.ix_(fit_rows, keep[j])]
            y_fit = X[fit_rows, j]

            train_missing = np.flatnonzero(
                missing[:n_train, j]
            )

            test_missing = n_train + np.flatnonzero(
                missing[n_train:, j]
            )

            prediction_rows = np.concatenate(
                (train_missing, test_missing)
            )

            if prediction_rows.size == 0:
                continue

            A_missing = D[
                np.ix_(prediction_rows, keep[j])
            ]

            if types[j] == "num":
                draws = _impute_num(
                    A_fit,
                    y_fit,
                    A_missing,
                    rng,
                )
            else:
                draws = _impute_cat(
                    A_fit,
                    y_fit,
                    A_missing,
                    K[j],
                    rng,
                )

            X[prediction_rows, j] = draws

            fill_numba(
                D,
                X,
                j,
                is_num[j],
                slice_start[j],
                K_array[j],
                mu,
                sd,
            )
    print(f"Done in {timeit.default_timer() - t0:.1f} seconds")
    return X[:n_train], X[n_train:]

def mice_train_test(
    X_train,
    X_test,
    num_idx,
    cat_idx,
    m=3,
    n_iter=5,
    seed=67,
    max_fit_rows=75000,
):
    X_train_codes, levels, K = fit_codes(
        X_train,
        cat_idx,
    )

    X_test_codes = transform_codes(
        X_test,
        levels,
        K,
    )

    d = X_train_codes.shape[1]

    types = [
        "num" if j in num_idx else "cat"
        for j in range(d)
    ]

    n_train = X_train_codes.shape[0]
    master = np.random.default_rng(seed)

    completed_train = []
    completed_test = []

    for _ in range(m):
        train_i, test_i = _chain_train_test(
            X_train_codes,
            X_test_codes,
            types,
            K,
            n_iter,
            int(master.integers(2**32)),
            max_fit_rows,
            n_train,
        )

        completed_train.append(
            from_codes(train_i, levels)
        )
        completed_test.append(
            from_codes(test_i, levels)
        )

    return completed_train, completed_test

#--------------------------------
# One-hot encoding of the completed datasets
#--------------------------------

def one_hot_encode_X(Xi, i, num_idx, cat_dict, num_dict, name_col_map_X):
    X1 = [np.ones_like(Xi[:,0]), Xi[:, num_idx]]
    if i==0:
        n = X1[1].shape[1] + 1
        num_dict = {n:j+1 for n,j in num_dict.items()}

    for name, j in cat_dict.items():
        values, inverse = np.unique(
            Xi[ :, j],
            return_inverse=True,
        )
        n_values = len(values)
        one_hot_columns = np.eye(n_values)[inverse]
    
        X1.append(one_hot_columns)        
        if i == 0:
            for value in values:
                feature_name = f"{name}_{value}"
                name_column_mapping_X[feature_name] = n
                n += 1
    if i == 0:
        name_column_mapping_X = {**num_dict, **name_column_mapping_X}
    return np.hstack(X1)

#--------------------------------
# Main function to preprocess the data
#--------------------------------

def preprocessing(data_dir, num_variables_selected, cat_variables_selected, name_missing_code_map, name_code_to_replace_map):
    print("Loading data...")
    start = timeit.default_timer()
    x_train, x_test, y_train, name_col_map = load_data(data_dir)
    t0 = timeit.default_timer()
    print(f"Data loaded in {t0 - start:.2f} seconds.")
    print("Encoding missing values and replacing codes...")
    x_train = replace_missing_codes(x_train, name_missing_code_map, name_col_map)
    x_train = replace_codes(x_train, name_code_to_replace_map, name_col_map)
    x_test = replace_missing_codes(x_test, name_missing_code_map, name_col_map)
    x_test = replace_codes(x_test, name_code_to_replace_map, name_col_map)
    t1 = timeit.default_timer()
    print(f"Missing values encoded and codes replaced in {t1 - t0:.2f} seconds.")
    print("Extracting selected columns...")
    X_tr_raw, name_idx_map, idx_name_map, num_idx, cat_idx, num_idx0, cat_idx0, cat_name_idx_map, num_name_idx_map = sub_set_columns(x_train, name_col_map, num_variables_selected, cat_variables_selected)
    X_te_raw = x_test[:, num_idx0 + cat_idx0]
    t2 = timeit.default_timer()
    print(f"Selected columns extracted in {t2 - t1:.2f} seconds.")
    print("Performing MICE imputation...")
    completed, completed_test = mice_train_test(X_tr_raw, X_te_raw, num_idx, cat_idx, m=2, n_iter=5, seed=67, max_fit_rows=75000)
    t3 = timeit.default_timer()
    print(f"MICE imputation completed in {t3 - t2:.2f} seconds.")
    print("One-hot encoding completed datasets...")
    name_column_mapping_X = {"Intercept":0}
    X1_tr = [one_hot_encode_X(Xi, i, num_idx, cat_name_idx_map, num_name_idx_map, name_column_mapping_X) for i, Xi in enumerate(completed)]
    X1_te = [one_hot_encode_X(Xi, i+1, num_idx, cat_name_idx_map, num_name_idx_map, name_column_mapping_X) for i, Xi in enumerate(completed_test)]
    t4 = timeit.default_timer()
    print(f"One-hot encoding completed in {t4 - t3:.2f} seconds.")
    print(f"Total preprocessing time: {t4 - start:.2f} seconds.")
    return X1_tr, X1_te, y_train, name_column_mapping_X
