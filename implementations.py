"""
Implementations of the six required methods for CS-433 Project 1
- Initial w is the initial weight vector.
- gamma is the step-size, and max iters is the number of steps to run.
- lambda is always the regularization parameter.
- For SGD, you must use the standard mini-batch-size 1 (sample just one datapoint).
- The mean squared error formula has a factor 0.5 to be consistent with the lecture notes.
- For least squares, you are allowed to use everything from numpy.linalg but not numpy.linalg.lstsq.
- All vectors should be implemented as 1D arrays
with shape (X,) instead of (X, 1).
"""

import numpy as np


def compute_mse(y, tx, w):
    """Compute the mean squared error."""
    e = y - tx @ w
    mse = 0.5 * np.mean(e**2)
    return mse


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    w = initial_w
    for n in range(max_iters):
        e = y - tx @ w
        gradient = -tx.T @ e / len(y)
        w = w - gamma * gradient
    loss = compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent"""
    w = initial_w
    for n in range(max_iters):
        n = np.random.randint(len(y))
        e = y[n] - tx[n] @ w
        gradient = -tx[n].T * e
        w = w - gamma * gradient
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations"""
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def _sigmoid(t):
    """Apply the sigmoid function on t, as exp(-log(1 + exp(-t))) so that large
    negative t does not overflow np.exp."""
    return np.exp(-np.logaddexp(0, -t))


def _logistic_loss(y, tx, w):
    """Compute the negative log likelihood loss.

    -[y log s(z) + (1 - y) log(1 - s(z))] equals log(1 + exp(z)) - y z, which stays
    finite where s(z) rounds to 0 or 1.
    """
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent"""
    w = initial_w
    for n in range(max_iters):
        gradient = (1 / y.shape[0]) * tx.T @ (_sigmoid(tx @ w) - y)
        w = w - gamma * gradient
    loss = _logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent"""
    w = initial_w
    for n in range(max_iters):
        gradient = (1 / y.shape[0]) * tx.T @ (_sigmoid(tx @ w) - y) + 2 * lambda_ * w
        w = w - gamma * gradient
    loss = _logistic_loss(y, tx, w)
    return w, loss
