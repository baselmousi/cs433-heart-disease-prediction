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

Shared arguments: y is the (N,) target vector (0/1 labels for the logistic methods),
tx the (N, D) feature matrix including the intercept column, and w a (D,) weight
vector. Every method returns (w, loss): the last weights and the loss at those weights,
without the penalty term for the regularized methods.
"""

import numpy as np


def compute_mse(y, tx, w):
    """Compute the mean squared error with the lecture's factor 1/2.

    Returns:
        float: 1/(2N) * sum of (y - tx w)^2.
    """
    e = y - tx @ w
    mse = 0.5 * np.mean(e**2)
    return mse


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full-batch gradient descent on the MSE.

    Args:
        y, tx, initial_w: see the module docstring.
        max_iters (int): number of gradient steps.
        gamma (float): step size.

    Returns:
        tuple: (w, loss), the last weights and their MSE.
    """
    w = initial_w
    for _ in range(max_iters):
        e = y - tx @ w
        gradient = -tx.T @ e / len(y)  # gradient of the MSE
        w = w - gamma * gradient
    loss = compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent on the MSE.

    Each step uses one row drawn uniformly at random (mini-batch size 1), so the
    result depends on NumPy's global random state.

    Args:
        y, tx, initial_w: see the module docstring.
        max_iters (int): number of single-row steps.
        gamma (float): step size.

    Returns:
        tuple: (w, loss), the last weights and their MSE on all rows.
    """
    w = initial_w
    for _ in range(max_iters):
        i = np.random.randint(len(y))
        e = y[i] - tx[i] @ w
        gradient = -tx[i] * e  # gradient of the squared error of row i
        w = w - gamma * gradient
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression by solving the normal equations tx^T tx w = tx^T y.

    Returns:
        tuple: (w, loss), the exact minimizer and its MSE.
    """
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression by solving (tx^T tx + 2 N lambda_ I) w = tx^T y.

    This minimizes MSE(w) + lambda_ * ||w||^2 (the intercept is penalized too).

    Args:
        lambda_ (float): regularization strength.

    Returns:
        tuple: (w, loss), the minimizer and its MSE without the penalty.
    """
    penalty = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T @ tx + penalty
    b = tx.T @ y
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
    """Logistic regression using full-batch gradient descent.

    Args:
        y (np.ndarray): (N,) labels in {0, 1}.
        tx, initial_w: see the module docstring.
        max_iters (int): number of gradient steps.
        gamma (float): step size.

    Returns:
        tuple: (w, loss), the last weights and their mean negative log likelihood.
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = (1 / y.shape[0]) * tx.T @ (_sigmoid(tx @ w) - y)
        w = w - gamma * gradient
    loss = _logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression using full-batch gradient descent.

    Minimizes the mean negative log likelihood + lambda_ * ||w||^2.

    Args:
        y (np.ndarray): (N,) labels in {0, 1}.
        lambda_ (float): regularization strength.
        tx, initial_w: see the module docstring.
        max_iters (int): number of gradient steps.
        gamma (float): step size.

    Returns:
        tuple: (w, loss), the last weights and their mean negative log likelihood
        without the penalty.
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = (1 / y.shape[0]) * tx.T @ (_sigmoid(tx @ w) - y) + 2 * lambda_ * w
        w = w - gamma * gradient
    loss = _logistic_loss(y, tx, w)
    return w, loss
