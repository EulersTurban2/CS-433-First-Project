import numpy as np

# Helper functions
def compute_mse(e):
    """Half mean squared error: (1/2N) * sum(e^2)."""
    return 1 / 2 * np.mean(e**2)

def mse_loss(y, tx, w):
    """MSE loss for linear regression (least squares / ridge)."""
    e = y - tx.dot(w)
    return compute_mse(e)

def compute_gradient(y, tx, w):
    """Computes the gradient at w."""
    e = y - np.dot(tx, w)
    gradient = -1 / y.shape[0] * np.dot(tx.T, e)
    return gradient

def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))

def logistic_loss(y, tx, w):
    """Binary cross-entropy / negative log-likelihood for logistic regression."""
    pred = sigmoid(tx.dot(w))
    return np.mean(-y * np.log(pred) - (1.0 - y) * np.log(1.0 - pred))

# Main 6 graded functions

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for linear regression (least squares / ridge)."""
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = mse_loss(y, tx, w)
    return w, loss
    
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm for linear regression (least squares / ridge)."""
    w = initial_w
    for _ in range(max_iters):
        i = np.random.choice(y.shape[0])
        w = w - gamma * compute_gradient(y[i:i+1], tx[i:i+1], w)
    loss = mse_loss(y, tx, w)
    return w, loss

def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = mse_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    # Loss must not include the penalty term!!
    loss = mse_loss(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = np.asarray(initial_w, dtype=float).copy()
    for _ in range(max_iters):
        grad = tx.T.dot(sigmoid(tx.dot(w)) - y) / y.shape[0]
        w = w - gamma * grad
    loss = logistic_loss(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # Loss must not include the penalty term!!
    w = np.asarray(initial_w, dtype=float).copy()
    for _ in range(max_iters):
        grad = tx.T.dot(sigmoid(tx.dot(w)) - y) / y.shape[0] + 2 * lambda_ * w
        w = w - gamma * grad
    loss = logistic_loss(y, tx, w)
    return w, loss
