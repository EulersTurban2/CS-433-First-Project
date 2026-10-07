import numpy as np

# Helper functions
def compute_mse(e):
    return 1 / 2 * np.mean(e**2)

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return compute_mse(e)

# Main 6 graded functions

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    pass

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    pass

def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, w)
    return w, mse

def ridge_regression(y, tx, lambda_):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    pass

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    pass
