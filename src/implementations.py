# Required file for our 6 graded functions implementations
import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    pass

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    pass

def least_squares(y, tx):
    pass

def ridge_regression(y, tx, lambda_):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    pass

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    pass
