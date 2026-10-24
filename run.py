"""Fit the submitted model and write results/predictions.csv.

Logistic regression on the basic cleaning from the exploratory notebook:
drop administrative columns, map the special codes, drop columns that are
more than 90% empty or constant, median-fill, and standardize.
Several of the six models tie on validation F1. Logistic regression is the
one with the highest AUROC in that tie. The cutoff is the mean of the five
validation cutoffs.
"""

from pathlib import Path

import numpy as np

from implementations import logistic_regression
from src.folds import stratified_folds
from src.helpers import create_csv_submission, load_csv_data, load_feature_names

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw_dataset"
SEED = 42
N_FOLDS = 5
LOG_ITERS = 300
GAMMA_LOG = 0.1

ADMIN = [
    "IDATE", "IMONTH", "IDAY", "IYEAR", "FMONTH",
    "SEQNO", "_PSU", "DISPCODE", "_STSTR",
    "_STRWT", "_RAWRAKE", "_WT2RAKE", "_LLCPWT",
]
SPECIAL = {
    "GENHLTH": (7, 9), "BPHIGH4": (7, 9), "INCOME2": (77, 99),
    "DIABETE3": (7, 9), "SMOKE100": (7, 9), "TOLDHI2": (7, 9),
    "HAVARTH3": (7, 9), "DIFFWALK": (7, 9), "QLACTLM2": (7, 9),
    "PHYSHLTH": (77, 99), "MENTHLTH": (77, 99),
}
ZERO_DAY = {"PHYSHLTH": 88, "MENTHLTH": 88}


def map_codes(X, names):
    X = np.array(X, dtype=float, copy=True)
    index = {name: j for j, name in enumerate(names)}
    for name, code in ZERO_DAY.items():
        j = index.get(name)
        if j is not None:
            X[X[:, j] == code, j] = 0.0
    for name, codes in SPECIAL.items():
        j = index.get(name)
        if j is not None:
            X[np.isin(X[:, j], codes), j] = np.nan
    return X


def _keep(X_fit, X_apply, names, mask):
    mask = np.asarray(mask, dtype=bool)
    return X_fit[:, mask], X_apply[:, mask], [n for n, k in zip(names, mask) if k]


def clean(X_fit, X_apply, names, impute="median", map_special=True, drop_above=0.90):
    names = [str(n) for n in names]
    X_fit = np.array(X_fit, dtype=float, copy=True)
    X_apply = np.array(X_apply, dtype=float, copy=True)

    X_fit, X_apply, names = _keep(X_fit, X_apply, names, [n not in ADMIN for n in names])
    if map_special:
        X_fit = map_codes(X_fit, names)
        X_apply = map_codes(X_apply, names)

    if drop_above is not None:
        missing_rate = np.isnan(X_fit).mean(axis=0)
        X_fit, X_apply, names = _keep(X_fit, X_apply, names, missing_rate <= drop_above)

    varying = []
    for j in range(X_fit.shape[1]):
        observed = X_fit[np.isfinite(X_fit[:, j]), j]
        varying.append(len(np.unique(observed)) > 1)
    X_fit, X_apply, names = _keep(X_fit, X_apply, names, varying)

    fill = np.nanmedian(X_fit, axis=0) if impute == "median" else np.nanmean(X_fit, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    for matrix in (X_fit, X_apply):
        holes = np.where(np.isnan(matrix))
        matrix[holes] = fill[holes[1]]

    mean = X_fit.mean(axis=0)
    std = X_fit.std(axis=0)
    std[std == 0] = 1.0
    return (X_fit - mean) / std, (X_apply - mean) / std, names


def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


def probabilities(X, w):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))


def choose_threshold(y, scores):
    y_pos = np.asarray(y) == 1
    grid = np.linspace(0.0, 1.0, 101)
    positive = scores[:, None] >= grid[None, :]
    tp = np.sum(positive & y_pos[:, None], axis=0).astype(float)
    fp = np.sum(positive & ~y_pos[:, None], axis=0).astype(float)
    fn = np.sum(~positive & y_pos[:, None], axis=0).astype(float)
    prec = np.divide(tp, tp + fp, out=np.zeros(len(grid)), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros(len(grid)), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(grid)), where=(prec + rec) > 0)
    return float(grid[int(np.argmax(f1))])


def fit_logistic(X, y):
    y01 = (y == 1).astype(float)
    w0 = np.zeros(X.shape[1])
    return logistic_regression(y01, X, w0, LOG_ITERS, GAMMA_LOG)[0]


def validation_cutoff(X_train, y_train, feature_names):
    thresholds = []
    for tr, va in stratified_folds(y_train, N_FOLDS, SEED):
        Xtr, Xva, _ = clean(X_train[tr], X_train[va], feature_names)
        Xtr, Xva = add_bias(Xtr), add_bias(Xva)
        w = fit_logistic(Xtr, y_train[tr])
        thresholds.append(choose_threshold(y_train[va], probabilities(Xva, w)))
    return float(np.mean(thresholds))


def main():
    X_train, X_test, y_train, _, test_ids = load_csv_data(str(DATA_DIR))
    feature_names = load_feature_names(str(DATA_DIR))
    y_train = np.asarray(y_train)

    cutoff = validation_cutoff(X_train, y_train, feature_names)
    Xtr, Xte, _ = clean(X_train, X_test, feature_names)
    Xtr, Xte = add_bias(Xtr), add_bias(Xte)
    w = fit_logistic(Xtr, y_train)
    pred = np.where(probabilities(Xte, w) >= cutoff, 1, -1).astype(int)

    out = ROOT / "results" / "predictions.csv"
    out.parent.mkdir(exist_ok=True)
    create_csv_submission(test_ids, pred.tolist(), str(out))
    n_pos = int(np.sum(pred == 1))
    print("logistic regression, basic cleaning")
    print(f"cutoff {cutoff:.3f}, predicted positive rate {n_pos / len(pred):.2%}, columns {Xtr.shape[1]}")
    print("Wrote", out)


if __name__ == "__main__":
    main()
