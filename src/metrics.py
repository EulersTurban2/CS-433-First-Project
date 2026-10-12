import numpy as np

"""
    Metric functions for evaluating classification models.
"""

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true != 1) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred != 1))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2 * (precision_value * recall_value) / (precision_value + recall_value) if (precision_value + recall_value) > 0 else 0.0


def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions if total_predictions > 0 else 0.0


def auroc(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    desc_score_indices = np.argsort(y_scores)
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]

    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    ranks = np.arange(1, len(y_true_sorted) + 1)
    sum_ranks_pos = np.sum(ranks[y_true_sorted == 1])

    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) 
    return auc if auc > 0 else np.nan