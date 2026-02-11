import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, recall_score


def pr_auc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def brier(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(brier_score_loss(y_true, y_prob))


def pick_threshold_by_alert_budget(y_prob, budget: float) -> float:
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.size == 0:
        return float("nan")
    k = int(np.ceil(budget * y_prob.size))
    k = max(1, min(k, y_prob.size))
    return float(np.partition(y_prob, -k)[-k])


def recall_at_threshold(y_true, y_prob, threshold: float) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.sum() == 0:
        return float("nan")
    y_pred = (y_prob >= threshold).astype(int)
    return float(recall_score(y_true, y_pred))


def onset_recall_at_threshold(y_true, y_prob, threshold: float) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.sum() == 0:
        return float("nan")

    onset = np.zeros_like(y_true)
    onset[0] = y_true[0]
    onset[1:] = (y_true[1:] == 1) & (y_true[:-1] == 0)

    if onset.sum() == 0:
        return float("nan")

    y_pred = (y_prob >= threshold).astype(int)
    return float(recall_score(onset.astype(int), y_pred))

