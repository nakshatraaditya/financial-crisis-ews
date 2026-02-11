from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score
)

def budget_threshold_topk(probs: np.ndarray, budget: float) -> float:
    if len(probs) == 0:
        return 1.0
    return float(np.quantile(probs, 1.0 - budget))

def event_level_recall(
    df_period: pd.DataFrame,
    alerts: np.ndarray,
    crisis_col: str
) -> float:
    tmp = df_period[["country", "year", crisis_col]].copy()
    tmp["alert"] = alerts

    captured, total = 0, 0
    for _, sub in tmp.groupby("country"):
        sub = sub.sort_values("year")
        crisis_years = sub.loc[sub[crisis_col] == 1, "year"].astype(int).tolist()
        for cy in crisis_years:
            total += 1
            w = sub[(sub["year"] >= cy - 2) & (sub["year"] <= cy - 1)]
            if w["alert"].sum() > 0:
                captured += 1

    return np.nan if total == 0 else captured / total

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "alert_rate": float(y_pred.mean()),
    }
    return out

def compute_prob_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    brier = float(brier_score_loss(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    return {"pr_auc": pr_auc, "brier": brier}

