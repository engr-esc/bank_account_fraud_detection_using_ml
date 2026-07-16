"""Evaluation helper functions for the bank account fraud detection project."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_for_review_rate(y_score, review_rate: float = 0.10) -> float:
    """Return the score threshold that flags approximately the top review_rate."""
    if not 0 < review_rate < 1:
        raise ValueError("review_rate must be between 0 and 1.")
    return float(np.quantile(np.asarray(y_score), 1 - review_rate))


def evaluate_binary_classifier(y_true, y_score, threshold: float) -> Dict[str, float]:
    """Evaluate a binary classifier using probability scores and a chosen threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_positive_rate": float(y_pred.mean()),
        "true_positive_rate": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else np.nan,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else np.nan,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def evaluate_top_k(y_true, y_score, review_rate: float = 0.10) -> Dict[str, float]:
    """Evaluate recall and precision among the top ranked applications."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    n_review = max(1, int(np.ceil(len(y_true) * review_rate)))

    ranked_indices = np.argsort(-y_score)[:n_review]
    selected_actual = y_true[ranked_indices]

    fraud_found = int(selected_actual.sum())
    total_fraud = int(y_true.sum())

    return {
        "review_rate": float(review_rate),
        "review_count": int(n_review),
        "fraud_found_at_k": fraud_found,
        "recall_at_k": float(fraud_found / total_fraud) if total_fraud else np.nan,
        "precision_at_k": float(fraud_found / n_review),
    }


def evaluate_business_kpis(
    y_true,
    y_score,
    review_rate: float,
    loss_per_fraud: float,
    review_cost: float,
    false_positive_cost: float,
) -> Dict[str, float]:
    """Estimate business value using transparent illustrative assumptions."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    n_review = max(1, int(np.ceil(len(y_true) * review_rate)))
    selected = np.argsort(-y_score)[:n_review]

    reviewed_actual = y_true[selected]
    fraud_detected = int(reviewed_actual.sum())
    false_positives = int(n_review - fraud_detected)

    avoided_loss = fraud_detected * loss_per_fraud
    total_review_cost = n_review * review_cost
    total_false_positive_cost = false_positives * false_positive_cost
    estimated_net_benefit = avoided_loss - total_review_cost - total_false_positive_cost

    return {
        "fraud_detected": fraud_detected,
        "false_positives_in_review_queue": false_positives,
        "avoided_loss": float(avoided_loss),
        "review_cost_total": float(total_review_cost),
        "false_positive_cost_total": float(total_false_positive_cost),
        "estimated_net_benefit": float(estimated_net_benefit),
    }


def summarize_fairness_gaps(group_metrics: pd.DataFrame, attribute_name: str) -> Dict[str, float]:
    """Summarize core group fairness gaps from a group metrics table."""
    def metric_range(series):
        valid = pd.Series(series).dropna()
        return np.nan if valid.empty else float(valid.max() - valid.min())

    def min_max_ratio(series):
        valid = pd.Series(series).dropna()
        if valid.empty or valid.max() == 0:
            return np.nan
        return float(valid.min() / valid.max())

    tpr_gap = metric_range(group_metrics["true_positive_rate"])
    fpr_gap = metric_range(group_metrics["false_positive_rate"])

    return {
        "attribute": attribute_name,
        "demographic_parity_difference": metric_range(group_metrics["selection_rate"]),
        "disparate_impact_ratio": min_max_ratio(group_metrics["selection_rate"]),
        "equal_opportunity_difference": tpr_gap,
        "false_positive_rate_difference": fpr_gap,
        "equalized_odds_difference": float(np.nanmax([tpr_gap, fpr_gap])),
    }
