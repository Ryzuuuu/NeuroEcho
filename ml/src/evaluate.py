"""Standalone evaluation utilities (metrics that matter for screening)."""
from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             f1_score, confusion_matrix)


def report(y_true, y_prob, threshold=0.5):
    """Return a dict of evaluation metrics for binary classification."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, y_prob), 3),
        "sensitivity_recall": round(recall_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred), 3),
        "f1": round(f1_score(y_true, y_pred), 3),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
