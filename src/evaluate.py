"""Central evaluation metrics module for SmartHire.

Provides standardized evaluation routines for:
- Classification (Resume Classifier & Fit Predictor):
  Accuracy, Precision, Recall, Macro/Weighted F1, ROC-AUC, Confusion Matrix.
- Repeated Stratified Cross-Validation evaluation with mean and std stats.
- Clustering:
  Inertia (Elbow Method), Silhouette Score.
- Recommendation:
  Precision@K, Mean Reciprocal Rank (MRR).
- Regression (Optional Salary Model):
  MAE, RMSE, R-squared.
"""

from typing import Dict, Any, List, Set, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_validate, BaseCrossValidator


def evaluate_classifier(
    y_true,
    y_pred,
    y_prob=None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive classification performance metrics for a single prediction set."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }

    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            else:
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def evaluate_cv_pipeline(
    pipeline,
    X,
    y,
    cv: BaseCrossValidator,
    n_jobs: int = -1,
) -> Dict[str, float]:
    """Evaluate a pipeline across cross-validation splits and compute mean and std dev metrics.

    Scoring metrics:
    - accuracy
    - precision_macro
    - recall_macro
    - f1_macro
    - f1_weighted
    """
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }

    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)

    stats = {
        "accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "precision_macro_mean": float(np.mean(cv_results["test_precision_macro"])),
        "precision_macro_std": float(np.std(cv_results["test_precision_macro"])),
        "recall_macro_mean": float(np.mean(cv_results["test_recall_macro"])),
        "recall_macro_std": float(np.std(cv_results["test_recall_macro"])),
        "f1_macro_mean": float(np.mean(cv_results["test_f1_macro"])),
        "f1_macro_std": float(np.std(cv_results["test_f1_macro"])),
        "f1_weighted_mean": float(np.mean(cv_results["test_f1_weighted"])),
        "f1_weighted_std": float(np.std(cv_results["test_f1_weighted"])),
    }
    return stats


def evaluate_clustering(matrix, cluster_labels) -> Dict[str, float]:
    """Calculate clustering quality metrics."""
    return {
        "silhouette_score": float(silhouette_score(matrix, cluster_labels)),
    }


def compute_precision_at_k(
    recommended_job_ids: List[Any],
    relevant_job_ids: Set[Any],
    k: int = 5,
) -> float:
    """Compute Precision@K for a list of recommended job IDs."""
    top_k = recommended_job_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for job_id in top_k if job_id in relevant_job_ids)
    return hits / k


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    """Calculate regression metrics for optional salary prediction model."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }
