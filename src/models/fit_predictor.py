"""Shortlisting / Fit Predictor module (Supervised Learning - Optional).

Model B: Predicts shortlisting probability (0/1 or continuous fit score)
from pairwise resume-job match features (skill overlap, text similarity, experience match).
"""

from pathlib import Path
from typing import Optional, Dict, Any
import joblib
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from src.config import FIT_PREDICTOR_PATH, RANDOM_STATE


def get_fit_predictor_model(model_type: str = "logistic_regression") -> BaseEstimator:
    """Instantiate a fit prediction model (e.g. Logistic Regression or XGBoost)."""
    if model_type == "logistic_regression":
        return LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
        except ImportError:
            raise ImportError("xgboost package is not installed.")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def save_fit_predictor(model: BaseEstimator, output_path: Optional[Path] = None) -> Path:
    """Serialize the shortlisting fit predictor artifact."""
    dest = output_path or FIT_PREDICTOR_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    return dest


def load_fit_predictor(file_path: Optional[Path] = None) -> BaseEstimator:
    """Load a serialized fit predictor artifact from disk."""
    path = file_path or FIT_PREDICTOR_PATH
    if not path.exists():
        raise FileNotFoundError(f"Fit predictor artifact not found at {path}")
    return joblib.load(path)


def predict_fit_score(feature_vector, model: BaseEstimator) -> float:
    """Predict shortlisting probability score between 0.0 and 1.0."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_vector)
        return float(probabilities[0, 1])
    return float(model.predict(feature_vector)[0])
