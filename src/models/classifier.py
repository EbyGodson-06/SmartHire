"""Resume Category Classifier module (Supervised Learning).

Model A: Maps resume text to job domains (25 classes)
using classical ML classifiers (LinearSVC, Logistic Regression).
"""

from pathlib import Path
from typing import Optional, Union, Tuple
import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

from src.config import CLASSIFIER_PATH, RANDOM_STATE
from src.features.text_features import build_tfidf_vectorizer, save_vectorizer


def get_classifier_model(
    model_type: str = "svm",
    class_weight: Optional[str] = "balanced",
    random_state: int = RANDOM_STATE,
) -> BaseEstimator:
    """Instantiate a classical classification model with class balancing."""
    if model_type == "svm":
        return LinearSVC(class_weight=class_weight, random_state=random_state)
    elif model_type == "logistic_regression":
        return LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=random_state)
    elif model_type == "baseline":
        return DummyClassifier(strategy="most_frequent")
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_classifier(
    X,
    y,
    model_type: str = "svm",
    max_features: int = 5000,
) -> Tuple[BaseEstimator, BaseEstimator]:
    """Train a standalone TF-IDF vectorizer and classifier on the full dataset.

    Returns:
        Tuple of (fitted_vectorizer, fitted_classifier).
    """
    vectorizer = build_tfidf_vectorizer(max_features=max_features)
    X_vec = vectorizer.fit_transform(X)

    model = get_classifier_model(model_type=model_type)
    model.fit(X_vec, y)

    return vectorizer, model


def save_classifier(model: BaseEstimator, output_path: Optional[Path] = None) -> Path:
    """Serialize the trained classifier artifact."""
    dest = output_path or CLASSIFIER_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    return dest


def load_classifier(file_path: Optional[Path] = None) -> BaseEstimator:
    """Load a serialized classifier artifact from disk."""
    path = file_path or CLASSIFIER_PATH
    if not path.exists():
        raise FileNotFoundError(f"Classifier artifact not found at {path}")
    return joblib.load(path)


def predict_category(
    resume_input: Union[str, np.ndarray],
    model: BaseEstimator,
    vectorizer: Optional[BaseEstimator] = None,
) -> str:
    """Predict job category for a resume string or pre-computed feature vector.

    Args:
        resume_input: Raw text string or vectorized array.
        model: Trained classifier estimator.
        vectorizer: Optional fitted TfidfVectorizer (required if resume_input is str).

    Returns:
        Predicted category string.
    """
    if isinstance(resume_input, str):
        if vectorizer is None:
            raise ValueError("Vectorizer must be provided when resume_input is a text string.")
        vec = vectorizer.transform([resume_input])
    else:
        vec = resume_input

    prediction = model.predict(vec)
    return str(prediction[0])
