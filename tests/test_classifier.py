"""Unit tests for Model A Resume Classifier."""

import pytest
import numpy as np
from src.features.text_features import build_tfidf_vectorizer
from src.models.classifier import (
    get_classifier_model,
    train_classifier,
    predict_category,
)


def test_build_tfidf_vectorizer():
    vectorizer = build_tfidf_vectorizer(max_features=500, min_df=1, sublinear_tf=True)
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.sublinear_tf is True
    assert vectorizer.max_features == 500


def test_get_classifier_model():
    lr = get_classifier_model("logistic_regression")
    assert lr.class_weight == "balanced"

    svm = get_classifier_model("svm")
    assert svm.class_weight == "balanced"

    baseline = get_classifier_model("baseline")
    assert baseline.strategy == "most_frequent"


def test_train_and_predict_classifier():
    X = [
        "Experienced Python developer with Django, Flask, SQL, and Git.",
        "Java software engineer proficient in Spring Boot, Hibernate, and Microservices.",
        "HR manager with recruiting, employee onboarding, talent management, and payroll.",
        "Data scientist with Python, machine learning, pandas, scikit-learn, and stats.",
    ]
    y = ["Python Developer", "Java Developer", "HR", "Data Science"]

    vec, clf = train_classifier(X, y, model_type="svm", max_features=100)
    assert vec is not None
    assert clf is not None

    test_resume = "Skilled Python and Django engineer with PostgreSQL and Git background."
    pred = predict_category(test_resume, clf, vectorizer=vec)
    assert pred in y
    assert pred == "Python Developer"
