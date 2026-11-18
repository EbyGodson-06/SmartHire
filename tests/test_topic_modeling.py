"""Tests for unsupervised Job Description Topic Modeling (NMF).

Validates:
- Non-negative Matrix Factorization (NMF) fitting, components, and non-negativity guarantees.
- Hyperparameter search (topic_range evaluation of Reconstruction Error and Topic Diversity).
- Topic top terms and semantic profile generation.
- Model serialization and deserialization.
- Integration test on canonical job corpus and trained NMF topic model.
"""

import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    JOB_MATRIX_PATH,
    NMF_MODEL_PATH,
    NMF_N_TOPICS,
    RANDOM_STATE,
)
from src.models.topic_modeling import (
    fit_nmf,
    evaluate_topic_range,
    get_topic_top_terms,
    get_topic_profiles,
    save_topic_model,
    load_topic_model,
)


@pytest.fixture
def synthetic_sparse_data():
    """Create reproducible synthetic sparse TF-IDF text data for topic modeling unit tests."""
    docs = [
        "python machine learning data science pandas numpy scikit-learn",
        "python deep learning pytorch tensorflow computer vision nlp",
        "java spring boot microservices rest api hibernate sql",
        "java enterprise backend spring cloud docker kubernetes",
        "registered nurse clinical patient care healthcare hospital icu",
        "nurse practitioner medical clinical patient healthcare telemetry",
        "sales representative business development account executive b2b crm",
        "inside sales marketing revenue pipeline prospecting cold calling",
        "financial analyst accounting general ledger gaap excel audit",
        "senior accountant balance sheet financial reporting tax payroll",
    ] * 5  # 50 docs
    vectorizer = TfidfVectorizer(max_features=40)
    matrix = vectorizer.fit_transform(docs)
    df = pd.DataFrame({
        "title": [f"Job {i}" for i in range(len(docs))],
        "company": ["Corp A"] * len(docs),
        "location": ["Remote"] * len(docs),
        "skills": ["python, sql" if i % 2 == 0 else "java, docker" for i in range(len(docs))],
        "description": docs,
        "experience": ["2-5 yrs"] * len(docs),
    })
    return matrix, vectorizer, df


def test_fit_nmf_synthetic(synthetic_sparse_data):
    """Test NMF fitting directly on sparse CSR matrix."""
    matrix, _, _ = synthetic_sparse_data
    n_topics = 3
    model, W = fit_nmf(matrix, n_topics=n_topics, max_iter=50, random_state=42)

    assert model is not None
    assert model.n_components == n_topics
    assert model.components_.shape == (n_topics, matrix.shape[1])
    assert W.shape == (matrix.shape[0], n_topics)

    # Strictly non-negative property of NMF
    assert (model.components_ >= 0).all(), "H contains negative values"
    assert (W >= 0).all(), "W contains negative values"


def test_evaluate_topic_range_synthetic(synthetic_sparse_data):
    """Test evaluation across range of topic counts."""
    matrix, vectorizer, _ = synthetic_sparse_data
    topic_range = [2, 3]
    results = evaluate_topic_range(
        matrix,
        vectorizer=vectorizer,
        topic_range=topic_range,
        max_iter=100,
        random_state=42,
    )

    assert len(results) == 2
    for r in results:
        assert "n_topics" in r
        assert "reconstruction_error" in r
        assert "topic_diversity" in r
        assert r["reconstruction_error"] > 0.0
        assert 0.0 <= r["topic_diversity"] <= 1.0


def test_get_topic_top_terms(synthetic_sparse_data):
    """Test extraction of top terms and weights for each topic."""
    matrix, vectorizer, _ = synthetic_sparse_data
    model, _ = fit_nmf(matrix, n_topics=3, max_iter=100, random_state=42)
    top_terms = get_topic_top_terms(model, vectorizer, top_n=5)

    assert len(top_terms) == 3
    for topic_id in range(3):
        assert topic_id in top_terms
        assert len(top_terms[topic_id]) == 5
        for term, weight in top_terms[topic_id]:
            assert isinstance(term, str)
            assert isinstance(weight, float)
            assert weight >= 0.0


def test_get_topic_profiles(synthetic_sparse_data):
    """Test semantic topic profile generation."""
    matrix, vectorizer, df = synthetic_sparse_data
    model, W = fit_nmf(matrix, n_topics=3, max_iter=100, random_state=42)
    profiles = get_topic_profiles(
        model=model,
        job_corpus_df=df,
        vectorizer=vectorizer,
        W=W,
        top_n_terms=5,
        top_n_titles=2,
        top_n_skills=2,
        skill_sample_size=10,
    )

    assert len(profiles) == 3
    total_docs = 0
    for profile in profiles:
        assert "topic_id" in profile
        assert "document_count" in profile
        assert "percentage" in profile
        assert "naukri_share" in profile
        assert "top_terms" in profile
        assert "meaningful_skills" in profile
        assert "example_postings" in profile
        total_docs += profile["document_count"]

    assert total_docs == len(df)


def test_save_and_load_topic_model(synthetic_sparse_data, tmp_path):
    """Test serialization and deserialization of NMF model."""
    matrix, _, _ = synthetic_sparse_data
    model, W_orig = fit_nmf(matrix, n_topics=3, max_iter=100, random_state=42)
    saved_path = tmp_path / "test_nmf.pkl"

    save_topic_model(model, saved_path)
    assert saved_path.exists()

    loaded_model = load_topic_model(saved_path)
    assert loaded_model.n_components == model.n_components
    np.testing.assert_allclose(model.components_, loaded_model.components_, rtol=1e-5)

    # Transforms match within numerical tolerance
    W_loaded = loaded_model.transform(matrix)
    np.testing.assert_allclose(W_orig, W_loaded, atol=1e-4)


def test_canonical_job_matrix_and_topic_model_integration():
    """Verify integration with canonical job matrix and trained NMF model if available."""
    assert JOB_MATRIX_PATH.exists(), f"Canonical job matrix missing at {JOB_MATRIX_PATH}"
    matrix = sp.load_npz(JOB_MATRIX_PATH)
    assert matrix.shape[0] == 140225
    assert matrix.shape[1] == 10000
    # Non-negativity check
    assert matrix.data.min() >= 0.0, "TF-IDF matrix has negative values"

    if NMF_MODEL_PATH.exists():
        model = load_topic_model(NMF_MODEL_PATH)
        assert model.n_components == NMF_N_TOPICS
        assert model.components_.shape == (NMF_N_TOPICS, 10000)
        assert (model.components_ >= 0).all()
