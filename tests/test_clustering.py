"""Tests for unsupervised job clustering module (MiniBatchKMeans, TruncatedSVD).

Validates:
- MiniBatchKMeans fitting, predict, and centroid extraction on sparse matrices.
- Hyperparameter search (k_range evaluation of Inertia and Silhouette score).
- Top terms and semantic cluster profile generation.
- Sparse 2D TruncatedSVD projection dimensionality reduction.
- Model serialization and deserialization.
- Integration test on canonical job corpus and trained K-Means model.
"""

import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    JOB_MATRIX_PATH,
    KMEANS_MODEL_PATH,
    KMEANS_N_CLUSTERS,
    RANDOM_STATE,
)
from src.models.clustering import (
    fit_minibatch_kmeans,
    evaluate_cluster_range,
    get_cluster_top_terms,
    get_cluster_profiles,
    compute_truncated_svd_projection,
    compute_pca_projection,
    save_clustering_model,
    load_clustering_model,
)


@pytest.fixture
def synthetic_sparse_data():
    """Create reproducible synthetic sparse TF-IDF text data for unit tests."""
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


def test_minibatch_kmeans_fit_and_predict(synthetic_sparse_data):
    """Test MiniBatchKMeans fitting directly on sparse CSR matrix."""
    matrix, _, _ = synthetic_sparse_data
    n_clusters = 3
    model = fit_minibatch_kmeans(matrix, n_clusters=n_clusters, batch_size=16, random_state=42)

    assert model is not None
    assert model.n_clusters == n_clusters
    assert model.cluster_centers_.shape == (n_clusters, matrix.shape[1])
    assert len(model.labels_) == matrix.shape[0]

    # Predict on the same matrix
    preds = model.predict(matrix)
    assert len(preds) == matrix.shape[0]
    assert set(preds).issubset(set(range(n_clusters)))


def test_evaluate_cluster_range_synthetic(synthetic_sparse_data):
    """Test evaluation across range of K with inertia and silhouette."""
    matrix, _, _ = synthetic_sparse_data
    k_range = [2, 3]
    eval_results = evaluate_cluster_range(
        matrix,
        k_range=k_range,
        batch_size=16,
        sample_size=30,
        random_state=42,
    )

    assert "k_values" in eval_results
    assert "inertias" in eval_results
    assert "silhouette_scores" in eval_results
    assert "sample_size" in eval_results
    assert eval_results["k_values"] == [2, 3]
    assert len(eval_results["inertias"]) == 2
    assert len(eval_results["silhouette_scores"]) == 2

    # Inertias should be positive floats
    for inertia in eval_results["inertias"]:
        assert isinstance(inertia, float)
        assert inertia > 0.0

    # Silhouette scores should be in [-1.0, 1.0]
    for sil in eval_results["silhouette_scores"]:
        assert isinstance(sil, float)
        assert -1.0 <= sil <= 1.0


def test_get_cluster_top_terms(synthetic_sparse_data):
    """Test centroid top terms extraction."""
    matrix, vectorizer, _ = synthetic_sparse_data
    model = fit_minibatch_kmeans(matrix, n_clusters=3, batch_size=16, random_state=42)
    top_terms = get_cluster_top_terms(model, vectorizer, top_n=5)

    assert len(top_terms) == 3
    for cluster_id in range(3):
        assert cluster_id in top_terms
        assert len(top_terms[cluster_id]) == 5
        assert all(isinstance(t, str) for t in top_terms[cluster_id])


def test_get_cluster_profiles(synthetic_sparse_data):
    """Test semantic cluster profile generation."""
    matrix, vectorizer, df = synthetic_sparse_data
    model = fit_minibatch_kmeans(matrix, n_clusters=3, batch_size=16, random_state=42)
    profiles = get_cluster_profiles(
        model=model,
        job_corpus_df=df,
        vectorizer=vectorizer,
        top_n_titles=2,
        top_n_skills=2,
        top_n_terms=3,
        skill_sample_size=10,
    )

    assert len(profiles) == 3
    total_size = 0
    for profile in profiles:
        assert "cluster_id" in profile
        assert "size" in profile
        assert "percentage" in profile
        assert "top_titles" in profile
        assert "top_terms" in profile
        assert "top_skills" in profile
        assert "example_postings" in profile
        assert profile["size"] > 0
        total_size += profile["size"]

    assert total_size == len(df)


def test_compute_truncated_svd_projection(synthetic_sparse_data):
    """Test 2D TruncatedSVD projection without converting sparse matrix to dense."""
    matrix, _, _ = synthetic_sparse_data
    sample_size = 25
    coords, sample_idx, svd = compute_truncated_svd_projection(
        job_matrix=matrix,
        n_components=2,
        sample_size=sample_size,
        random_state=42,
    )

    assert coords.shape == (sample_size, 2)
    assert len(sample_idx) == sample_size
    assert svd.n_components == 2
    assert not np.isnan(coords).any()

    # Verify backward compatibility alias
    coords_alias, idx_alias, _ = compute_pca_projection(
        job_matrix=matrix,
        n_components=2,
        sample_size=sample_size,
        random_state=42,
    )
    np.testing.assert_array_equal(coords, coords_alias)
    np.testing.assert_array_equal(sample_idx, idx_alias)


def test_save_and_load_clustering_model(synthetic_sparse_data, tmp_path):
    """Test serialization and loading of MiniBatchKMeans."""
    matrix, _, _ = synthetic_sparse_data
    model = fit_minibatch_kmeans(matrix, n_clusters=3, batch_size=16, random_state=42)
    saved_path = tmp_path / "test_kmeans.pkl"

    save_clustering_model(model, saved_path)
    assert saved_path.exists()

    loaded_model = load_clustering_model(saved_path)
    assert loaded_model.n_clusters == model.n_clusters
    np.testing.assert_array_equal(model.cluster_centers_, loaded_model.cluster_centers_)

    # Predictions match
    preds_orig = model.predict(matrix)
    preds_loaded = loaded_model.predict(matrix)
    np.testing.assert_array_equal(preds_orig, preds_loaded)


def test_canonical_job_matrix_and_model_integration():
    """Verify integration with canonical job matrix and trained K-Means model if available."""
    assert JOB_MATRIX_PATH.exists(), f"Canonical job matrix missing at {JOB_MATRIX_PATH}"
    matrix = sp.load_npz(JOB_MATRIX_PATH)
    assert matrix.shape[0] == 140225
    assert matrix.shape[1] == 10000

    if KMEANS_MODEL_PATH.exists():
        model = load_clustering_model(KMEANS_MODEL_PATH)
        assert model.n_clusters == KMEANS_N_CLUSTERS
        assert model.cluster_centers_.shape == (KMEANS_N_CLUSTERS, 10000)

        # Labels sanity check
        labels = model.labels_
        assert len(labels) == 140225
        assert not np.isnan(labels).any()
        assert labels.min() == 0
        assert labels.max() == KMEANS_N_CLUSTERS - 1

        # Check all clusters are non-empty
        unique, counts = np.unique(labels, return_counts=True)
        assert len(unique) == KMEANS_N_CLUSTERS
        assert sum(counts) == 140225
        for count in counts:
            assert count > 0
