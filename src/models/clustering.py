"""Clustering and Job Family Discovery module (Unsupervised Learning).

Implements:
- Scalable MiniBatchKMeans on high-dimensional sparse TF-IDF job vectors.
- Inertia (Elbow Method) and sampled Silhouette Score evaluation across K = 3 to 10.
- Semantic cluster interpretation (representative job titles, top terms, canonical skills).
- Sparse-compatible 2D dimensionality reduction using TruncatedSVD.
- Model serialization and loading utilities.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterable, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

from src.config import (
    RANDOM_STATE,
    KMEANS_MODEL_PATH,
    KMEANS_N_CLUSTERS,
    KMEANS_BATCH_SIZE,
    CLUSTERING_SAMPLE_SIZE,
)
from src.features.match_features import extract_job_skills


def fit_minibatch_kmeans(
    job_matrix: sp.csr_matrix,
    n_clusters: int = KMEANS_N_CLUSTERS,
    batch_size: int = KMEANS_BATCH_SIZE,
    random_state: int = RANDOM_STATE,
    n_init: int = 3,
    max_iter: int = 100,
) -> MiniBatchKMeans:
    """Fit a MiniBatchKMeans model on the sparse TF-IDF job matrix.

    MiniBatchKMeans operates directly on sparse CSR matrices without dense conversion,
    offering high scalability and fast convergence on 100k+ job postings.

    Args:
        job_matrix: Sparse CSR matrix of job documents (N, V).
        n_clusters: Number of target job family clusters.
        batch_size: Size of mini-batches for incremental updates.
        random_state: Reproducible seed.
        n_init: Number of initializations.
        max_iter: Maximum iterations over mini-batches.

    Returns:
        Fitted MiniBatchKMeans model.
    """
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
    )
    model.fit(job_matrix)
    return model


def evaluate_cluster_range(
    job_matrix: sp.csr_matrix,
    k_range: Iterable[int] = range(3, 11),
    batch_size: int = KMEANS_BATCH_SIZE,
    sample_size: int = CLUSTERING_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Evaluate clustering performance across a range of k values.

    Computes:
    - Inertia across the complete dataset (140,225 records).
    - Silhouette Score on a reproducible representative sample to manage O(N^2) complexity.

    Args:
        job_matrix: Sparse CSR matrix.
        k_range: Iterable of cluster counts to test (e.g. 3..10).
        batch_size: Mini-batch size.
        sample_size: Number of records sampled for silhouette calculation.
        random_state: Reproducible seed.

    Returns:
        Dict containing:
            - k_values: List[int]
            - inertias: List[float] (full dataset)
            - silhouette_scores: List[float] (sampled)
            - sample_size: int
    """
    k_list = list(k_range)
    inertias = []
    silhouettes = []

    # Draw reproducible random sample for silhouette evaluation
    n_samples = job_matrix.shape[0]
    actual_sample_size = min(sample_size, n_samples)
    rng = np.random.default_rng(random_state)
    sample_indices = rng.choice(n_samples, size=actual_sample_size, replace=False)
    sample_matrix = job_matrix[sample_indices]

    for k in k_list:
        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=batch_size,
            random_state=random_state,
            n_init=3,
        )
        model.fit(job_matrix)
        inertias.append(float(model.inertia_))

        # Predict cluster labels for the representative sample
        labels_sample = model.predict(sample_matrix)
        sil = float(silhouette_score(sample_matrix, labels_sample))
        silhouettes.append(sil)

    return {
        "k_values": k_list,
        "inertias": inertias,
        "silhouette_scores": silhouettes,
        "sample_size": actual_sample_size,
    }


def get_cluster_top_terms(
    model: MiniBatchKMeans,
    vectorizer,
    top_n: int = 15,
) -> Dict[int, List[str]]:
    """Extract top TF-IDF centroid keywords for each cluster."""
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = np.array(vectorizer.get_feature_names_out())

    cluster_terms = {}
    for cluster_idx in range(model.n_clusters):
        top_terms = terms[order_centroids[cluster_idx, :top_n]].tolist()
        cluster_terms[cluster_idx] = top_terms

    return cluster_terms


def get_cluster_profiles(
    model: MiniBatchKMeans,
    job_corpus_df: pd.DataFrame,
    vectorizer=None,
    top_n_titles: int = 5,
    top_n_skills: int = 5,
    top_n_terms: int = 10,
    skill_sample_size: int = 200,
    random_state: int = RANDOM_STATE,
) -> List[Dict[str, Any]]:
    """Build detailed semantic profiles and interpretations for all discovered clusters.

    Args:
        model: Fitted MiniBatchKMeans model.
        job_corpus_df: DataFrame of 140,225 job postings.
        vectorizer: Fitted TF-IDF vectorizer (optional, for centroid terms).
        top_n_titles: Number of top job titles to report.
        top_n_skills: Number of top canonical skills to report.
        top_n_terms: Number of centroid terms to report.
        skill_sample_size: Number of postings per cluster sampled for skill extraction.
        random_state: Random seed for sample selection.

    Returns:
        List of cluster profile dictionaries.
    """
    labels = model.labels_
    total_records = len(job_corpus_df)
    profiles = []

    # Centroid terms if vectorizer provided
    cluster_terms = get_cluster_top_terms(model, vectorizer, top_n=top_n_terms) if vectorizer else {}

    rng = np.random.default_rng(random_state)

    for cluster_id in range(model.n_clusters):
        cluster_mask = (labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        c_size = int(len(cluster_indices))
        c_pct = round((c_size / total_records) * 100.0, 2)

        sub_df = job_corpus_df.iloc[cluster_indices]

        # Top titles
        top_titles = sub_df["title"].value_counts().head(top_n_titles).index.tolist()

        # Representative canonical skills sampled across cluster postings
        sample_size = min(skill_sample_size, c_size)
        sample_idx = rng.choice(c_size, size=sample_size, replace=False)
        sample_postings = sub_df.iloc[sample_idx]

        skills_collected = []
        for _, row in sample_postings.iterrows():
            skills_collected.extend(extract_job_skills(row["skills"], row["description"]))

        top_skills = pd.Series(skills_collected).value_counts().head(top_n_skills).index.tolist()

        # Example postings
        example_rows = sub_df.head(3)[["title", "company", "location", "experience"]].to_dict(orient="records")

        profile = {
            "cluster_id": cluster_id,
            "size": c_size,
            "percentage": c_pct,
            "top_titles": top_titles,
            "top_terms": cluster_terms.get(cluster_id, []),
            "top_skills": top_skills,
            "example_postings": example_rows,
        }
        profiles.append(profile)

    return profiles


def compute_truncated_svd_projection(
    job_matrix: sp.csr_matrix,
    n_components: int = 2,
    sample_size: int = CLUSTERING_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, TruncatedSVD]:
    """Compute 2D TruncatedSVD projection directly on sparse TF-IDF feature space.

    TruncatedSVD serves as a sparse-compatible two-dimensional linear projection
    of high-dimensional TF-IDF vectors. Unlike standard PCA, it operates directly
    on sparse CSR matrices without mean-centering, avoiding dense matrix allocation
    and memory exhaustion on 140k+ records.

    Args:
        job_matrix: Sparse CSR matrix.
        n_components: Dimensionality (default 2).
        sample_size: Number of records sampled for visualization.
        random_state: Random seed.

    Returns:
        Tuple of (2D coordinates array, sample indices array, fitted TruncatedSVD model).
    """
    n_samples = job_matrix.shape[0]
    actual_sample_size = min(sample_size, n_samples)
    rng = np.random.default_rng(random_state)
    sample_indices = rng.choice(n_samples, size=actual_sample_size, replace=False)
    sample_matrix = job_matrix[sample_indices]

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    coords = svd.fit_transform(sample_matrix)

    return coords, sample_indices, svd


# Backward compatibility alias
compute_pca_projection = compute_truncated_svd_projection


def save_clustering_model(
    model: MiniBatchKMeans,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Serialize the trained clustering model to disk."""
    dest = Path(output_path or KMEANS_MODEL_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    return dest


def load_clustering_model(
    file_path: Optional[Union[str, Path]] = None,
) -> MiniBatchKMeans:
    """Load a serialized clustering model from disk."""
    path = Path(file_path or KMEANS_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Clustering model artifact not found at {path}")
    return joblib.load(path)
