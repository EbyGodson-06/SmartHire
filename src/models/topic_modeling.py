"""Job Description Topic Modeling module (Unsupervised Learning).

Implements:
- Non-negative Matrix Factorization (NMF) on high-dimensional sparse TF-IDF job vectors.
- Systematic evaluation of topic counts (reconstruction error, topic diversity).
- Semantic topic profiling (top terms with weights, representative titles, canonical skills, actual postings).
- Document-topic distribution mapping and model serialization/loading.
- Technical documentation of LDA representations and memory/computational barriers.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterable, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.decomposition import NMF

from src.config import (
    RANDOM_STATE,
    NMF_MODEL_PATH,
    NMF_N_TOPICS,
    NMF_MAX_ITER,
)
from src.features.match_features import extract_job_skills


# Cross-cluster generic noise terms to filter for domain-meaningful skill reporting
GENERIC_CROSS_CLUSTER_TERMS = {"Go", "Human Resources", "Recruitment", "Payroll"}


def fit_nmf(
    job_matrix: sp.csr_matrix,
    n_topics: int = NMF_N_TOPICS,
    max_iter: int = NMF_MAX_ITER,
    random_state: int = RANDOM_STATE,
    init: str = "nndsvda",
) -> Tuple[NMF, np.ndarray]:
    """Fit Non-negative Matrix Factorization (NMF) on a sparse TF-IDF matrix.

    Factorizes X (N, V) into W (N, K) and H (K, V) such that X ~ W * H,
    where W is the document-topic distribution and H is the topic-term distribution.
    Operates natively on non-negative sparse CSR matrices using Coordinate Descent.

    Args:
        job_matrix: Sparse CSR matrix of job TF-IDF vectors (N, V).
        n_topics: Number of latent topics to extract (K).
        max_iter: Maximum coordinate descent iterations.
        random_state: Reproducible seed.
        init: Initialization strategy ('nndsvda' handles sparsity well).

    Returns:
        Tuple of (fitted NMF model, document-topic matrix W).
    """
    model = NMF(
        n_components=n_topics,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
        solver="cd",
    )
    W = model.fit_transform(job_matrix)
    return model, W


def evaluate_topic_range(
    job_matrix: sp.csr_matrix,
    vectorizer,
    topic_range: Iterable[int] = (5, 6, 8, 10, 12),
    max_iter: int = 100,
    random_state: int = RANDOM_STATE,
) -> List[Dict[str, Any]]:
    """Evaluate NMF topic modeling across a range of topic counts.

    Computes:
    - Reconstruction Error (Frobenius norm ||X - W*H||_F).
    - Topic Diversity: The ratio of unique words among the top-15 terms across all topics.
      Higher diversity indicates non-redundant, distinct topic definitions.

    Args:
        job_matrix: Sparse CSR matrix (N, V).
        vectorizer: Fitted TF-IDF vectorizer (for vocabulary terms).
        topic_range: Iterable of topic counts to test.
        max_iter: Max iterations per fit.
        random_state: Random seed.

    Returns:
        List of dicts containing evaluation metrics for each tested K.
    """
    terms = np.array(vectorizer.get_feature_names_out())
    results = []

    for k in topic_range:
        model = NMF(
            n_components=k,
            init="nndsvda",
            max_iter=max_iter,
            random_state=random_state,
            solver="cd",
        )
        model.fit(job_matrix)
        recon_err = float(model.reconstruction_err_)
        H = model.components_

        # Calculate topic diversity among top-15 terms
        all_top_terms = []
        for topic_idx in range(k):
            top_idx = H[topic_idx].argsort()[::-1][:15]
            all_top_terms.extend(terms[top_idx].tolist())
        diversity = len(set(all_top_terms)) / len(all_top_terms)

        results.append({
            "n_topics": k,
            "reconstruction_error": recon_err,
            "topic_diversity": diversity,
        })

    return results


def get_topic_top_terms(
    model: NMF,
    vectorizer,
    top_n: int = 15,
) -> Dict[int, List[Tuple[str, float]]]:
    """Extract top terms and their positive weights for each discovered topic.

    Args:
        model: Fitted NMF model.
        vectorizer: Fitted TF-IDF vectorizer.
        top_n: Number of terms per topic.

    Returns:
        Dict mapping topic_id -> list of (term, weight) tuples sorted descending.
    """
    terms = np.array(vectorizer.get_feature_names_out())
    H = model.components_
    topic_terms = {}

    for topic_id in range(model.n_components):
        weights = H[topic_id]
        top_indices = weights.argsort()[::-1][:top_n]
        topic_terms[topic_id] = [
            (terms[idx], float(weights[idx]))
            for idx in top_indices
        ]

    return topic_terms


def get_topic_profiles(
    model: NMF,
    job_corpus_df: pd.DataFrame,
    vectorizer,
    W: Optional[np.ndarray] = None,
    job_matrix: Optional[sp.csr_matrix] = None,
    top_n_terms: int = 15,
    top_n_titles: int = 5,
    top_n_skills: int = 5,
    skill_sample_size: int = 100,
    random_state: int = RANDOM_STATE,
) -> List[Dict[str, Any]]:
    """Generate comprehensive semantic profiles and interpretations for all topics.

    Args:
        model: Fitted NMF model.
        job_corpus_df: Canonical DataFrame of 140,225 job postings.
        vectorizer: Fitted TF-IDF vectorizer.
        W: Document-topic matrix (optional; computed if job_matrix provided).
        job_matrix: Sparse CSR matrix (used to compute W if W is None).
        top_n_terms: Number of top terms to report.
        top_n_titles: Number of representative job titles.
        top_n_skills: Number of meaningful canonical skills.
        skill_sample_size: Number of postings sampled for skill extraction.
        random_state: Random seed.

    Returns:
        List of topic profile dictionaries.
    """
    if W is None:
        if job_matrix is None:
            raise ValueError("Either W or job_matrix must be provided.")
        W = model.transform(job_matrix)

    total_records = len(job_corpus_df)
    dominant_topic = W.argmax(axis=1)
    topic_terms_dict = get_topic_top_terms(model, vectorizer, top_n=top_n_terms)
    rng = np.random.default_rng(random_state)
    profiles = []

    for topic_id in range(model.n_components):
        doc_indices = np.where(dominant_topic == topic_id)[0]
        n_docs = int(len(doc_indices))
        pct = round((n_docs / total_records) * 100.0, 2)

        sub_df = job_corpus_df.iloc[doc_indices]
        top_titles = sub_df["title"].value_counts().head(top_n_titles).index.tolist()

        # Measure Naukri share to track scraping boilerplate influence
        # Naukri records are the first 21,914 rows of the corpus
        n_naukri = int((doc_indices < 21914).sum())
        naukri_share = round((n_naukri / n_docs) * 100.0, 1) if n_docs > 0 else 0.0

        # Sample postings for skill extraction
        sample_size = min(skill_sample_size, n_docs)
        if sample_size > 0:
            sample_idx = rng.choice(n_docs, size=sample_size, replace=False)
            sample_postings = sub_df.iloc[sample_idx]
            skills_collected = []
            for _, row in sample_postings.iterrows():
                skills_collected.extend(extract_job_skills(row["skills"], row["description"]))
            raw_top_skills = pd.Series(skills_collected).value_counts()
            meaningful_skills = [
                s for s in raw_top_skills.index
                if s not in GENERIC_CROSS_CLUSTER_TERMS
            ][:top_n_skills]
        else:
            meaningful_skills = []

        # Representative actual postings: select highest-loading exemplars on this topic
        if topic_id == 0:
            # Topic 0 is a mixed commercial/finance domain: balance top commercial/sales with genuine finance/accounting
            sorted_idx = doc_indices[np.argsort(-W[doc_indices, topic_id])]
            finance_mask = (
                job_corpus_df.iloc[sorted_idx]["title"].str.contains(
                    "controller|accountant|financial reporting", case=False, na=False
                )
                & ~job_corpus_df.iloc[sorted_idx]["title"].str.contains(
                    "sales|business development", case=False, na=False
                )
            ).values
            top_commercial = sorted_idx[~finance_mask][:2]
            top_finance = sorted_idx[finance_mask][:1]
            top_exemplar_indices = np.concatenate([top_commercial, top_finance])
        else:
            top_exemplar_indices = doc_indices[np.argsort(-W[doc_indices, topic_id])[:3]]
        example_rows = []
        for idx in top_exemplar_indices:
            row = job_corpus_df.iloc[idx]
            excerpt = (row["description"] or "")[:220].replace("\n", " ").strip()
            example_rows.append({
                "title": row["title"],
                "company": row["company"],
                "location": row["location"],
                "experience": row["experience"],
                "weight": round(float(W[idx, topic_id]), 4),
                "description_excerpt": excerpt,
            })

        profile = {
            "topic_id": topic_id,
            "document_count": n_docs,
            "percentage": pct,
            "naukri_share": naukri_share,
            "top_terms": [t[0] for t in topic_terms_dict[topic_id]],
            "top_term_weights": topic_terms_dict[topic_id],
            "representative_titles": top_titles,
            "meaningful_skills": meaningful_skills,
            "example_postings": example_rows,
        }
        profiles.append(profile)

    return profiles


def save_topic_model(
    model: NMF,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Serialize the trained NMF topic model to disk."""
    dest = Path(output_path or NMF_MODEL_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest)
    return dest


def load_topic_model(
    file_path: Optional[Union[str, Path]] = None,
) -> NMF:
    """Load a serialized NMF topic model from disk."""
    path = Path(file_path or NMF_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Topic model artifact not found at {path}")
    return joblib.load(path)
