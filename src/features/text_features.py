"""Text feature extraction module for SmartHire.

Provides TF-IDF vectorization pipelines for resumes and job descriptions.
"""

from pathlib import Path
from typing import Optional, Tuple, List
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_VECTORIZER_PATH,
    JOB_TFIDF_MAX_FEATURES,
    JOB_TFIDF_NGRAM_RANGE,
    JOB_TFIDF_MIN_DF,
    JOB_TFIDF_MAX_DF,
    JOB_VECTORIZER_PATH,
)


def build_tfidf_vectorizer(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
    min_df: int = 1,
    max_df: float = 0.90,
    sublinear_tf: bool = True,
    stop_words: str = "english",
) -> TfidfVectorizer:
    """Instantiate a TF-IDF vectorizer configured for resume classification text (Model A).

    Args:
        max_features: Maximum vocabulary size.
        ngram_range: Lower and upper boundary of n-values for n-grams.
        min_df: Minimum document frequency threshold.
        max_df: Maximum document frequency threshold (ignore corpus-wide stop terms).
        sublinear_tf: Apply sublinear scaling (1 + log(tf)) to dampen high term frequency effects.
        stop_words: Standard stop word list to filter.

    Returns:
        Configured TfidfVectorizer instance.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        stop_words=stop_words,
    )


def build_job_vectorizer(
    max_features: int = JOB_TFIDF_MAX_FEATURES,
    ngram_range: Tuple[int, int] = JOB_TFIDF_NGRAM_RANGE,
    min_df: int = JOB_TFIDF_MIN_DF,
    max_df: float = JOB_TFIDF_MAX_DF,
    sublinear_tf: bool = True,
    stop_words: str = "english",
) -> TfidfVectorizer:
    """Instantiate a TF-IDF vectorizer configured for the 140k job corpus (Model B Recommender).

    Configured with robust document frequency bounds to filter ultra-rare noise while
    damping repetitive high-frequency keywords.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        stop_words=stop_words,
    )


def prepare_job_text(df: pd.DataFrame) -> List[str]:
    """Combine title, skills, and description into a single corpus document per job.

    Handles missing text fields safely without fabricating data.

    Args:
        df: DataFrame containing at least ['title', 'skills', 'description'] columns.

    Returns:
        List of combined text strings corresponding to each job row.
    """
    titles = df["title"].fillna("").astype(str).str.strip()
    skills = df["skills"].fillna("").astype(str).str.strip()
    descs = df["description"].fillna("").astype(str).str.strip()

    # Combine non-empty fields with space separator
    combined = titles + " " + skills + " " + descs
    return combined.str.strip().tolist()


def save_vectorizer(
    vectorizer: TfidfVectorizer,
    output_path: Optional[Path] = None,
) -> Path:
    """Serialize a fitted TF-IDF vectorizer to disk."""
    dest = output_path or TFIDF_VECTORIZER_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, dest)
    return dest


def load_vectorizer(file_path: Optional[Path] = None) -> TfidfVectorizer:
    """Load a serialized TF-IDF vectorizer from disk."""
    path = file_path or TFIDF_VECTORIZER_PATH
    if not path.exists():
        raise FileNotFoundError(f"Vectorizer artifact not found at {path}")
    return joblib.load(path)


def save_job_vectorizer(
    vectorizer: TfidfVectorizer,
    output_path: Optional[Path] = None,
) -> Path:
    """Serialize the fitted job recommender TF-IDF vectorizer to disk."""
    dest = output_path or JOB_VECTORIZER_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, dest)
    return dest


def load_job_vectorizer(file_path: Optional[Path] = None) -> TfidfVectorizer:
    """Load the serialized job recommender TF-IDF vectorizer from disk."""
    path = file_path or JOB_VECTORIZER_PATH
    if not path.exists():
        raise FileNotFoundError(f"Job vectorizer artifact not found at {path}")
    return joblib.load(path)
