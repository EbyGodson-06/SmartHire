"""Job Recommender Engine module (Unsupervised Learning).

Computes Cosine Similarity between TF-IDF representations of candidate
resumes and the indexed 140,225 job corpus to retrieve Top-N matching job postings.
Integrates transparent skill-gap analysis for each recommendation.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    JOB_CORPUS_COLUMNS,
    JOB_VECTORIZER_PATH,
    JOB_MATRIX_PATH,
    JOB_TFIDF_MAX_FEATURES,
    JOB_TFIDF_NGRAM_RANGE,
    JOB_TFIDF_MIN_DF,
    JOB_TFIDF_MAX_DF,
)
from src.features.text_features import build_job_vectorizer, prepare_job_text
from src.features.match_features import (
    extract_skills_from_text,
    extract_job_skills,
    compute_skill_overlap,
    analyze_skill_gap,
)


class JobRecommender:
    """Content-based job recommender using TF-IDF and Cosine Similarity."""

    def __init__(
        self,
        job_corpus_df: Optional[pd.DataFrame] = None,
        job_matrix: Optional[sp.csr_matrix] = None,
        vectorizer: Optional[TfidfVectorizer] = None,
    ):
        self.job_corpus_df = job_corpus_df
        self.job_matrix = job_matrix
        self.vectorizer = vectorizer

    def fit(
        self,
        job_corpus_df: pd.DataFrame,
        vectorizer: Optional[TfidfVectorizer] = None,
    ) -> "JobRecommender":
        """Index the job corpus and fit the TF-IDF feature space.

        Args:
            job_corpus_df: DataFrame containing the 6 standard corpus columns:
                ['title', 'company', 'location', 'skills', 'description', 'experience'].
            vectorizer: Optional pre-configured TfidfVectorizer. If None, builds default.

        Returns:
            self: The fitted JobRecommender instance.
        """
        # Validate columns
        missing_cols = [col for col in JOB_CORPUS_COLUMNS if col not in job_corpus_df.columns]
        if missing_cols:
            raise ValueError(f"Job corpus DataFrame missing required columns: {missing_cols}")

        self.job_corpus_df = job_corpus_df.reset_index(drop=True)

        # Prepare unified job document strings (title + skills + description)
        job_docs = prepare_job_text(self.job_corpus_df)

        # Initialize vectorizer if not provided
        if vectorizer is None:
            self.vectorizer = build_job_vectorizer()
        else:
            self.vectorizer = vectorizer

        # Fit and transform into sparse CSR matrix
        self.job_matrix = self.vectorizer.fit_transform(job_docs)

        # Ensure matrix is in CSR format for fast row slicing and dot products
        if not sp.isspmatrix_csr(self.job_matrix):
            self.job_matrix = self.job_matrix.tocsr()

        return self

    def recommend(
        self,
        resume_text: str,
        top_n: int = 10,
        include_skill_gap: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return Top-N matching jobs ranked by cosine similarity.

        Args:
            resume_text: Raw candidate resume text.
            top_n: Number of recommendations to return (default 10).
            include_skill_gap: Whether to compute skill overlap diagnostics for each match.

        Returns:
            List of dictionaries containing rank, job metadata, similarity score,
            and skill-gap analysis.
        """
        if self.job_matrix is None or self.vectorizer is None or self.job_corpus_df is None:
            raise ValueError("Recommender is not initialized or fitted. Call fit() or load() first.")

        if top_n <= 0:
            return []

        # Handle empty or whitespace resume text safely
        if not resume_text or not isinstance(resume_text, str) or not resume_text.strip():
            return []

        # Vectorize single resume query -> shape (1, V)
        resume_vec = self.vectorizer.transform([resume_text])

        # If resume contains zero vocabulary terms, handle gracefully
        if resume_vec.nnz == 0:
            # All similarities will be 0.0
            similarities = np.zeros(self.job_matrix.shape[0], dtype=np.float32)
        else:
            # Cosine similarity: since TF-IDF vectors are L2-normalized,
            # cosine similarity is the sparse matrix-vector dot product
            similarities = self.job_matrix.dot(resume_vec.T).toarray().ravel()

        # Handle top_n bounds
        n_candidates = len(self.job_corpus_df)
        k = min(top_n, n_candidates)

        # Fast top-K retrieval
        if k < n_candidates:
            # Argpartition gets top K indices unsorted, then sort only top K
            partitioned_idx = np.argpartition(similarities, -k)[-k:]
            top_indices = partitioned_idx[np.argsort(similarities[partitioned_idx])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]

        # Candidate resume skills extracted once for all recommendations
        resume_skills = extract_skills_from_text(resume_text) if include_skill_gap else set()

        results = []
        for rank, idx in enumerate(top_indices, 1):
            row = self.job_corpus_df.iloc[idx]
            sim_score = float(similarities[idx])
            sim_percentage = round(sim_score * 100.0, 2)

            rec = {
                "rank": rank,
                "title": row["title"],
                "company": row["company"],
                "location": row["location"],
                "skills": row["skills"],
                "experience": row["experience"],
                "description": row["description"],
                "similarity_score": round(sim_score, 4),
                "similarity_percentage": sim_percentage,
            }

            if include_skill_gap:
                job_skills = extract_job_skills(row["skills"], row["description"])
                gap_analysis = compute_skill_overlap(resume_skills, job_skills)
                gap_analysis["resume_skills_identified"] = sorted(list(resume_skills))
                gap_analysis["job_skills_identified"] = sorted(list(job_skills))
                rec["skill_gap"] = gap_analysis

            results.append(rec)

        return results

    def recommend_df(
        self,
        resume_text: str,
        top_n: int = 10,
        include_skill_gap: bool = True,
    ) -> pd.DataFrame:
        """Return Top-N matching jobs as a formatted pandas DataFrame.

        Useful for notebook display and Streamlit rendering.
        """
        recs = self.recommend(resume_text, top_n=top_n, include_skill_gap=include_skill_gap)
        if not recs:
            return pd.DataFrame()

        rows = []
        for r in recs:
            row_dict = {
                "Rank": r["rank"],
                "Title": r["title"],
                "Company": r["company"] if r["company"] else "Not specified",
                "Location": r["location"] if r["location"] else "Not specified",
                "Skills": r["skills"] if r["skills"] else "Not specified",
                "Experience": r["experience"] if r["experience"] else "Not specified",
                "Match Score (%)": r["similarity_percentage"],
            }
            if include_skill_gap and "skill_gap" in r:
                gap = r["skill_gap"]
                row_dict["Matching Skills"] = ", ".join(gap["matching_skills"]) if gap["matching_skills"] else "None"
                row_dict["Missing Skills"] = ", ".join(gap["missing_skills"]) if gap["missing_skills"] else "None"
                row_dict["Skill Overlap (%)"] = f"{gap['overlap_percentage']}%" if gap["has_skill_info"] else "N/A (Info Missing)"
            rows.append(row_dict)

        return pd.DataFrame(rows)

    def save(
        self,
        vectorizer_path: Optional[Union[str, Path]] = None,
        matrix_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """Serialize the fitted TF-IDF vectorizer and sparse job matrix to disk."""
        if self.vectorizer is None or self.job_matrix is None:
            raise ValueError("Cannot save an unfitted recommender.")

        v_path = Path(vectorizer_path or JOB_VECTORIZER_PATH)
        m_path = Path(matrix_path or JOB_MATRIX_PATH)

        v_path.parent.mkdir(parents=True, exist_ok=True)
        m_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.vectorizer, v_path)
        sp.save_npz(m_path, self.job_matrix)

        return {"vectorizer_path": v_path, "matrix_path": m_path}

    @classmethod
    def load(
        cls,
        job_corpus_df: pd.DataFrame,
        vectorizer_path: Optional[Union[str, Path]] = None,
        matrix_path: Optional[Union[str, Path]] = None,
    ) -> "JobRecommender":
        """Instantiate a JobRecommender from serialized vectorizer and sparse matrix files."""
        v_path = Path(vectorizer_path or JOB_VECTORIZER_PATH)
        m_path = Path(matrix_path or JOB_MATRIX_PATH)

        if not v_path.exists():
            raise FileNotFoundError(f"Job vectorizer artifact not found at {v_path}")
        if not m_path.exists():
            raise FileNotFoundError(f"Job matrix artifact not found at {m_path}")

        vectorizer = joblib.load(v_path)
        job_matrix = sp.load_npz(m_path)

        if len(job_corpus_df) != job_matrix.shape[0]:
            raise ValueError(
                f"Row count mismatch between corpus DataFrame ({len(job_corpus_df)}) "
                f"and sparse matrix ({job_matrix.shape[0]})."
            )

        return cls(
            job_corpus_df=job_corpus_df.reset_index(drop=True),
            job_matrix=job_matrix,
            vectorizer=vectorizer,
        )
