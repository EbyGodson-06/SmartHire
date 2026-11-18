"""End-to-end SmartHire application pipeline module.

Wires together:
1. Resume input & document text extraction (PDF, DOCX, TXT, or text paste).
2. Text preprocessing and cleaning (preserving technical entities and programming languages).
3. Model A resume category classification with raw LinearSVC decision margin scores.
4. Content-based job recommendation using TF-IDF cosine similarity against the 140k job corpus,
   supporting pre-top-N filtering across locations, experience levels, and minimum cosine similarity.
5. Canonical skill extraction and skill-gap analysis against target postings.
6. Unsupervised K-Means job family cluster prediction and NMF topic loading profiling.
"""

from pathlib import Path
from typing import Union, BinaryIO, Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

from src.parsing.resume_parser import extract_resume_text
from src.data.preprocess import clean_text
from src.models.classifier import predict_category
from src.models.recommender import JobRecommender
from src.features.match_features import (
    extract_skills_from_text,
    extract_job_skills,
    compute_skill_overlap,
)

# Approved Human-Readable Cluster Labels from Notebook 04
KMEANS_CLUSTER_LABELS: Dict[int, str] = {
    0: "Accounting, Finance & Corporate Administration",
    1: "Mixed Indian Postings (Naukri Corpus / Web Scraping Boilerplate Dominant)",
    2: "Operations, Maintenance, Retail & Customer Support",
    3: "Sales, Account Management & Digital Marketing",
    4: "Healthcare, Nursing & Clinical Care Services",
    5: "Technical Engineering, Software Architecture & Data Systems",
}

# Approved Human-Readable NMF Topic Labels from Notebook 04
NMF_TOPIC_LABELS: Dict[int, str] = {
    0: "Mixed Business Mgmt, Sales & Financial Ops",
    1: "Scraped Web / Naukri Template Artifact",
    2: "Healthcare, Nursing & Clinical Care",
    3: "Software Engineering & Technical Systems",
    4: "EEO & Legal Compliance Disclaimers",
    5: "Mixed Customer Service, Retail & Facility Ops",
}


def extract_and_validate_resume(
    file_input: Optional[Union[str, Path, BinaryIO]] = None,
    filename: str = "",
    text_input: str = "",
) -> Tuple[str, str]:
    """Extract and preprocess resume text from uploaded file or raw text input.

    Args:
        file_input: Optional file path or file-like binary stream (PDF, DOCX, TXT).
        filename: Optional filename for format resolution.
        text_input: Optional fallback raw resume text pasted directly.

    Returns:
        Tuple of (raw_extracted_text, cleaned_text).

    Raises:
        ValueError: If no input is provided, format is unsupported, or cleaned text is empty.
        RuntimeError: If document parser fails due to corrupted data or missing parser.
    """
    raw_text = ""

    if file_input is not None:
        raw_text = extract_resume_text(file_input, filename=filename)
    elif text_input and text_input.strip():
        raw_text = text_input.strip()
    else:
        raise ValueError("No resume input provided. Please upload a resume file or paste resume text.")

    if not raw_text or not raw_text.strip():
        raise ValueError("Extracted resume text is empty. Please provide a valid resume.")

    cleaned_text = clean_text(raw_text)

    if not cleaned_text or not cleaned_text.strip():
        raise ValueError("Resume text contains no readable alphanumeric content after preprocessing.")

    return raw_text, cleaned_text


def classify_resume_domain(
    cleaned_text: str,
    model: BaseEstimator,
    vectorizer: TfidfVectorizer,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Classify resume into one of 25 job categories using Model A (LinearSVC).

    Retrieves predicted class and top-K decision scores.
    IMPORTANT: LinearSVC decision scores are signed raw geometric margins (distances to
    hyperplanes in feature space), NOT probabilities or confidence percentages.

    Args:
        cleaned_text: Preprocessed resume text string.
        model: Trained LinearSVC classifier.
        vectorizer: Fitted TF-IDF vectorizer (5,000 features).
        top_k: Number of top decision margin classes to return.

    Returns:
        Dict containing:
            - predicted_category (str): Primary predicted domain.
            - top_categories (List[Dict]): Top-K categories with raw decision margins.
            - score_type (str): Explicitly identifies metric as 'raw_decision_margin'.
    """
    if not cleaned_text or not cleaned_text.strip():
        raise ValueError("Cleaned resume text must not be empty.")

    vec = vectorizer.transform([cleaned_text])
    predicted = model.predict(vec)[0]

    top_categories = []
    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function(vec)[0]
        classes = model.classes_
        ranked_indices = np.argsort(decision_scores)[::-1]
        k = min(top_k, len(classes))
        for idx in ranked_indices[:k]:
            top_categories.append({
                "rank": len(top_categories) + 1,
                "category": str(classes[idx]),
                "decision_score": round(float(decision_scores[idx]), 4),
            })
    else:
        top_categories.append({
            "rank": 1,
            "category": str(predicted),
            "decision_score": 1.0,
        })

    return {
        "predicted_category": str(predicted),
        "top_categories": top_categories,
        "score_type": "raw_decision_margin",
    }


def recommend_jobs_for_resume(
    resume_text: str,
    recommender: JobRecommender,
    top_n: int = 10,
    min_similarity: float = 0.0,
    locations: Optional[List[str]] = None,
    experience_levels: Optional[List[str]] = None,
    include_skill_gap: bool = True,
) -> List[Dict[str, Any]]:
    """Retrieve Top-N job recommendations and comprehensive skill-gap diagnostics.

    Applies pre-top-N filtering across locations, experience levels, and minimum
    cosine similarity before selecting the Top-N candidates, strictly preserving
    the relative cosine similarity ranking order.

    Args:
        resume_text: Raw or cleaned candidate resume text.
        recommender: Fitted or loaded JobRecommender instance.
        top_n: Number of recommendations to retrieve (default 10).
        min_similarity: Minimum cosine similarity threshold in [0.0, 1.0].
        locations: Optional list of location keywords to filter by.
        experience_levels: Optional list of experience levels to filter by.
        include_skill_gap: Whether to compute skill overlap against each posting.

    Returns:
        List of structured recommendation dictionaries including skill-gap details.
    """
    if not resume_text or not resume_text.strip():
        return []

    if recommender.job_matrix is None or recommender.vectorizer is None or recommender.job_corpus_df is None:
        raise ValueError("Recommender is not initialized or fitted. Call fit() or load() first.")

    # Check if active filters are present
    has_loc_filter = bool(locations and [l for l in locations if l.strip()])
    has_exp_filter = bool(experience_levels and [e for e in experience_levels if e.strip()])
    has_min_sim = min_similarity > 0.0

    # Fast path: no filtering required, use baseline recommender method
    if not has_loc_filter and not has_exp_filter and not has_min_sim:
        recommendations = recommender.recommend(
            resume_text=resume_text,
            top_n=top_n,
            include_skill_gap=include_skill_gap,
        )
        candidate_skills = sorted(list(extract_skills_from_text(resume_text)))
        for rec in recommendations:
            rec["candidate_skills"] = candidate_skills
            if include_skill_gap and "skill_gap" in rec:
                gap = rec["skill_gap"]
                rec["matching_skills"] = list(gap.get("matching_skills", []))
                rec["missing_skills"] = list(gap.get("missing_skills", []))
                rec["overlap_percentage"] = gap.get("overlap_percentage", 0.0)
                rec["has_skill_info"] = gap.get("has_skill_info", False)
        return recommendations

    # Filtered path: Compute full matrix similarity and filter candidates BEFORE selecting Top-N
    resume_vec = recommender.vectorizer.transform([resume_text])
    if resume_vec.nnz == 0:
        return []

    similarities = recommender.job_matrix.dot(resume_vec.T).toarray().ravel()
    corpus_df = recommender.job_corpus_df

    valid_mask = np.ones(len(similarities), dtype=bool)

    if has_min_sim:
        valid_mask &= (similarities >= min_similarity)

    if has_loc_filter:
        loc_terms = [l.strip().lower() for l in locations if l.strip()]
        loc_series = corpus_df["location"].fillna("").astype(str).str.lower()
        loc_match = loc_series.apply(lambda loc: any(term in loc for term in loc_terms)).values
        valid_mask &= loc_match

    if has_exp_filter:
        exp_terms = [e.strip().lower() for e in experience_levels if e.strip()]
        exp_series = corpus_df["experience"].fillna("").astype(str).str.lower()
        exp_match = exp_series.apply(lambda exp: any(term in exp for term in exp_terms)).values
        valid_mask &= exp_match

    passing_indices = np.where(valid_mask)[0]
    if len(passing_indices) == 0:
        return []

    # Sort passing indices strictly by cosine similarity descending
    k = min(top_n, len(passing_indices))
    sorted_passing = passing_indices[np.argsort(similarities[passing_indices])[::-1][:k]]

    candidate_skills = sorted(list(extract_skills_from_text(resume_text))) if include_skill_gap else set()

    results = []
    for rank, idx in enumerate(sorted_passing, 1):
        row = corpus_df.iloc[idx]
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
            "candidate_skills": sorted(list(candidate_skills)),
        }

        if include_skill_gap:
            job_skills = extract_job_skills(row["skills"], row["description"])
            gap_analysis = compute_skill_overlap(candidate_skills, job_skills)
            gap_analysis["resume_skills_identified"] = sorted(list(candidate_skills))
            gap_analysis["job_skills_identified"] = sorted(list(job_skills))
            rec["skill_gap"] = gap_analysis
            rec["matching_skills"] = list(gap_analysis.get("matching_skills", []))
            rec["missing_skills"] = list(gap_analysis.get("missing_skills", []))
            rec["overlap_percentage"] = gap_analysis.get("overlap_percentage", 0.0)
            rec["has_skill_info"] = gap_analysis.get("has_skill_info", False)

        results.append(rec)

    return results


def analyze_career_clusters_and_topics(
    cleaned_text: str,
    job_vectorizer: TfidfVectorizer,
    kmeans_model: Optional[BaseEstimator] = None,
    nmf_model: Optional[BaseEstimator] = None,
) -> Dict[str, Any]:
    """Analyze resume alignment with unsupervised K-Means job clusters and NMF topics.

    Mathematical Convention:
    - In Non-negative Matrix Factorization (X ~ W * H), H is the topic-term component
      matrix (nmf.components_), while W is the document-topic loading matrix.
    - nmf.transform(resume_vector) computes the document-topic loading vector (W).
    - These values are NMF Document-Topic Loadings (Raw Weights W), NOT probabilities or
      percentages. They do not sum to 100%. A resume can legitimately load across multiple topics.
    - K-Means cluster labels distinguish genuine career families from the
      scraped Naukri boilerplate artifact cluster (Cluster 1).

    Args:
        cleaned_text: Preprocessed resume text string.
        job_vectorizer: Fitted job corpus TF-IDF vectorizer (10,000 features).
        kmeans_model: Optional trained MiniBatchKMeans model.
        nmf_model: Optional trained NMF topic model.

    Returns:
        Dict containing K-Means cluster assignment and NMF topic loadings profile.
    """
    if not cleaned_text or not cleaned_text.strip():
        return {"kmeans": None, "nmf": None}

    job_vec = job_vectorizer.transform([cleaned_text])

    cluster_result = None
    if kmeans_model is not None:
        c_id = int(kmeans_model.predict(job_vec)[0])
        cluster_result = {
            "cluster_id": c_id,
            "cluster_label": KMEANS_CLUSTER_LABELS.get(c_id, f"Cluster {c_id}"),
            "is_boilerplate_cluster": (c_id == 1),
            "description": (
                "Corpus artifact dominated by scraped portal navigation boilerplate."
                if c_id == 1
                else "Coherent career domain family discovered via MiniBatchKMeans."
            ),
        }

    topic_result = None
    if nmf_model is not None:
        weights = nmf_model.transform(job_vec)[0]  # raw non-negative loadings
        topic_loadings = [
            {
                "topic_id": i,
                "topic_name": NMF_TOPIC_LABELS.get(i, f"Topic {i}"),
                "loading": round(float(weights[i]), 4),
            }
            for i in range(len(weights))
        ]
        dom_idx = int(np.argmax(weights))
        topic_result = {
            "dominant_topic_id": dom_idx,
            "dominant_topic_name": NMF_TOPIC_LABELS.get(dom_idx, f"Topic {dom_idx}"),
            "dominant_topic_loading": round(float(weights[dom_idx]), 4),
            "topic_loadings": topic_loadings,
        }

    return {
        "kmeans": cluster_result,
        "nmf": topic_result,
    }


def run_pipeline(
    file_input: Optional[Union[str, Path, BinaryIO]] = None,
    filename: str = "",
    text_input: str = "",
    classifier: Optional[BaseEstimator] = None,
    classifier_vectorizer: Optional[TfidfVectorizer] = None,
    recommender: Optional[JobRecommender] = None,
    kmeans_model: Optional[BaseEstimator] = None,
    nmf_model: Optional[BaseEstimator] = None,
    top_n: int = 10,
    min_similarity: float = 0.0,
    locations: Optional[List[str]] = None,
    experience_levels: Optional[List[str]] = None,
    include_skill_gap: bool = True,
) -> Dict[str, Any]:
    """Execute the end-to-end SmartHire processing pipeline with filtering and topic profiling.

    Flow:
        Resume Input
        -> Text Extraction
        -> Text Cleaning
        -> Resume Classification (Model A)
        -> Pre-top-N Filtered Recommendation (Cosine Similarity)
        -> Skill Gap Analysis (Canonical Normalization)
        -> Unsupervised Cluster & Topic Profiling (K-Means & NMF)
        -> Structured Results

    Returns:
        Dict containing raw text, cleaned text, candidate skills, classification,
        recommendations, and unsupervised cluster/topic insights.
    """
    # 1. Extraction and Preprocessing
    raw_text, cleaned_text = extract_and_validate_resume(
        file_input=file_input,
        filename=filename,
        text_input=text_input,
    )

    # 2. Candidate Canonical Skills
    candidate_skills = sorted(list(extract_skills_from_text(cleaned_text)))

    # 3. Model A Domain Classification
    classification_result = None
    if classifier is not None and classifier_vectorizer is not None:
        classification_result = classify_resume_domain(
            cleaned_text=cleaned_text,
            model=classifier,
            vectorizer=classifier_vectorizer,
            top_k=3,
        )

    # 4. Job Recommendations & Skill-Gap Analysis
    recommendations = []
    if recommender is not None:
        recommendations = recommend_jobs_for_resume(
            resume_text=cleaned_text,
            recommender=recommender,
            top_n=top_n,
            min_similarity=min_similarity,
            locations=locations,
            experience_levels=experience_levels,
            include_skill_gap=include_skill_gap,
        )

    # 5. Unsupervised Career Clusters & NMF Topic Alignment
    career_insights = None
    if recommender is not None and recommender.vectorizer is not None:
        career_insights = analyze_career_clusters_and_topics(
            cleaned_text=cleaned_text,
            job_vectorizer=recommender.vectorizer,
            kmeans_model=kmeans_model,
            nmf_model=nmf_model,
        )

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "candidate_skills": candidate_skills,
        "classification": classification_result,
        "recommendations": recommendations,
        "total_recommendations": len(recommendations),
        "career_insights": career_insights,
    }
