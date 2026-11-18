"""Integration tests for SmartHire application pipeline (Week 3 Phase 2).

Tests:
- PDF/DOCX/TXT text extraction and validation.
- Empty resume and whitespace handling.
- Corrupted documents and invalid format handling.
- Resume classification integration with raw decision margins.
- Job recommender integration with skill-gap outputs.
- End-to-end run_pipeline execution.
"""

import io
import pytest
import numpy as np
import pandas as pd
from docx import Document
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from src.pipeline import (
    extract_and_validate_resume,
    classify_resume_domain,
    recommend_jobs_for_resume,
    run_pipeline,
)
from src.models.recommender import JobRecommender
from src.features.text_features import build_job_vectorizer, load_vectorizer
from src.models.classifier import load_classifier
from src.config import CLASSIFIER_PATH, TFIDF_VECTORIZER_PATH


@pytest.fixture(scope="module")
def classifier_and_vectorizer():
    """Load the approved Model A classifier and vectorizer (small ~1 MB artifact)."""
    clf = load_classifier(CLASSIFIER_PATH)
    vec = load_vectorizer(TFIDF_VECTORIZER_PATH)
    return clf, vec


@pytest.fixture
def synthetic_recommender():
    """Create a lightweight synthetic JobRecommender for fast unit testing."""
    corpus_data = {
        "title": [
            "Senior Python Developer",
            "Junior Data Scientist",
            "DevOps Engineer",
        ],
        "company": ["Alpha Tech", "Beta AI", "Cloud Corp"],
        "location": ["Bengaluru", "Remote", "Pune"],
        "skills": [
            "Python, Django, PostgreSQL, Docker",
            "Python, Machine Learning, NLP, Scikit-Learn",
            "AWS, Kubernetes, Docker, CI/CD",
        ],
        "description": [
            "Looking for an experienced Python developer with Django and relational databases.",
            "Passionate junior data scientist with strong ML and NLP skills.",
            "DevOps engineer managing AWS cloud infrastructure with Kubernetes.",
        ],
        "experience": ["3-5 years", "1-2 years", "2-4 years"],
    }
    df = pd.DataFrame(corpus_data)
    rec = JobRecommender()
    rec.fit(df)
    return rec


# --------------------------------------------------------------------------
# 1. Parsing & Extraction Tests
# --------------------------------------------------------------------------

def test_extract_and_validate_resume_txt():
    """Verify TXT buffer extraction and cleaning."""
    txt_content = b"Experienced Python Engineer with PostgreSQL and Docker skills."
    buf = io.BytesIO(txt_content)
    raw, cleaned = extract_and_validate_resume(file_input=buf, filename="resume.txt")

    assert "Experienced Python Engineer" in raw
    assert "experienced python engineer" in cleaned
    assert "postgresql" in cleaned
    assert "docker" in cleaned


def test_extract_and_validate_resume_docx():
    """Verify DOCX buffer extraction and cleaning."""
    doc = Document()
    doc.add_paragraph("Data Analyst with SQL and Tableau expertise.")
    doc.add_paragraph("Bachelor of Science in Computer Science.")
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    raw, cleaned = extract_and_validate_resume(file_input=buf, filename="resume.docx")
    assert "Data Analyst with SQL" in raw
    assert "data analyst with sql and tableau" in cleaned


def test_extract_and_validate_resume_pdf():
    """Verify PDF buffer extraction and cleaning."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots()
        ax.text(0.1, 0.5, "Machine Learning Engineer with PyTorch and NLP experience.")
        pdf.savefig(fig)
        plt.close(fig)

    buf.seek(0)
    raw, cleaned = extract_and_validate_resume(file_input=buf, filename="resume.pdf")
    assert "Machine Learning Engineer" in raw
    assert "machine learning engineer" in cleaned
    assert "nlp experience" in cleaned


def test_extract_and_validate_resume_text_fallback():
    """Verify raw text paste input fallback."""
    pasted = "Senior React Developer with JavaScript, TypeScript, and Redux."
    raw, cleaned = extract_and_validate_resume(text_input=pasted)
    assert raw == pasted
    assert "senior react developer" in cleaned


# --------------------------------------------------------------------------
# 2. Input Validation & Error Handling Tests
# --------------------------------------------------------------------------

def test_empty_resume_handling():
    """Verify empty inputs raise informative ValueError."""
    with pytest.raises(ValueError, match="No resume input provided"):
        extract_and_validate_resume()

    with pytest.raises(ValueError, match="Extracted resume text is empty"):
        extract_and_validate_resume(file_input=io.BytesIO(b""), filename="empty.txt")

    with pytest.raises(ValueError, match="No resume input provided"):
        extract_and_validate_resume(text_input="   \n\t  ")


def test_invalid_format_handling():
    """Verify unsupported formats raise ValueError."""
    buf = io.BytesIO(b"binary data")
    with pytest.raises(ValueError, match="Unsupported file format"):
        extract_and_validate_resume(file_input=buf, filename="resume.png")


def test_corrupted_document_handling():
    """Verify corrupted PDF/DOCX streams raise RuntimeError."""
    corrupt_pdf = io.BytesIO(b"Not a valid PDF header")
    with pytest.raises(RuntimeError, match="Failed to extract text from PDF"):
        extract_and_validate_resume(file_input=corrupt_pdf, filename="bad.pdf")

    corrupt_docx = io.BytesIO(b"Not a valid docx zip")
    with pytest.raises(RuntimeError, match="Failed to extract text from DOCX"):
        extract_and_validate_resume(file_input=corrupt_docx, filename="bad.docx")


# --------------------------------------------------------------------------
# 3. Model A Classification Integration Tests
# --------------------------------------------------------------------------

def test_classify_resume_domain_integration(classifier_and_vectorizer):
    """Verify Model A classification returns top-3 raw decision margins."""
    clf, vec = classifier_and_vectorizer
    resume_text = "Data Scientist with Python, PyTorch, Deep Learning, Natural Language Processing, and SQL."

    result = classify_resume_domain(resume_text, clf, vec, top_k=3)

    assert result["predicted_category"] == "Data Science"
    assert result["score_type"] == "raw_decision_margin"
    assert len(result["top_categories"]) == 3

    top1 = result["top_categories"][0]
    assert top1["rank"] == 1
    assert top1["category"] == "Data Science"
    assert isinstance(top1["decision_score"], float)

    # Margins must be strictly descending; top1 is the maximum margin across all 25 classes
    assert result["top_categories"][0]["decision_score"] >= result["top_categories"][1]["decision_score"]
    assert result["top_categories"][1]["decision_score"] >= result["top_categories"][2]["decision_score"]


# --------------------------------------------------------------------------
# 4. Job Recommender & Skill-Gap Integration Tests
# --------------------------------------------------------------------------

def test_recommend_jobs_for_resume_integration(synthetic_recommender):
    """Verify recommendation generation and structured skill-gap metadata."""
    resume_text = "Python developer experienced in Django, PostgreSQL, and Docker."

    recs = recommend_jobs_for_resume(
        resume_text=resume_text,
        recommender=synthetic_recommender,
        top_n=2,
        include_skill_gap=True,
    )

    assert len(recs) == 2
    top_match = recs[0]

    assert top_match["rank"] == 1
    assert "Python" in top_match["title"]
    assert "similarity_score" in top_match
    assert "similarity_percentage" in top_match
    assert "candidate_skills" in top_match
    assert "matching_skills" in top_match
    assert "missing_skills" in top_match
    assert "overlap_percentage" in top_match
    assert top_match["has_skill_info"] is True

    # Candidate should have Django, PostgreSQL, Python, Docker
    assert "Python" in top_match["candidate_skills"]
    assert "Django" in top_match["matching_skills"]
    assert top_match["overlap_percentage"] > 0.0


# --------------------------------------------------------------------------
# 5. End-to-End Pipeline Integration Test
# --------------------------------------------------------------------------

def test_run_pipeline_end_to_end(classifier_and_vectorizer, synthetic_recommender):
    """Verify run_pipeline executes the full workflow without errors."""
    clf, vec = classifier_and_vectorizer
    input_text = "Senior Python engineer with Machine Learning, NLP, Docker, and PostgreSQL."

    output = run_pipeline(
        text_input=input_text,
        classifier=clf,
        classifier_vectorizer=vec,
        recommender=synthetic_recommender,
        top_n=3,
        include_skill_gap=True,
    )

    assert output["raw_text"] == input_text
    assert output["cleaned_text"] != ""
    assert isinstance(output["candidate_skills"], list)
    assert "Python" in output["candidate_skills"]
    assert "Docker" in output["candidate_skills"]

    # Classification
    assert output["classification"] is not None
    assert output["classification"]["predicted_category"] != ""

    # Recommendations
    assert output["total_recommendations"] == 3
    assert len(output["recommendations"]) == 3
    assert output["recommendations"][0]["rank"] == 1


# --------------------------------------------------------------------------
# 6. Real Artifact Integration Test (Using Loaded Recommender Slice)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_recommender_slice():
    """Load real job_vectorizer and a representative slice of the job matrix and corpus.

    Exercises the exact serialized vectorizer and matrix weights without loading
    the full 2 GB application dataset into memory for every test run.
    """
    import joblib
    import scipy.sparse as sp
    from src.config import JOB_VECTORIZER_PATH, JOB_MATRIX_PATH, JOB_CORPUS_PROCESSED_PATH

    job_vec = joblib.load(JOB_VECTORIZER_PATH)
    full_mat = sp.load_npz(JOB_MATRIX_PATH)
    mat_slice = full_mat[:200]
    df_slice = pd.read_parquet(JOB_CORPUS_PROCESSED_PATH).iloc[:200].copy()

    return JobRecommender(job_corpus_df=df_slice, job_matrix=mat_slice, vectorizer=job_vec)


def test_run_pipeline_with_real_recommender_artifacts(classifier_and_vectorizer, real_recommender_slice):
    """Verify run_pipeline using BOTH the actual saved classifier and real recommender artifacts."""
    clf, vec = classifier_and_vectorizer
    resume_text = "Data Scientist with Python, Machine Learning, Deep Learning, PyTorch, and NLP."

    output = run_pipeline(
        text_input=resume_text,
        classifier=clf,
        classifier_vectorizer=vec,
        recommender=real_recommender_slice,
        top_n=3,
        include_skill_gap=True,
    )

    # 1. Classification with real LinearSVC
    assert output["classification"]["predicted_category"] == "Data Science"
    assert output["classification"]["score_type"] == "raw_decision_margin"

    # 2. Recommender with real TF-IDF weights and matrix slice
    recs = output["recommendations"]
    assert len(recs) == 3
    for r in recs:
        assert isinstance(r["similarity_score"], float)
        assert 0.0 <= r["similarity_score"] <= 1.0  # Cosine similarity bounds
        assert "rank" in r
        assert "skill_gap" in r


# --------------------------------------------------------------------------
# 7. Pre-Top-N Filtering Correctness & NMF Terminology Audit Tests
# --------------------------------------------------------------------------

@pytest.fixture
def filtering_test_recommender():
    """Synthetic multi-city, multi-experience recommender for testing pre-Top-N filtering."""
    corpus_data = {
        "title": [
            "Senior Python Developer - Alpha",      # Sim ~0.75, Bengaluru, Senior
            "Mid Python Developer - Beta",          # Sim ~0.29, Bengaluru, Mid
            "Python Engineer - Gamma",              # Sim ~0.28, Bengaluru, Entry
            "Python Tech Lead - Delta",             # Sim ~0.27, Bengaluru, Senior
            "Python Cloud Architect - Epsilon",     # Sim ~0.25, Pune, Senior  <-- Outside unfiltered Top 3!
            "Junior Python Coder - Zeta",           # Sim ~0.15, Pune, Entry
            "Frontend React Engineer - Eta",        # Sim ~0.00, Mumbai, Mid
        ],
        "company": ["Co A", "Co B", "Co C", "Co D", "Co E", "Co F", "Co G"],
        "location": ["Bengaluru", "Bengaluru", "Bengaluru", "Bengaluru", "Pune", "Pune", "Mumbai"],
        "skills": [
            "Python, Django, AWS, Docker",
            "Python, Django, Docker",
            "Python, SQL",
            "Python, Architecture",
            "Python, AWS, Docker, Kubernetes",
            "Python, Git",
            "React, JavaScript, HTML, CSS",
        ],
        "description": [
            "Senior python developer with django aws docker in Bengaluru",
            "Mid python developer with django docker in Bengaluru",
            "Python engineer with sql in Bengaluru",
            "Python tech lead with architecture in Bengaluru",
            "Python cloud architect with aws docker kubernetes in Pune",
            "Junior python coder with git in Pune",
            "Frontend react engineer in Mumbai",
        ],
        "experience": ["Senior", "Mid", "Entry", "Senior", "Senior", "Entry", "Mid"],
    }
    df = pd.DataFrame(corpus_data)
    rec = JobRecommender()
    rec.fit(df)
    return rec


def test_pre_top_n_filtering_includes_job_outside_unfiltered_top_n(filtering_test_recommender):
    """Verify a job outside the unfiltered Top-2 is included when it satisfies filters."""
    rec = filtering_test_recommender
    resume = "Senior Python developer with AWS and Docker"

    # 1. Unfiltered Top 2 are Alpha and Beta (both Bengaluru roles)
    unfiltered = recommend_jobs_for_resume(resume, rec, top_n=2)
    unfiltered_titles = [r["title"] for r in unfiltered]
    assert len(unfiltered) == 2
    assert "Python Cloud Architect - Epsilon" not in unfiltered_titles

    # 2. Filtered for Pune: Epsilon (which was outside unfiltered Top 2) must now be returned as Rank 1!
    filtered_pune = recommend_jobs_for_resume(resume, rec, top_n=2, locations=["Pune"])
    assert len(filtered_pune) >= 1
    assert filtered_pune[0]["title"] == "Python Cloud Architect - Epsilon"
    assert filtered_pune[0]["location"] == "Pune"
    assert filtered_pune[0]["rank"] == 1


def test_pre_top_n_filtering_satisfies_all_selected_filters(filtering_test_recommender):
    """Verify returned jobs strictly satisfy every selected filter simultaneously."""
    rec = filtering_test_recommender
    resume = "Python developer with Django and Docker"

    # Filter for Location=Bengaluru, Experience=Senior, Min Similarity=0.20
    results = recommend_jobs_for_resume(
        resume_text=resume,
        recommender=rec,
        top_n=5,
        min_similarity=0.20,
        locations=["Bengaluru"],
        experience_levels=["Senior"],
    )

    assert len(results) > 0
    for r in results:
        assert r["location"] == "Bengaluru"
        assert r["experience"] == "Senior"
        assert r["similarity_score"] >= 0.20


def test_pre_top_n_filtering_sorted_descending(filtering_test_recommender):
    """Verify that filtered results remain strictly sorted by descending cosine similarity."""
    rec = filtering_test_recommender
    resume = "Python developer with AWS, Docker, Kubernetes, and Git"

    results = recommend_jobs_for_resume(
        resume_text=resume,
        recommender=rec,
        top_n=10,
        locations=["Bengaluru", "Pune"],
    )

    scores = [r["similarity_score"] for r in results]
    assert len(scores) >= 2
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]


def test_pre_top_n_filtering_respects_top_n_bound(filtering_test_recommender):
    """Verify that no more than the requested top_n jobs are returned."""
    rec = filtering_test_recommender
    resume = "Python developer"

    for req_n in [1, 2, 4]:
        results = recommend_jobs_for_resume(resume, rec, top_n=req_n, locations=["Bengaluru"])
        assert len(results) <= req_n


def test_pre_top_n_filtering_empty_results_graceful(filtering_test_recommender):
    """Verify that impossible filter criteria returns an empty list gracefully."""
    rec = filtering_test_recommender
    resume = "Python developer"

    # Location not in synthetic corpus
    results_loc = recommend_jobs_for_resume(resume, rec, top_n=5, locations=["NonExistentCity"])
    assert results_loc == []

    # Impossible similarity threshold (> 1.0)
    results_sim = recommend_jobs_for_resume(resume, rec, top_n=5, min_similarity=1.50)
    assert results_sim == []


def test_pre_top_n_filtering_disabled_matches_recommender(filtering_test_recommender):
    """Verify that with filters disabled, results match the existing recommender exactly."""
    rec = filtering_test_recommender
    resume = "Python engineer with SQL"

    direct_recs = rec.recommend(resume, top_n=4, include_skill_gap=True)
    pipeline_recs = recommend_jobs_for_resume(
        resume_text=resume,
        recommender=rec,
        top_n=4,
        min_similarity=0.0,
        locations=None,
        experience_levels=None,
        include_skill_gap=True,
    )

    assert len(direct_recs) == len(pipeline_recs)
    for d, p in zip(direct_recs, pipeline_recs):
        assert d["rank"] == p["rank"]
        assert d["title"] == p["title"]
        assert d["similarity_score"] == p["similarity_score"]
        assert d["skill_gap"]["overlap_percentage"] == p["skill_gap"]["overlap_percentage"]


def test_nmf_document_topic_loadings_terminology():
    """Verify NMF transform outputs raw non-negative weights W without percentage normalization."""
    from src.pipeline import analyze_career_clusters_and_topics
    from src.models.topic_modeling import load_topic_model
    from src.features.text_features import load_job_vectorizer
    from src.config import NMF_MODEL_PATH, JOB_VECTORIZER_PATH

    nmf = load_topic_model(NMF_MODEL_PATH)
    vec = load_job_vectorizer(JOB_VECTORIZER_PATH)

    resume_text = "Experienced Data Scientist in Machine Learning, Deep Learning, PyTorch, and NLP."
    insights = analyze_career_clusters_and_topics(resume_text, vec, nmf_model=nmf)

    nmf_info = insights["nmf"]
    assert nmf_info is not None
    assert "topic_loadings" in nmf_info
    assert len(nmf_info["topic_loadings"]) == 6

    # Verify all loadings are non-negative floats
    loadings = [t["loading"] for t in nmf_info["topic_loadings"]]
    for val in loadings:
        assert isinstance(val, float)
        assert val >= 0.0

    # Verify loadings are NOT normalized to 100% or 1.0
    # In NMF coordinate descent, raw W loadings are typically small decimals (e.g. 0.01 - 0.05)
    raw_sum = sum(loadings)
    assert not np.isclose(raw_sum, 1.0, atol=1e-3)
    assert not np.isclose(raw_sum, 100.0, atol=1e-3)


