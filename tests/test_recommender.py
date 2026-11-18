"""Unit and integration tests for Job Recommender and Skill-Gap Engine."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.sparse as sp

from src.config import JOB_CORPUS_PROCESSED_PATH, JOB_CORPUS_COLUMNS
from src.features.text_features import prepare_job_text, build_job_vectorizer
from src.features.match_features import (
    extract_skills_from_text,
    extract_job_skills,
    compute_skill_overlap,
    analyze_skill_gap,
    normalize_skill_to_canonical,
    match_c,
    match_r,
    DEFAULT_SKILL_VOCABULARY,
    GENERIC_TERMS_TO_EXCLUDE,
)
from src.models.recommender import JobRecommender


@pytest.fixture
def sample_job_corpus():
    """Create a controlled mini job corpus for fast unit tests."""
    data = [
        {
            "title": "Senior Python Data Scientist",
            "company": "DataTech Labs",
            "location": "Bengaluru",
            "skills": "python, machine learning, sql, pandas",
            "description": "Looking for a Data Scientist proficient in Python, Scikit-learn, Machine Learning, and SQL.",
            "experience": "3 - 5 yrs",
        },
        {
            "title": "Full Stack Java Developer",
            "company": "Enterprise Cloud Corp",
            "location": "Hyderabad",
            "skills": "java, spring boot, microservices",
            "description": "Enterprise Java developer with expertise in Spring Boot, REST API, Hibernate, and MySQL.",
            "experience": "4 - 8 yrs",
        },
        {
            "title": "Cloud DevOps Engineer",
            "company": "InfraScale Inc",
            "location": "Remote",
            "skills": "aws, docker, kubernetes, ci/cd",
            "description": "Build CI/CD pipelines using Jenkins, Docker, Kubernetes, Terraform, and AWS cloud.",
            "experience": "2 - 5 yrs",
        },
        {
            "title": "HR Generalist",
            "company": "People Solutions",
            "location": "Mumbai",
            "skills": "human resources, recruitment, onboarding",
            "description": "Manage recruitment, onboarding, employee engagement, HR policies, and payroll.",
            "experience": "1 - 3 yrs",
        },
        {
            "title": "Data Entry Clerk",
            "company": "Local Services",
            "location": "Chennai",
            "skills": "",  # Empty skills to test missing skill handling
            "description": "Typing and data entry work. Basic computer knowledge required.",
            "experience": "0 - 1 yrs",
        },
    ]
    return pd.DataFrame(data)


# ==============================================================================
# AUDIT TESTS: CANONICAL NORMALIZATION & SKILL RESOLUTION
# ==============================================================================

def test_nlp_and_natural_language_processing_unification():
    """Verify that NLP and Natural Language Processing resolve to the same canonical skill."""
    norm_nlp = normalize_skill_to_canonical("nlp")
    norm_full = normalize_skill_to_canonical("natural language processing")
    assert norm_nlp == "Natural Language Processing"
    assert norm_full == "Natural Language Processing"
    assert norm_nlp == norm_full

    # Candidate has 'Natural Language Processing', Job requires 'NLP' -> 100% match, 0 missing
    res = compute_skill_overlap(["Natural Language Processing"], ["nlp"])
    assert res["matching_skills"] == ["Natural Language Processing"]
    assert res["missing_skills"] == []
    assert res["overlap_percentage"] == 100.0


def test_ml_and_machine_learning_unification():
    """Verify that ML and Machine Learning resolve to the same canonical skill."""
    norm_ml = normalize_skill_to_canonical("ml")
    norm_full = normalize_skill_to_canonical("machine learning")
    assert norm_ml == "Machine Learning"
    assert norm_full == "Machine Learning"
    assert norm_ml == norm_full

    res = compute_skill_overlap(["ml"], ["machine learning"])
    assert res["matching_skills"] == ["Machine Learning"]
    assert res["missing_skills"] == []
    assert res["overlap_percentage"] == 100.0


def test_k8s_and_kubernetes_unification():
    """Verify that K8s and Kubernetes resolve to the same canonical skill."""
    norm_k8s = normalize_skill_to_canonical("k8s")
    norm_kube = normalize_skill_to_canonical("kubernetes")
    assert norm_k8s == "Kubernetes"
    assert norm_kube == "Kubernetes"
    assert norm_k8s == norm_kube

    res = compute_skill_overlap(["kubernetes"], ["k8s"])
    assert res["matching_skills"] == ["Kubernetes"]
    assert res["missing_skills"] == []
    assert res["overlap_percentage"] == 100.0


def test_duplicate_aliases_do_not_inflate_denominator():
    """Verify that mentioning both 'Kubernetes' and 'K8s' counts as 1 required skill."""
    job_skills_with_aliases = ["kubernetes", "k8s", "docker"]
    resume_skills = ["kubernetes"]

    res = compute_skill_overlap(resume_skills, job_skills_with_aliases)
    # Total unique required skills should be 2 ('Kubernetes', 'Docker'), not 3
    assert res["matching_skills"] == ["Kubernetes"]
    assert res["missing_skills"] == ["Docker"]
    assert res["overlap_count"] == 1
    assert res["overlap_percentage"] == 50.0  # 1 / 2 = 50.0%, NOT 1 / 3 (33.3%)


def test_generic_terms_excluded_from_skills():
    """Verify broad categories (engineering, information technology, compliance) are excluded."""
    for term in ["engineering", "information technology", "compliance", "general business"]:
        assert normalize_skill_to_canonical(term) is None

    # Test that extract_job_skills filters out generic LinkedIn categories
    extracted = extract_job_skills("engineering, information technology", "Job description with Python")
    assert "Engineering" not in extracted
    assert "Information Technology" not in extracted
    assert "Python" in extracted


def test_single_letter_boundary_handling_c_and_r():
    """Verify C and R language matching without false positives from P&C, B2C, R&D, etc."""
    # Test C rejection
    assert not match_c("Remote opportunity for P&C insurance actuary.")
    assert not match_c("Platform built for B2C consumer products.")
    assert not match_c("Reporting to C-level executive team.")
    # Test C acceptance
    assert match_c("Proficient in C, C++, and Python.")
    assert match_c("Embedded C software engineer.")
    assert match_c("Experience in C programming and Linux.")

    # Test R rejection
    assert not match_r("5 years in R&D team working on algorithms.")
    assert not match_r("Compliant with Section R-22 protocols.")
    # Test R acceptance
    assert match_r("Proficient in Python and R for data science.")
    assert match_r("Experience with R programming and RStudio.")
    assert match_r("Skills: R, SQL, Tableau.")


def test_human_services_does_not_trigger_human_resources():
    """Verify that 'human services' agency/experience is never confused with 'Human Resources'."""
    text_human_services = (
        "Work experience in a human services agency ideally related to "
        "human services programs including electronic benefits transfer (EBT)."
    )
    extracted = extract_skills_from_text(text_human_services)
    assert "Human Resources" not in extracted


def test_hr_hourly_pay_rate_rejection():
    """Verify pay rate ($60/hr, /hr) and shift durations do not trigger 'Human Resources'."""
    from src.features.match_features import match_hr

    text_rate = "Job location: Washington, DC. Pay rate: $60/hr C2C on W2. Experience in human services."
    assert not match_hr(text_rate)
    extracted = extract_skills_from_text(text_rate)
    assert "Human Resources" not in extracted

    # Genuine HR context should trigger
    text_real_hr = "Senior HR Generalist managing HR policies, employee onboarding, and talent acquisition."
    assert match_hr(text_real_hr)
    assert "Human Resources" in extract_skills_from_text(text_real_hr)


def test_continuous_integration_distinct_from_cicd():
    """Verify Continuous Integration and Continuous Deployment are not prematurely merged into CI/CD."""
    ci_norm = normalize_skill_to_canonical("continuous integration")
    cd_norm = normalize_skill_to_canonical("continuous deployment")
    cicd_norm = normalize_skill_to_canonical("ci/cd")

    assert ci_norm == "Continuous Integration"
    assert cd_norm == "Continuous Deployment"
    assert cicd_norm == "CI/CD"
    assert ci_norm != cicd_norm
    assert cd_norm != cicd_norm


def test_pyspark_and_apache_spark_specificity():
    """Verify PySpark and Apache Spark maintain distinct specific identities."""
    spark_norm = normalize_skill_to_canonical("spark")
    pyspark_norm = normalize_skill_to_canonical("pyspark")

    assert spark_norm == "Apache Spark"
    assert pyspark_norm == "PySpark"
    assert spark_norm != pyspark_norm


def test_generic_node_does_not_match_nodejs():
    """Verify that graph/tree/cluster 'node' does not falsely trigger 'Node.js'."""
    text_node_cluster = (
        "Implemented graph traversal algorithms identifying each leaf node and "
        "worker node in the distributed cluster."
    )
    extracted = extract_skills_from_text(text_node_cluster)
    assert "Node.js" not in extracted

    text_real_node = "Built full-stack web applications using Node.js, Express, and React."
    extracted_real = extract_skills_from_text(text_real_node)
    assert "Node.js" in extracted_real


# ==============================================================================
# STANDARD RECOMMENDER & SKILL OVERLAP TESTS
# ==============================================================================

def test_prepare_job_text_safe_handling(sample_job_corpus):
    """Test that job text preparation combines title, skills, and description safely."""
    docs = prepare_job_text(sample_job_corpus)
    assert len(docs) == len(sample_job_corpus)
    assert "Senior Python Data Scientist" in docs[0]
    assert "machine learning" in docs[0]
    assert "Data Entry Clerk" in docs[4]


def test_extract_skills_canonical_formatting():
    """Verify regex boundary handling returns canonical formatted names."""
    text = "Proficient in Python, C++, C#, .NET core, and R for data science. Experienced in Machine Learning."
    skills = extract_skills_from_text(text)
    assert "Python" in skills
    assert "C++" in skills
    assert "C#" in skills
    assert ".NET" in skills
    assert "R" in skills
    assert "Machine Learning" in skills
    assert "Data Science" in skills


def test_compute_skill_overlap_missing_job_skills():
    """Verify that jobs with no skill information do NOT claim skills are missing."""
    resume_skills = ["Python", "SQL"]
    job_skills = []

    result = compute_skill_overlap(resume_skills, job_skills)
    assert result["has_skill_info"] is False
    assert result["status"] == "Insufficient skill information"
    assert result["matching_skills"] == []
    assert result["missing_skills"] == []
    assert result["overlap_percentage"] == 0.0


def test_job_recommender_fit_and_ranking(sample_job_corpus):
    """Test recommender fitting, descending ranking, and top_n behavior."""
    recommender = JobRecommender().fit(sample_job_corpus)

    resume_query = "Data Scientist with strong Python, machine learning, and SQL experience."
    top_recs = recommender.recommend(resume_query, top_n=3, include_skill_gap=True)

    assert len(top_recs) == 3
    # Scores must be strictly in descending order
    scores = [r["similarity_score"] for r in top_recs]
    assert scores == sorted(scores, reverse=True)

    # Top recommendation should be the Data Scientist role
    top_job = top_recs[0]
    assert "Data Scientist" in top_job["title"]
    assert top_job["rank"] == 1
    assert top_job["similarity_score"] > 0.1

    # Check skill gap output
    gap = top_job["skill_gap"]
    assert "Python" in gap["matching_skills"]
    assert gap["has_skill_info"] is True


def test_job_recommender_top_n_bounds_and_empty(sample_job_corpus):
    """Test edge cases: empty resume, top_n=1, top_n exceeding corpus size."""
    recommender = JobRecommender().fit(sample_job_corpus)

    # Empty resume
    assert recommender.recommend("", top_n=5) == []
    assert recommender.recommend("   ", top_n=5) == []

    # top_n = 1
    rec_1 = recommender.recommend("Java Spring Boot Developer", top_n=1)
    assert len(rec_1) == 1
    assert "Java" in rec_1[0]["title"]

    # top_n exceeding corpus size (corpus has 5 jobs)
    rec_all = recommender.recommend("Python", top_n=100)
    assert len(rec_all) == 5

    # top_n <= 0
    assert recommender.recommend("Python", top_n=0) == []
    assert recommender.recommend("Python", top_n=-5) == []


def test_recommender_save_and_load(sample_job_corpus, tmp_path):
    """Test saving and loading JobRecommender artifacts."""
    recommender = JobRecommender().fit(sample_job_corpus)
    vec_path = tmp_path / "job_vec.pkl"
    mat_path = tmp_path / "job_mat.npz"

    recommender.save(vectorizer_path=vec_path, matrix_path=mat_path)
    assert vec_path.exists()
    assert mat_path.exists()

    loaded = JobRecommender.load(
        job_corpus_df=sample_job_corpus,
        vectorizer_path=vec_path,
        matrix_path=mat_path,
    )

    query = "DevOps engineer with Docker and AWS"
    orig_recs = recommender.recommend(query, top_n=2)
    loaded_recs = loaded.recommend(query, top_n=2)

    assert orig_recs[0]["title"] == loaded_recs[0]["title"]
    assert orig_recs[0]["similarity_score"] == loaded_recs[0]["similarity_score"]


def test_canonical_job_corpus_integrity():
    """Verify that canonical processed job corpus has exact 140,225 rows and 6 columns."""
    assert JOB_CORPUS_PROCESSED_PATH.exists(), "Processed job corpus parquet must exist."
    df = pd.read_parquet(JOB_CORPUS_PROCESSED_PATH)
    assert len(df) == 140225, f"Corpus row count must be 140,225; found {len(df)}"
    assert list(df.columns) == JOB_CORPUS_COLUMNS, f"Columns mismatch: {list(df.columns)}"


def test_serialized_job_artifacts_and_fast_inference():
    """Verify serialized job vectorizer and matrix load in sub-second and perform inference."""
    from src.config import JOB_VECTORIZER_PATH, JOB_MATRIX_PATH
    import scipy.sparse as sp
    import joblib

    assert JOB_VECTORIZER_PATH.exists(), "Serialized job_vectorizer.pkl must exist."
    assert JOB_MATRIX_PATH.exists(), "Serialized job_tfidf_matrix.npz must exist."

    vec = joblib.load(JOB_VECTORIZER_PATH)
    mat = sp.load_npz(JOB_MATRIX_PATH)

    assert mat.shape == (140225, 10000), f"Expected shape (140225, 10000), got {mat.shape}"
    sparsity = 1.0 - (mat.nnz / (mat.shape[0] * mat.shape[1]))
    assert sparsity > 0.95, f"Expected sparsity > 95%, got {sparsity*100:.2f}%"

    df_head = pd.read_parquet(JOB_CORPUS_PROCESSED_PATH)
    recommender = JobRecommender(job_corpus_df=df_head, job_matrix=mat, vectorizer=vec)

    recs = recommender.recommend(
        "Python Data Scientist with Machine Learning and SQL expertise",
        top_n=5,
        include_skill_gap=True,
    )

    assert len(recs) == 5
    # Verify strict descending order
    scores = [r["similarity_score"] for r in recs]
    assert scores == sorted(scores, reverse=True)

    # Each rank should be unique 1..5
    ranks = [r["rank"] for r in recs]
    assert ranks == [1, 2, 3, 4, 5]

    # Check skill gap keys
    for r in recs:
        assert "skill_gap" in r
        assert "matching_skills" in r["skill_gap"]
        assert "missing_skills" in r["skill_gap"]
        assert "overlap_percentage" in r["skill_gap"]


def test_recommend_df_formatting(sample_job_corpus):
    """Verify recommend_df outputs a clean pandas DataFrame with required columns."""
    recommender = JobRecommender().fit(sample_job_corpus)
    df_recs = recommender.recommend_df("Data Scientist with Python and SQL", top_n=3)

    assert isinstance(df_recs, pd.DataFrame)
    assert len(df_recs) == 3
    expected_cols = [
        "Rank", "Title", "Company", "Location", "Skills",
        "Experience", "Match Score (%)", "Matching Skills",
        "Missing Skills", "Skill Overlap (%)"
    ]
    for col in expected_cols:
        assert col in df_recs.columns
