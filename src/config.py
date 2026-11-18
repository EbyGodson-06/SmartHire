"""Configuration module for SmartHire.

Defines project directory paths, data paths, model paths,
constants, and hyperparameter defaults across the pipeline.
"""

from pathlib import Path

# Base project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Specific raw dataset paths
RESUME_RAW_PATH = RAW_DATA_DIR / "UpdatedResumeDataSet.csv"
NAUKRI_RAW_PATH = RAW_DATA_DIR / "naukri_com-job_sample.csv"

# LinkedIn raw dataset directory and relational paths
LINKEDIN_RAW_DIR = RAW_DATA_DIR / "linkedin"
LINKEDIN_POSTINGS_PATH = LINKEDIN_RAW_DIR / "postings.csv"
LINKEDIN_JOB_SKILLS_PATH = LINKEDIN_RAW_DIR / "jobs" / "job_skills.csv"
LINKEDIN_SKILLS_MAP_PATH = LINKEDIN_RAW_DIR / "mappings" / "skills.csv"
LINKEDIN_COMPANIES_PATH = LINKEDIN_RAW_DIR / "companies" / "companies.csv"

# Backward compatibility alias
LINKEDIN_RAW_PATH = LINKEDIN_POSTINGS_PATH

# Processed output paths
JOB_CORPUS_PROCESSED_PATH = PROCESSED_DATA_DIR / "merged_job_corpus.parquet"
JOB_CORPUS_PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "merged_job_corpus.csv"
RESUMES_PROCESSED_PATH = PROCESSED_DATA_DIR / "cleaned_resumes.parquet"
RESUMES_PROCESSED_CSV_PATH = PROCESSED_DATA_DIR / "cleaned_resumes.csv"

# Models and artifacts paths
MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
FIT_PREDICTOR_PATH = MODELS_DIR / "fit_predictor.pkl"
JOB_VECTORIZER_PATH = MODELS_DIR / "job_vectorizer.pkl"
JOB_MATRIX_PATH = MODELS_DIR / "job_tfidf_matrix.npz"
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_model.pkl"
NMF_MODEL_PATH = MODELS_DIR / "nmf_model.pkl"

# Reports and figures paths
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.pdf"

# Common job corpus unified schema (Official Section 2 & 7)
JOB_CORPUS_COLUMNS = [
    "title",
    "company",
    "location",
    "skills",
    "description",
    "experience",
]

# Random seed and default hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Classifier TF-IDF Parameters (Model A)
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# Recommender TF-IDF Parameters (Model B Unsupervised)
JOB_TFIDF_MAX_FEATURES = 10000
JOB_TFIDF_NGRAM_RANGE = (1, 2)
JOB_TFIDF_MIN_DF = 1
JOB_TFIDF_MAX_DF = 0.85

# Clustering Parameters (Unsupervised Job Families)
KMEANS_N_CLUSTERS = 6
KMEANS_BATCH_SIZE = 2048
CLUSTERING_SAMPLE_SIZE = 10000

# Topic Modeling Parameters (NMF Unsupervised Learning)
NMF_N_TOPICS = 6
NMF_MAX_ITER = 120
