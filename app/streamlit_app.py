"""SmartHire – Resume-to-Job Matching & Career Guidance Engine.

An industrial AI/ML dashboard built using classical machine learning (LinearSVC,
TF-IDF Cosine Similarity, MiniBatchKMeans, and Non-negative Matrix Factorization).
Provides candidate domain classification, semantic job recommendations,
canonical skill-gap diagnostics, and career cluster alignment.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import Counter
import pandas as pd
import numpy as np
import streamlit as st

# Ensure repository root is on sys.path for robust standalone `streamlit run app/streamlit_app.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    CLASSIFIER_PATH,
    TFIDF_VECTORIZER_PATH,
    JOB_CORPUS_PROCESSED_PATH,
    JOB_VECTORIZER_PATH,
    JOB_MATRIX_PATH,
    KMEANS_MODEL_PATH,
    NMF_MODEL_PATH,
)
from src.features.text_features import load_vectorizer
from src.models.classifier import load_classifier
from src.models.clustering import load_clustering_model
from src.models.topic_modeling import load_topic_model
from src.models.recommender import JobRecommender
from src.pipeline import (
    extract_and_validate_resume,
    classify_resume_domain,
    recommend_jobs_for_resume,
    analyze_career_clusters_and_topics,
    run_pipeline,
    KMEANS_CLUSTER_LABELS,
    NMF_TOPIC_LABELS,
)


# ==============================================================================
# 1. CACHED RESOURCE LOADERS
# ==============================================================================

@st.cache_resource(show_spinner="Loading SmartHire Model A Domain Classifier...")
def load_cached_classifier():
    """Load and cache the Model A LinearSVC classifier and resume vectorizer."""
    classifier = load_classifier(CLASSIFIER_PATH)
    vectorizer = load_vectorizer(TFIDF_VECTORIZER_PATH)
    return classifier, vectorizer


@st.cache_resource(show_spinner="Loading Canonical Job Corpus & Recommender Engine (140,225 records)...")
def load_cached_recommender():
    """Load and cache the JobRecommender with the 140,225 job corpus and sparse TF-IDF matrix."""
    corpus_df = pd.read_parquet(JOB_CORPUS_PROCESSED_PATH)
    recommender = JobRecommender.load(
        job_corpus_df=corpus_df,
        vectorizer_path=JOB_VECTORIZER_PATH,
        matrix_path=JOB_MATRIX_PATH,
    )
    return recommender


@st.cache_resource(show_spinner="Loading Unsupervised Job Clustering (K-Means) & Topic Models (NMF)...")
def load_cached_unsupervised_models():
    """Load and cache the MiniBatchKMeans (K=6) and NMF (K=6) models."""
    kmeans_model = load_clustering_model(KMEANS_MODEL_PATH)
    nmf_model = load_topic_model(NMF_MODEL_PATH)
    return kmeans_model, nmf_model


# ==============================================================================
# 2. UI STYLING & CUSTOM CSS (PREMIUM SAAS DASHBOARD)
# ==============================================================================

def apply_custom_styling():
    """Inject clean, modern SaaS dashboard CSS with white background and blue accents."""
    st.markdown(
        """
        <style>
        /* Base Page Background & Global Typography */
        .stApp {
            background-color: #FFFFFF !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
            color: #0F172A !important;
        }

        /* Layout & Spacing */
        .main .block-container {
            padding-top: 1.6rem !important;
            padding-bottom: 3.5rem !important;
            max-width: 1240px !important;
        }

        /* Top Heading Overrides */
        h1 {
            color: #0F172A !important;
            font-size: 1.7rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.025em !important;
            margin-bottom: 0.35rem !important;
            padding-top: 0 !important;
        }
        h2 {
            color: #0F172A !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.015em !important;
            margin-top: 1.2rem !important;
            margin-bottom: 0.5rem !important;
        }
        h3 {
            color: #0F172A !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            margin-top: 1rem !important;
            margin-bottom: 0.4rem !important;
        }
        h4, h5, h6 {
            color: #0F172A !important;
            font-weight: 600 !important;
        }

        /* Sidebar Styling: Pure Clean White Appearance */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.8rem !important;
            padding-bottom: 2rem !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #0F172A !important;
            font-size: 0.98rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            margin-bottom: 0.4rem !important;
        }

        /* File Uploader Dropzone Styling */
        div[data-testid="stFileUploader"] {
            background-color: #F8FAFC !important;
            border: 1.5px dashed #CBD5E1 !important;
            border-radius: 8px !important;
            padding: 12px 14px !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stFileUploader"]:hover {
            border-color: #2563EB !important;
            background-color: #F1F5F9 !important;
        }
        div[data-testid="stFileUploader"] small {
            color: #64748B !important;
        }

        /* Text Area & Text Inputs */
        div[data-testid="stTextArea"] textarea {
            border-radius: 8px !important;
            border: 1px solid #CBD5E1 !important;
            color: #0F172A !important;
            background-color: #FFFFFF !important;
            font-size: 0.88rem !important;
        }
        div[data-testid="stTextArea"] textarea:focus {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 1px #2563EB !important;
        }

        /* Primary Action Button */
        div.stButton > button[kind="primary"] {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            padding: 0.55rem 1.2rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 2px rgba(37, 99, 235, 0.2) !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #1D4ED8 !important;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3) !important;
        }

        /* Metric Cards: Modern Clean SaaS KPI Style */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            padding: 16px 20px !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04), 0 1px 2px rgba(15, 23, 42, 0.02) !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #CBD5E1 !important;
            box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.06) !important;
        }
        div[data-testid="stMetricLabel"] > div,
        div[data-testid="stMetricLabel"] > p {
            font-size: 0.76rem !important;
            color: #64748B !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.04em !important;
            white-space: normal !important;
            overflow-wrap: break-word !important;
            line-height: 1.25 !important;
            margin-bottom: 4px !important;
        }
        div[data-testid="stMetricValue"] > div {
            font-size: 1.35rem !important;
            color: #0F172A !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            white-space: normal !important;
            overflow-wrap: break-word !important;
            word-break: break-word !important;
            line-height: 1.25 !important;
        }

        /* Modern SaaS Tabs */
        div[data-testid="stTabs"] {
            margin-top: 1rem !important;
        }
        div[data-testid="stTabs"] button[role="tab"] {
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            color: #64748B !important;
            padding: 8px 14px !important;
            border-radius: 6px 6px 0 0 !important;
            border-bottom: 2px solid transparent !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stTabs"] button[role="tab"]:hover {
            color: #0F172A !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #2563EB !important;
            border-bottom: 2px solid #2563EB !important;
            background-color: transparent !important;
        }

        /* Expanders */
        div[data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            margin-bottom: 12px !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02) !important;
            overflow: hidden !important;
        }
        div[data-testid="stExpander"] summary {
            font-weight: 600 !important;
            color: #0F172A !important;
            font-size: 0.95rem !important;
            padding: 12px 16px !important;
        }
        div[data-testid="stExpander"] summary:hover {
            color: #2563EB !important;
        }

        /* DataFrames */
        div[data-testid="stDataFrame"] {
            border: 1px solid #E2E8F0 !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }

        /* Custom SaaS Card Components */
        .saas-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 20px 24px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            margin-bottom: 16px;
        }
        .saas-card-feature {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 20px;
            height: 100%;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            transition: all 0.2s ease;
        }
        .saas-card-feature:hover {
            border-color: #CBD5E1;
            box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.06);
        }
        .saas-feature-icon {
            width: 38px;
            height: 38px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.15rem;
            margin-bottom: 12px;
        }
        .saas-feature-title {
            font-size: 1rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 6px;
        }
        .saas-feature-desc {
            font-size: 0.85rem;
            color: #64748B;
            line-height: 1.55;
        }
        .saas-pill-tag {
            display: inline-flex;
            align-items: center;
            background-color: #F1F5F9;
            color: #334155;
            border: 1px solid #E2E8F0;
            border-radius: 9999px;
            padding: 4px 10px;
            font-size: 0.75rem;
            font-weight: 500;
            letter-spacing: 0.02em;
        }

        /* Skill Pill Badges */
        .skill-tag-matched {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background-color: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
            border-radius: 9999px;
            padding: 4px 10px;
            margin: 3px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .skill-tag-missing {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background-color: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FECACA;
            border-radius: 9999px;
            padding: 4px 10px;
            margin: 3px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .skill-tag-neutral {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background-color: #F8FAFC;
            color: #334155;
            border: 1px solid #E2E8F0;
            border-radius: 9999px;
            padding: 4px 10px;
            margin: 3px;
            font-size: 0.8rem;
            font-weight: 500;
        }

        /* Actionable Guidance Callout */
        .guidance-box {
            background-color: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-left: 4px solid #2563EB;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 3. MAIN APPLICATION
# ==============================================================================

def main():
    st.set_page_config(
        page_title="SmartHire | Resume Matcher & Career Guidance Engine",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styling()

    # Application Header & Branding Banner
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 12px; border-bottom: 1px solid #E2E8F0; margin-bottom: 18px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="width: 42px; height: 42px; background: #2563EB; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: #FFFFFF; font-weight: 800; font-size: 1.25rem; letter-spacing: -0.02em; box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);">
                    SH
                </div>
                <div>
                    <div style="font-size: 1.45rem; font-weight: 700; color: #0F172A; letter-spacing: -0.02em; line-height: 1.2;">
                        SmartHire
                    </div>
                    <div style="font-size: 0.85rem; color: #64748B; font-weight: 400; margin-top: 1px;">
                        Enterprise Resume-to-Job Matching & Career Guidance Engine
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                <span class="saas-pill-tag">Classical ML Engine</span>
                <span class="saas-pill-tag">140,225 Indexed Roles</span>
                <span class="saas-pill-tag">157 Canonical Skills</span>
                <span class="saas-pill-tag" style="background-color: #ECFDF5; color: #065F46; border-color: #A7F3D0;">Production Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("SmartHire – Resume-to-Job Matching & Career Guidance Engine")
    st.markdown(
        "<div style='font-size: 0.92rem; color: #64748B; margin-bottom: 20px; line-height: 1.5;'>"
        "An industrial AI/ML system combining <strong>Supervised Classification</strong> (LinearSVC), "
        "<strong>Content-Based Retrieval</strong> (TF-IDF Cosine Similarity over 140,225 postings), "
        "and <strong>Unsupervised Latent Semantic Modeling</strong> (MiniBatchKMeans & NMF Topic Modeling). "
        "<span style='color: #2563EB; font-weight: 500;'>Strictly Classical Machine Learning &bull; Transparent &bull; Zero GenAI Hallucinations</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Load All Cached Model Artifacts
    try:
        classifier, classifier_vectorizer = load_cached_classifier()
        recommender = load_cached_recommender()
        kmeans_model, nmf_model = load_cached_unsupervised_models()
    except Exception as exc:
        st.error(f"❌ Failed to load system model artifacts: {exc}")
        st.info("Please verify that all artifacts exist in `models/` and processed datasets exist in `data/processed/`.")
        st.stop()

    # --------------------------------------------------------------------------
    # SIDEBAR: RESUME INPUT & FILTER CONTROLS
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("Resume Intake")

        input_method = st.radio(
            "Input Mode:",
            options=["File Upload (.pdf, .docx, .txt)", "Direct Text Paste"],
            index=0,
        )

        uploaded_file = None
        pasted_text = ""
        filename_label = ""

        if input_method == "File Upload (.pdf, .docx, .txt)":
            uploaded_file = st.file_uploader(
                "Upload Candidate CV:",
                type=["pdf", "docx", "txt"],
                help="Supported document formats: PDF (via pypdf), DOCX (via python-docx), or plain text (.txt).",
            )
            if uploaded_file is not None:
                filename_label = uploaded_file.name
        else:
            pasted_text = st.text_area(
                "Paste Resume Text:",
                height=180,
                placeholder="Paste candidate resume text here (skills, work history, projects, domain experience)...",
            )
            filename_label = "Direct Text Input"

        st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 16px 0;'>", unsafe_allow_html=True)
        st.header("Recommendation Filters")

        top_n = st.slider(
            "Top-N Recommendations",
            min_value=3,
            max_value=25,
            value=10,
            step=1,
            help="Number of top job postings to retrieve by cosine similarity.",
        )

        min_similarity = st.slider(
            "Minimum Cosine Similarity",
            min_value=0.00,
            max_value=0.50,
            value=0.00,
            step=0.02,
            format="%.2f",
            help="Filter out postings with cosine similarity below this threshold before retrieving Top-N.",
        )

        location_options = [
            "Bengaluru",
            "Delhi NCR",
            "Mumbai",
            "Hyderabad",
            "Pune",
            "Chennai",
            "Noida",
            "United States",
            "New York, NY",
            "Chicago, IL",
            "Remote",
        ]
        selected_locations = st.multiselect(
            "Location Filter:",
            options=location_options,
            default=[],
            help="Leave empty to match all corpus locations.",
        )

        experience_options = [
            "Entry Level",
            "Mid-Senior Level",
            "Associate",
            "Director",
            "Internship",
            "Executive",
        ]
        selected_experience = st.multiselect(
            "Experience Level Filter:",
            options=experience_options,
            default=[],
            help="Leave empty to match all experience levels.",
        )

        analyze_button = st.button("Analyze Resume & Match Jobs", type="primary", use_container_width=True)

        st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 18px 0;'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 14px;">
                <div style="font-size: 0.76rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin-bottom: 6px;">
                    System Specification
                </div>
                <div style="font-size: 0.8rem; color: #334155; line-height: 1.6;">
                    &bull; <strong>Model A:</strong> LinearSVC (25 domains)<br>
                    &bull; <strong>Corpus:</strong> 140,225 indexed jobs<br>
                    &bull; <strong>Features:</strong> TF-IDF (10,000 vocab)<br>
                    &bull; <strong>Metric:</strong> Unit Cosine Similarity<br>
                    &bull; <strong>Skills:</strong> 157 canonical skills
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # PIPELINE EXECUTION & SESSION STATE MANAGEMENT
    # --------------------------------------------------------------------------
    if analyze_button:
        if uploaded_file is None and not pasted_text.strip():
            st.warning("⚠️ Please upload a resume document or paste resume text in the sidebar before analyzing.")
            return

        with st.spinner("Executing pipeline: parsing, cleaning, classification, similarity ranking, and skill-gap diagnostics..."):
            try:
                pipeline_output = run_pipeline(
                    file_input=uploaded_file,
                    filename=filename_label if uploaded_file is not None else "",
                    text_input=pasted_text,
                    classifier=classifier,
                    classifier_vectorizer=classifier_vectorizer,
                    recommender=recommender,
                    kmeans_model=kmeans_model,
                    nmf_model=nmf_model,
                    top_n=top_n,
                    min_similarity=min_similarity,
                    locations=selected_locations,
                    experience_levels=selected_experience,
                    include_skill_gap=True,
                )
                pipeline_output["input_source"] = filename_label
                st.session_state["pipeline_output"] = pipeline_output
                st.session_state["active_filters"] = {
                    "top_n": top_n,
                    "min_similarity": min_similarity,
                    "locations": selected_locations,
                    "experience": selected_experience,
                }
            except ValueError as ve:
                st.error(f"❌ Input Validation Error: {ve}")
                return
            except RuntimeError as re:
                st.error(
                    f"❌ Document Parsing Error: {re}\n\n"
                    "**Guidance:** If your document is scanned, password-protected, or corrupted, "
                    "please select **'Direct Text Paste'** in the sidebar to paste your resume content directly."
                )
                return
            except Exception as ex:
                st.error(f"❌ Pipeline Execution Error: {ex}")
                return

    # Check if analysis results exist in session state
    if "pipeline_output" not in st.session_state:
        st.info("👆 Please upload a resume or paste CV text in the sidebar and click **'Analyze Resume & Match Jobs'** to begin.")

        # Professional SaaS Dashboard Landing Section
        st.markdown(
            """
            <div class="saas-card" style="margin-top: 16px; border-left: 4px solid #2563EB;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-bottom: 6px;">
                    Intelligent Candidate-to-Job Matching with Geometric Precision
                </div>
                <div style="font-size: 0.9rem; color: #64748B; line-height: 1.6;">
                    SmartHire bridges the gap between candidate qualifications and enterprise hiring requirements using audited, transparent classical machine learning algorithms. Benchmark your resume against 140,225 real job postings across 25 career domains, identify high-priority technical skill gaps, and explore latent semantic job clusters.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_feat1, col_feat2, col_feat3 = st.columns(3)
        with col_feat1:
            st.markdown(
                """
                <div class="saas-card-feature">
                    <div class="saas-feature-icon" style="background-color: #EFF6FF; color: #2563EB;">🎯</div>
                    <div class="saas-feature-title">Content-Based Retrieval</div>
                    <div class="saas-feature-desc">
                        Sparse TF-IDF Cosine Similarity engine matching across 140,225 deduplicated postings with sub-second retrieval (&lt; 80 ms) and pre-top-N eligibility filtering.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_feat2:
            st.markdown(
                """
                <div class="saas-card-feature">
                    <div class="saas-feature-icon" style="background-color: #F0FDF4; color: #16A34A;">🏆</div>
                    <div class="saas-feature-title">Domain Classification</div>
                    <div class="saas-feature-desc">
                        LinearSVC classifier predicting across 25 career categories (Repeated 3-Fold CV Macro F1: 0.8620) with signed orthogonal hyperplane decision margins.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_feat3:
            st.markdown(
                """
                <div class="saas-card-feature">
                    <div class="saas-feature-icon" style="background-color: #F0FDFA; color: #0D9488;">🔍</div>
                    <div class="saas-feature-title">Canonical Skill Gap</div>
                    <div class="saas-feature-desc">
                        157 canonical skill taxonomy with bidirectional alias resolution (e.g., NLP &harr; Natural Language Processing) to isolate missing competencies.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px 24px; margin-top: 18px;">
                <div style="font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B; margin-bottom: 12px;">
                    Verified System Benchmarks & Specifications
                </div>
                <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                    <div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #0F172A;">140,225</div>
                        <div style="font-size: 0.8rem; color: #64748B;">Indexed Postings</div>
                    </div>
                    <div style="border-left: 1px solid #E2E8F0; padding-left: 24px;">
                        <div style="font-size: 1.35rem; font-weight: 700; color: #0F172A;">25 Domains</div>
                        <div style="font-size: 0.8rem; color: #64748B;">Classified Categories</div>
                    </div>
                    <div style="border-left: 1px solid #E2E8F0; padding-left: 24px;">
                        <div style="font-size: 1.35rem; font-weight: 700; color: #0F172A;">157 Skills</div>
                        <div style="font-size: 0.8rem; color: #64748B;">Canonical Taxonomy</div>
                    </div>
                    <div style="border-left: 1px solid #E2E8F0; padding-left: 24px;">
                        <div style="font-size: 1.35rem; font-weight: 700; color: #0F172A;">0.8620</div>
                        <div style="font-size: 0.8rem; color: #64748B;">Model A Macro F1</div>
                    </div>
                    <div style="border-left: 1px solid #E2E8F0; padding-left: 24px;">
                        <div style="font-size: 1.35rem; font-weight: 700; color: #0F172A;">&lt; 80 ms</div>
                        <div style="font-size: 0.8rem; color: #64748B;">Retrieval Latency</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    output = st.session_state["pipeline_output"]
    recs = output.get("recommendations", [])
    candidate_skills = output.get("candidate_skills", [])
    classification = output.get("classification", {})
    career_insights = output.get("career_insights", {})

    # Top KPI Banner
    pred_domain = classification.get("predicted_category", "N/A")
    top_score = recs[0]["similarity_score"] if recs else 0.0
    top_overlap = recs[0]["overlap_percentage"] if recs else 0.0

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Predicted Domain", pred_domain)
    kpi_col2.metric("Matching Jobs Found", f"{len(recs)} roles")
    kpi_col3.metric("Top Cosine Similarity", f"{top_score:.4f}" if recs else "0.0000")
    kpi_col4.metric("Extracted Skills", f"{len(candidate_skills)} skills")

    st.write("")

    # ==============================================================================
    # FOUR-TAB DASHBOARD INTERFACE
    # ==============================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "Recommended Jobs & Rankings",
        "Skill-Gap Diagnostics",
        "Career Clusters & Topics",
        "Resume Diagnostics",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: RECOMMENDED JOBS & RANKINGS
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader("Content-Based Job Recommendations")
        st.caption(
            "Jobs retrieved by computing the sparse TF-IDF cosine similarity between the candidate resume and 140,225 job postings. "
            "**Scientific Note:** 'Cosine Similarity' is a dimensionless geometric metric in [0.0, 1.0] representing directional alignment in TF-IDF space; "
            "it is **not** a probability or percentage of requirements met. 'Skill Overlap' measures the true percentage of required skills satisfied."
        )

        if not recs:
            st.warning(
                "⚠️ No job recommendations matched your query with the current filter criteria.\n\n"
                "**Suggestions:**\n"
                "- Lower the **Minimum Cosine Similarity** slider in the sidebar.\n"
                "- Clear or broaden the **Location Filter** and **Experience Filter** selections.\n"
                "- Ensure your resume text contains standard industry keywords."
            )
        else:
            # High-level Summary Table
            table_rows = []
            for r in recs:
                gap = r.get("skill_gap", {})
                overlap_text = (
                    f"{gap.get('overlap_percentage', 0.0):.1f}%"
                    if gap.get("has_skill_info")
                    else "N/A (Info Missing)"
                )
                table_rows.append({
                    "Rank": r["rank"],
                    "Job Title": r["title"],
                    "Company": r["company"] if r["company"] else "Not specified",
                    "Location": r["location"] if r["location"] else "Not specified",
                    "Experience": r["experience"] if r["experience"] else "Not specified",
                    "Cosine Similarity": f"{r['similarity_score']:.4f}",
                    "Skill Overlap": overlap_text,
                })

            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            # Detailed Expandable Cards
            st.markdown("### Detailed Job Profiles & Requirements")
            for r in recs:
                gap = r.get("skill_gap", {})
                matching = r.get("matching_skills", [])
                missing = r.get("missing_skills", [])
                has_info = r.get("has_skill_info", False)

                expander_title = (
                    f"Rank {r['rank']} | {r['title']} @ {r['company'] or 'Confidential'} "
                    f"({r['location'] or 'Location N/A'}) — Cosine Sim: {r['similarity_score']:.4f}"
                )

                with st.expander(expander_title):
                    card_col1, card_col2, card_col3 = st.columns(3)
                    card_col1.metric("Cosine Similarity", f"{r['similarity_score']:.4f}")
                    if has_info:
                        card_col2.metric("Skill Overlap", f"{gap.get('overlap_percentage', 0.0):.1f}%")
                        card_col3.metric("Required Skills Count", f"{len(matching) + len(missing)}")
                    else:
                        card_col2.metric("Skill Overlap", "N/A")
                        card_col3.metric("Skill Info Status", "Insufficient")

                    st.markdown("**Structured Skill Breakdown:**")
                    if matching:
                        match_badges = "".join([f"<span class='skill-tag-matched'>✓ {s}</span>" for s in matching])
                        st.markdown(f"**Matching Skills ({len(matching)}):**<br>{match_badges}", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color: #64748B; font-style: italic; font-size: 0.88rem;'>No matching canonical skills detected for this posting.</span>", unsafe_allow_html=True)

                    if has_info:
                        if missing:
                            miss_badges = "".join([f"<span class='skill-tag-missing'>✕ {s}</span>" for s in missing])
                            st.markdown(f"**Missing Required Skills ({len(missing)}):**<br>{miss_badges}", unsafe_allow_html=True)
                        else:
                            st.markdown("<span style='color: #059669; font-weight: 500; font-size: 0.88rem;'>All explicit job skill requirements are possessed by candidate!</span>", unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ *This posting did not provide explicit structured technical skills in its corpus record.*")

                    if r.get("skills"):
                        st.markdown(f"**Corpus Skill Metadata:** `{r['skills']}`")

                    st.markdown("**Full Job Description:**")
                    desc = r.get("description", "")
                    if desc:
                        st.text_area(
                            f"Description (Rank {r['rank']})",
                            value=desc,
                            height=140,
                            disabled=True,
                            key=f"desc_{r['rank']}",
                        )
                    else:
                        st.caption("No textual job description provided.")

    # --------------------------------------------------------------------------
    # TAB 2: SKILL GAP & CAREER GUIDANCE
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("Skill-Gap Diagnostics & Career Guidance")
        st.caption(
            "Comparative evaluation of candidate skills against target posting requirements using canonical skill normalization. "
            "Abbreviation aliases (e.g. 'NLP' and 'Natural Language Processing') are normalized to prevent false negatives."
        )

        # Candidate Skills Inventory
        st.markdown("#### 1. Candidate's Extracted Skills")
        if candidate_skills:
            badges_html = "".join([f"<span class='skill-tag-neutral'>📌 {s}</span>" for s in candidate_skills])
            st.markdown(f"**Total Identified Skills ({len(candidate_skills)}):**<br>{badges_html}", unsafe_allow_html=True)
        else:
            st.warning("No canonical skills from the 157-skill vocabulary were identified in the resume text.")

        st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 18px 0;'>", unsafe_allow_html=True)

        if not recs:
            st.info("Please generate job recommendations to evaluate specific posting skill gaps.")
        else:
            st.markdown("#### 2. Target Job Skill-Gap Analysis")

            job_selector_options = [
                f"Rank {r['rank']}: {r['title']} @ {r['company'] or 'Not specified'}"
                for r in recs
            ]
            selected_job_idx = st.selectbox(
                "Select Target Job to Inspect:",
                options=range(len(recs)),
                format_func=lambda i: job_selector_options[i],
            )

            chosen_job = recs[selected_job_idx]
            chosen_gap = chosen_job.get("skill_gap", {})
            matching_skills = chosen_job.get("matching_skills", [])
            missing_skills = chosen_job.get("missing_skills", [])
            has_info = chosen_job.get("has_skill_info", False)

            sg_col1, sg_col2, sg_col3, sg_col4 = st.columns(4)
            sg_col1.metric("Job Rank", f"#{chosen_job['rank']}")
            sg_col2.metric("Cosine Similarity", f"{chosen_job['similarity_score']:.4f}")
            if has_info:
                sg_col3.metric("Skill Overlap", f"{chosen_gap.get('overlap_percentage', 0.0):.1f}%")
                sg_col4.metric("Missing Skills", f"{len(missing_skills)}")
            else:
                sg_col3.metric("Skill Overlap", "N/A")
                sg_col4.metric("Skill Status", "Insufficient Info")

            st.write("")
            col_match, col_miss = st.columns(2)

            with col_match:
                st.markdown(
                    """
                    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; height: 100%; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">
                        <div style="font-size: 0.95rem; font-weight: 600; color: #0F172A; margin-bottom: 10px;">
                            ✓ Matching Possessed Skills
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                if matching_skills:
                    match_html = "".join([f"<span class='skill-tag-matched'>✓ {s}</span>" for s in matching_skills])
                    st.markdown(match_html, unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color: #64748B; font-size: 0.88rem;'>No matching technical skills detected for this posting.</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_miss:
                st.markdown(
                    """
                    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px; height: 100%; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);">
                        <div style="font-size: 0.95rem; font-weight: 600; color: #0F172A; margin-bottom: 10px;">
                            ✕ Missing Required Skills
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                if has_info:
                    if missing_skills:
                        miss_html = "".join([f"<span class='skill-tag-missing'>✕ {s}</span>" for s in missing_skills])
                        st.markdown(miss_html, unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color: #059669; font-weight: 500; font-size: 0.88rem;'>You possess 100% of the required technical skills for this role!</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color: #64748B; font-size: 0.88rem;'>Insufficient skill information in this job posting record.</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 20px 0;'>", unsafe_allow_html=True)

            # Actionable Aggregated Guidance
            st.markdown("#### 3. Actionable Career Guidance & Skill Priorities")
            st.caption(
                "Objective recommendations derived strictly from recurring missing skills across your top recommended roles. "
                "No fabricated courses, certifications, or subjective requirements."
            )

            # Aggregate all missing skills across recommendations
            all_missing = []
            for r in recs:
                if r.get("has_skill_info"):
                    all_missing.extend(r.get("missing_skills", []))

            if all_missing:
                missing_counter = Counter(all_missing)
                top_missing_skills = missing_counter.most_common(6)

                st.markdown("**Most In-Demand Skills Missing from Your Profile:**")
                for i in range(0, len(top_missing_skills), 3):
                    chunk = top_missing_skills[i : i + 3]
                    cols = st.columns(3)
                    for col, (skill, count) in zip(cols, chunk):
                        col.metric(skill, f"Needed in {count} roles")

                skills_list_str = ", ".join([f"<strong>`{s}`</strong>" for s, _ in top_missing_skills[:3]])
                st.markdown(
                    f"""
                    <div class="guidance-box">
                        <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 4px; color: #1E40AF;">
                            Strategic Skill Development Priority
                        </div>
                        <div style="font-size: 0.88rem; line-height: 1.6; color: #1E3A8A;">
                            To substantially increase candidate alignment for <strong>{pred_domain}</strong> positions, prioritize building documented project experience in: {skills_list_str}. Demonstrating these core technologies in portfolio repositories or work history will directly close your highest-frequency gaps.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.success("Your profile covers the technical requirements across all evaluated matching postings!")

    # --------------------------------------------------------------------------
    # TAB 3: CAREER CLUSTERS & TOPICS
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("Career Clusters & Latent Topic Alignment")
        st.caption(
            "Combines Supervised Resume Classification (Model A LinearSVC), "
            "Unsupervised Job Family Discovery (K-Means K=6), and Latent Semantic Factorization (NMF K=6)."
        )

        col_cls, col_kmn = st.columns(2)

        # Section 1: Model A Classification & Margins
        with col_cls:
            st.markdown("#### 1. Model A Domain Classification")
            st.metric("Predicted Domain Category", pred_domain)

            st.markdown("**Top-3 LinearSVC Decision Margins:**")
            top_cats = classification.get("top_categories", [])
            margin_df = pd.DataFrame([
                {
                    "Rank": item["rank"],
                    "Category": item["category"],
                    "LinearSVC Margin": f"{item['decision_score']:+.4f}",
                }
                for item in top_cats
            ])
            st.dataframe(margin_df, use_container_width=True, hide_index=True)
            st.caption(
                "ℹ️ **Decision Margins Note:** LinearSVC decision margins represent signed orthogonal distances ($w^T x + b$) "
                "from the separating hyperplanes in TF-IDF space (positive indicates predicted domain membership). "
                "These are raw geometric margins, **not** confidence percentages or calibrated probabilities."
            )

        # Section 2: K-Means Job Family
        with col_kmn:
            st.markdown("#### 2. K-Means Job Family Cluster (K = 6)")
            kmeans_info = career_insights.get("kmeans", {})
            if kmeans_info:
                c_id = kmeans_info["cluster_id"]
                c_label = kmeans_info["cluster_label"]
                st.metric(f"Assigned Job Family (Cluster {c_id})", c_label)
                if kmeans_info.get("is_boilerplate_cluster"):
                    st.warning(
                        "⚠️ **Cluster 1 Note:** This cluster is an artifact dominated by web crawler and portal navigation boilerplate "
                        "(`candidate profile`, `company profile`) rather than a genuine career domain."
                    )
                else:
                    st.info(f"ℹ️ **Cluster Context:** {kmeans_info.get('description', '')}")

                # Reference table of grounded cluster labels
                with st.expander("View All Grounded K-Means Cluster Labels"):
                    cluster_table = [
                        {"Cluster ID": k, "Grounded Domain Label": v}
                        for k, v in KMEANS_CLUSTER_LABELS.items()
                    ]
                    st.dataframe(pd.DataFrame(cluster_table), use_container_width=True, hide_index=True)

        st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 20px 0;'>", unsafe_allow_html=True)

        # Section 3: NMF Topic Modeling Profile
        st.markdown("#### 3. NMF Latent Topic Profile (K = 6)")
        st.caption(
            "Non-negative Matrix Factorization ($X \\approx W \\cdot H$) decomposes job descriptions and candidate resumes into 6 semantic latent topics. "
            "**Mathematical Convention:** $H$ is the topic-term component matrix (`nmf.components_`), while $W$ is the document-topic matrix. "
            "For a candidate resume, `nmf.transform(resume_vector)` produces its document-topic loading vector ($W$). "
            "These values are **NMF Document-Topic Loadings (Raw Weights)**, **not** probabilities, confidence scores, or percentages. "
            "They do not sum to 100%, and a resume can legitimately exhibit non-zero loadings across multiple career topics."
        )

        nmf_info = career_insights.get("nmf", {})
        if nmf_info:
            topic_loadings = nmf_info.get("topic_loadings", [])
            dom_topic_name = nmf_info.get("dominant_topic_name", "N/A")
            dom_weight = nmf_info.get("dominant_topic_loading", 0.0)

            t_col1, t_col2 = st.columns([1, 2])
            with t_col1:
                st.metric("Dominant NMF Topic", dom_topic_name)
                st.metric("Dominant Document-Topic Loading (W)", f"{dom_weight:.4f}")

            with t_col2:
                # Topic Loading Bar Chart with SaaS Primary Blue Color
                chart_df = pd.DataFrame({
                    "Topic": [t["topic_name"] for t in topic_loadings],
                    "NMF Document-Topic Loadings (Raw Weights)": [t["loading"] for t in topic_loadings],
                }).set_index("Topic")
                st.bar_chart(chart_df, color="#2563EB")

            # Detailed Topic Table
            st.markdown("**Complete 6-Topic Document Loadings (W):**")
            topic_table = [
                {
                    "Topic ID": t["topic_id"],
                    "Topic Description": t["topic_name"],
                    "NMF Document-Topic Loadings (Raw Weights)": f"{t['loading']:.4f}",
                }
                for t in topic_loadings
            ]
            st.dataframe(pd.DataFrame(topic_table), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 4: RESUME PREVIEW & DIAGNOSTICS
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("Resume Text Extraction & Diagnostics")
        st.caption(
            "Displays the raw input metadata, extraction status, and cleaned preprocessed text used by the vectorizer and classifier."
        )

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        d_col1.metric("Input Source", output.get("input_source", "Direct Input"))
        d_col2.metric("Extraction Status", "✅ Success")
        d_col3.metric("Raw Characters", len(output.get("raw_text", "")))
        d_col4.metric("Cleaned Words", len(output.get("cleaned_text", "").split()))

        st.markdown("#### Extracted Canonical Skills")
        if candidate_skills:
            st.markdown(
                "".join([f"<span class='skill-tag-neutral'>📌 {s}</span>" for s in candidate_skills]),
                unsafe_allow_html=True,
            )
        else:
            st.info("No canonical technical skills identified in the extracted text.")

        st.markdown("#### Cleaned Resume Text Preview")
        st.text_area(
            "Preprocessed Resume Text (Normalized, Lowercase, Artifacts Stripped):",
            value=output.get("cleaned_text", ""),
            height=260,
            disabled=True,
        )

        st.caption(
            "Text cleaned using `clean_text()` preserving programming languages (C++, C#, .NET, Node.js, R) "
            "while stripping HTML entities, URLs, emails, phone numbers, and formatting artifacts."
        )


if __name__ == "__main__":
    main()
