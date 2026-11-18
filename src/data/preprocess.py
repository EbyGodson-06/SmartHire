"""Preprocessing and text cleaning utilities for SmartHire.

Handles:
- Text cleaning and HTML/artifact stripping while preserving programming
  languages and technical terms (e.g. C++, C#, .NET, Node.js).
- Resume deduplication and normalization (166 unique records across 25 domains).
- Naukri job listings cleaning, location standardization, and schema mapping.
- LinkedIn job postings joining (1-to-1 aggregated skills), company recovery, and schema mapping.
- Merging datasets into a unified 6-column job corpus:
  [title, company, location, skills, description, experience].
"""

import html
import re
from typing import List, Optional, Set
import pandas as pd

from src.config import JOB_CORPUS_COLUMNS


def clean_text(text: str, lowercase: bool = True) -> str:
    """Clean and normalize raw text string.

    Performs:
    - HTML entity decoding (e.g. &amp; -> &)
    - HTML tag removal (<...>)
    - URL removal (http/https/www)
    - Email address removal
    - Long telephone/numeric sequences removal (10+ digits)
    - Unicode artifact and bullet stripping (\ufffd, \u2022, \u00a0, etc.)
    - Whitespace normalization
    - Preserves meaningful technical terms, programming languages (C++, C#, .NET, Node.js, R),
      and domain punctuation.
    """
    if not isinstance(text, str):
        return ""

    # Unescape HTML entities
    cleaned = html.unescape(text)

    # Remove HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Remove email addresses
    cleaned = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", cleaned)

    # Remove 10+ digit phone numbers while preserving software version numbers (e.g. 3.10)
    cleaned = re.sub(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3,5}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b", " ", cleaned)

    # Remove replacement characters, strange bullet points, non-breaking spaces
    cleaned = re.sub(r"[\ufffd\u2022\u2023\u25e6\u2043\u2219\u00a0\r\t]", " ", cleaned)

    # Convert to lowercase if requested (default True for NLP consistency)
    if lowercase:
        cleaned = cleaned.lower()

    # Normalize multiple whitespace characters to a single space
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()

    return cleaned


def clean_naukri_description(text: str) -> str:
    """Clean Naukri job description and strip crawler boilerplate."""
    if not isinstance(text, str):
        return ""

    cleaned = text
    # Strip crawler boilerplate prefix: 'Job Description  Send me Jobs like this'
    cleaned = re.sub(r"^\s*job description\s*[\ufffd\-–—]?\s*send me jobs like this\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"send me jobs like this", " ", cleaned, flags=re.IGNORECASE)

    return clean_text(cleaned)


def normalize_location(loc: str) -> str:
    """Normalize and harmonize job location strings.

    Standardizes common Indian tech hubs and location variations
    (e.g. 'Bengaluru/Bangalore , Bengaluru / Bangalore' -> 'Bengaluru'),
    while preserving international locations cleanly.
    """
    if not isinstance(loc, str) or not loc.strip():
        return ""

    cleaned = re.sub(r"\s+", " ", loc).strip(" ,")

    # City mapping dictionary for common Indian variations
    city_map = {
        "bangalore": "Bengaluru",
        "bengaluru": "Bengaluru",
        "mumbai": "Mumbai",
        "hyderabad": "Hyderabad",
        "secunderabad": "Hyderabad",
        "delhi": "Delhi NCR",
        "ncr": "Delhi NCR",
        "noida": "Noida",
        "gurgaon": "Gurugram",
        "gurugram": "Gurugram",
        "chennai": "Chennai",
        "pune": "Pune",
        "kolkata": "Kolkata",
        "calcutta": "Kolkata",
        "ahmedabad": "Ahmedabad",
    }

    parts = [p.strip() for p in re.split(r"[,/]", cleaned) if p.strip()]
    found = []
    for part in parts:
        part_clean = re.sub(r"[^a-zA-Z]", "", part).lower()
        for key, val in city_map.items():
            if key in part_clean:
                if val not in found:
                    found.append(val)
                break

    if found:
        return ", ".join(found)

    return cleaned


def preprocess_resumes(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the raw resume dataset.

    - Deduplicates on ('Category', 'Resume') down to the 166 unique records.
    - Cleans resume text while preserving technical keywords.
    - Validates that all 25 categories are preserved.

    Returns:
        pd.DataFrame with columns ['Category', 'Resume']
    """
    # Deduplicate in the processed copy only (raw is untouched)
    deduped = raw_df.drop_duplicates(subset=["Category", "Resume"]).copy()

    # Apply text cleaning
    deduped["Resume"] = deduped["Resume"].apply(clean_text)

    # Validate 25 categories
    category_count = deduped["Category"].nunique()
    if category_count != 25:
        raise ValueError(f"Expected 25 resume categories, but found {category_count}.")

    return deduped.reset_index(drop=True)


def preprocess_naukri(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess Naukri dataset and map to the unified 6-column schema.

    Mapping:
    - jobtitle -> title
    - company -> company
    - joblocation_address -> location
    - skills -> skills
    - jobdescription -> description
    - experience -> experience
    """
    df = pd.DataFrame()
    df["title"] = raw_df["jobtitle"].fillna("").apply(clean_text)
    df["company"] = raw_df["company"].fillna("").apply(clean_text)
    df["location"] = raw_df["joblocation_address"].fillna("").apply(normalize_location)
    df["skills"] = raw_df["skills"].fillna("").apply(clean_text)
    df["description"] = raw_df["jobdescription"].fillna("").apply(clean_naukri_description)
    df["experience"] = raw_df["experience"].fillna("").apply(clean_text)

    # Drop exact duplicate rows across all 6 columns
    cleaned_df = df.drop_duplicates(subset=JOB_CORPUS_COLUMNS).reset_index(drop=True)
    return cleaned_df


def preprocess_linkedin(
    postings_df: pd.DataFrame,
    agg_skills_df: Optional[pd.DataFrame] = None,
    companies_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Preprocess LinkedIn dataset and map to the unified 6-column schema.

    - Connects job_skills (1-to-1 aggregated per job_id, avoiding row duplication).
    - Fills missing company_name from companies_df where available.
    - Maps columns:
      title -> title
      company_name -> company
      location -> location
      skills -> skills (from job_skills)
      description -> description
      formatted_experience_level -> experience
    """
    df = postings_df.copy()

    # Join aggregated skills (1-to-1 on job_id)
    if agg_skills_df is not None:
        df = df.merge(agg_skills_df, on="job_id", how="left")
    else:
        df["skills"] = ""

    # Recover company name from companies_df if missing
    if companies_df is not None and "company_id" in df.columns:
        comp_lookup = companies_df.drop_duplicates(subset=["company_id"])
        missing_comp = df["company_name"].isnull() & df["company_id"].notnull()
        if missing_comp.any():
            merged_comp = df.loc[missing_comp, ["company_id"]].merge(comp_lookup, on="company_id", how="left")
            df.loc[missing_comp, "company_name"] = merged_comp["name"].values

    # Map to unified schema
    unified = pd.DataFrame()
    unified["title"] = df["title"].fillna("").apply(clean_text)
    unified["company"] = df["company_name"].fillna("").apply(clean_text)
    unified["location"] = df["location"].fillna("").apply(normalize_location)
    unified["skills"] = df["skills"].fillna("").apply(clean_text)
    unified["description"] = df["description"].fillna("").apply(clean_text)
    unified["experience"] = df["formatted_experience_level"].fillna("").apply(clean_text)

    # Drop exact duplicates across all 6 columns
    cleaned_df = unified.drop_duplicates(subset=JOB_CORPUS_COLUMNS).reset_index(drop=True)
    return cleaned_df


def build_unified_job_corpus(
    naukri_cleaned_df: pd.DataFrame,
    linkedin_cleaned_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine cleaned Naukri and LinkedIn records into the final unified job corpus.

    Schema: [title, company, location, skills, description, experience]
    """
    corpus = pd.concat([naukri_cleaned_df[JOB_CORPUS_COLUMNS], linkedin_cleaned_df[JOB_CORPUS_COLUMNS]], ignore_index=True)
    # Deduplicate identical postings across corpora
    deduped_corpus = corpus.drop_duplicates(subset=JOB_CORPUS_COLUMNS).reset_index(drop=True)
    return deduped_corpus
