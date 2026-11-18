"""Data loading utilities for SmartHire.

Functions to inspect and read raw datasets (resumes, Naukri, LinkedIn,
and supplementary job datasets) and load processed assets.
"""

from pathlib import Path
from typing import Optional, Union, Iterator
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    RESUME_RAW_PATH,
    NAUKRI_RAW_PATH,
    LINKEDIN_POSTINGS_PATH,
    LINKEDIN_JOB_SKILLS_PATH,
    LINKEDIN_SKILLS_MAP_PATH,
    LINKEDIN_COMPANIES_PATH,
    RESUMES_PROCESSED_CSV_PATH,
    RESUMES_PROCESSED_PATH,
    JOB_CORPUS_PROCESSED_PATH,
    JOB_CORPUS_PROCESSED_CSV_PATH,
)


def load_raw_resumes(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw resume dataset (962 records, 25 categories)."""
    path = file_path or RESUME_RAW_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw resume dataset not found at {path}")
    return pd.read_csv(path)


def load_raw_naukri_jobs(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw Naukri job postings (22,000 records)."""
    path = file_path or NAUKRI_RAW_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw Naukri dataset not found at {path}")
    return pd.read_csv(path)


def load_raw_linkedin_jobs(
    file_path: Optional[Path] = None,
    chunksize: Optional[int] = None,
    usecols: Optional[list] = None,
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """Load raw LinkedIn job postings (123,849 records).

    Supports chunked loading for memory efficiency.
    """
    path = file_path or LINKEDIN_POSTINGS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw LinkedIn dataset not found at {path}")
    if chunksize:
        return pd.read_csv(path, chunksize=chunksize, usecols=usecols, low_memory=False)
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def load_linkedin_skills_mapping(
    job_skills_path: Optional[Path] = None,
    skills_map_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load LinkedIn job skills and map skill abbreviations to standard skill names.

    Aggregates skills per job_id into a single comma-separated string,
    guaranteeing a unique 1-to-1 relationship per job_id.

    Returns:
        pd.DataFrame with columns: ['job_id', 'skills']
    """
    js_path = job_skills_path or LINKEDIN_JOB_SKILLS_PATH
    sm_path = skills_map_path or LINKEDIN_SKILLS_MAP_PATH

    if not js_path.exists() or not sm_path.exists():
        raise FileNotFoundError("LinkedIn job_skills or skills mapping file not found.")

    job_skills_df = pd.read_csv(js_path)
    skills_map_df = pd.read_csv(sm_path)

    # Join on skill_abr
    merged = job_skills_df.merge(skills_map_df, on="skill_abr", how="left")
    merged["skill_resolved"] = merged["skill_name"].fillna(merged["skill_abr"])

    # Aggregate to one row per job_id
    agg_skills = (
        merged.groupby("job_id")["skill_resolved"]
        .apply(lambda s: ", ".join(sorted(set(s.dropna()))))
        .reset_index(name="skills")
    )
    return agg_skills


def load_linkedin_companies(
    companies_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load LinkedIn companies mapping table (company_id -> company name)."""
    c_path = companies_path or LINKEDIN_COMPANIES_PATH
    if not c_path.exists():
        raise FileNotFoundError(f"LinkedIn companies file not found at {c_path}")
    return pd.read_csv(c_path, usecols=["company_id", "name"])


def load_cleaned_resumes(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the processed, deduplicated resumes dataset."""
    path = file_path or RESUMES_PROCESSED_CSV_PATH
    if not path.exists():
        parquet_path = RESUMES_PROCESSED_PATH
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        raise FileNotFoundError(f"Cleaned resumes dataset not found at {path}")
    return pd.read_csv(path)


def load_merged_job_corpus(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the processed unified job corpus."""
    path = file_path or JOB_CORPUS_PROCESSED_PATH
    if not path.exists():
        csv_path = JOB_CORPUS_PROCESSED_CSV_PATH
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(f"Merged job corpus not found at {path}")
    return pd.read_parquet(path)
