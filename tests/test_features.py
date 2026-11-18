"""Unit tests for text processing and feature calculation."""

import pytest
from src.data.preprocess import clean_text
from src.features.match_features import compute_skill_overlap


def test_clean_text_basic():
    raw_sample = "Experienced Python & Machine Learning Engineer! Email: dev@example.com, Visit: https://github.com"
    cleaned = clean_text(raw_sample)
    assert "https" not in cleaned
    assert "dev@example.com" not in cleaned
    assert "python" in cleaned
    assert "machine learning engineer" in cleaned


def test_compute_skill_overlap():
    resume_skills = {"python", "machine learning", "pandas", "git"}
    job_skills = {"python", "sql", "machine learning", "docker"}

    overlap_info = compute_skill_overlap(resume_skills, job_skills)

    assert overlap_info["overlap_count"] == 2
    assert "python" in overlap_info["common_skills"]
    assert "machine learning" in overlap_info["common_skills"]
    assert "sql" in overlap_info["missing_skills"]
    assert "docker" in overlap_info["missing_skills"]
    assert overlap_info["overlap_ratio"] == 0.5  # 2 out of 4


def test_compute_skill_overlap_empty_job_skills():
    resume_skills = {"python"}
    job_skills = set()

    overlap_info = compute_skill_overlap(resume_skills, job_skills)
    assert overlap_info["overlap_count"] == 0
    assert overlap_info["overlap_ratio"] == 0.0
