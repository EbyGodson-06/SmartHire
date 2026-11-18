"""Streamlit UI integration tests using streamlit.testing.v1.AppTest.

Tests:
- Application startup without exceptions.
- Initial guidance display when no resume is loaded.
- Form validation warning when clicking Analyze without input.
- Switching between File Upload and Direct Text Paste modes.
- End-to-end analysis rendering (KPI banner, 4 tabs, dataframes, metrics).
- Restrictive filtering displaying user-friendly warning message.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def test_app_startup_and_initial_state():
    """Verify application boots cleanly and displays default instructions."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()

    assert not at.exception
    assert len(at.title) >= 1
    # Check that initial guidance banner is present
    assert any("upload a resume" in info.value.lower() for info in at.info)


def test_app_empty_input_warning():
    """Verify clicking Analyze with no input displays validation warning."""
    at = AppTest.from_file(APP_PATH, default_timeout=45)
    at.run()

    # Click the Analyze button
    at.sidebar.button[0].click()
    at.run()

    assert not at.exception
    assert any("upload a resume document or paste" in w.value.lower() for w in at.warning)


def test_app_direct_text_analysis_flow():
    """Verify direct text input runs complete pipeline and renders all 4 tabs."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    # Switch radio to Direct Text Paste
    at.sidebar.radio[0].set_value("Direct Text Paste")
    at.run()

    # Input sample resume
    sample_cv = (
        "Senior Data Scientist with experience in Machine Learning, Deep Learning, "
        "Python, Scikit-Learn, PyTorch, Pandas, and SQL."
    )
    at.sidebar.text_area[0].set_value(sample_cv)
    at.run()

    # Click Analyze
    at.sidebar.button[0].click()
    at.run()

    assert not at.exception

    # Check that 4 tabs are rendered
    assert len(at.tabs) == 4

    # Check top KPI banner metrics
    metric_labels = [m.label for m in at.metric]
    assert "Predicted Domain" in metric_labels
    assert "Matching Jobs Found" in metric_labels
    assert "Top Cosine Similarity" in metric_labels
    assert "Extracted Skills" in metric_labels

    # Check Tab 3 metrics
    assert "Predicted Domain Category" in metric_labels
    assert "Dominant NMF Topic" in metric_labels
    assert "Dominant Document-Topic Loading (W)" in metric_labels

    # Check that dataframes are rendered (Summary table, margins, NMF loadings)
    assert len(at.dataframe) >= 2


def test_app_restrictive_filter_no_results_warning():
    """Verify that impossible filters display a clear actionable warning in Tab 1."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    # Switch radio to Direct Text Paste
    at.sidebar.radio[0].set_value("Direct Text Paste")
    at.run()

    sample_cv = "Python Developer with Django and PostgreSQL"
    at.sidebar.text_area[0].set_value(sample_cv)

    # Set impossible min similarity slider (e.g. 0.50 for a short query) or impossible filter
    at.sidebar.slider[1].set_value(0.50)  # min_similarity
    at.run()

    at.sidebar.button[0].click()
    at.run()

    assert not at.exception
    # When 0 recommendations match, a warning message is shown in Tab 1
    # Check that app handles empty recommendations gracefully
    metric_dict = {m.label: m.value for m in at.metric}
    if metric_dict.get("Matching Jobs Found") == "0 roles":
        assert any("no job recommendations matched" in w.value.lower() for w in at.warning)
