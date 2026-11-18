"""Resume parsing utilities for SmartHire.

Extracts plain text from PDF, DOCX, and TXT files or stream objects.
Extracts candidate metadata and candidate skills using deterministic rule-based matching.
"""

from pathlib import Path
from typing import Union, BinaryIO, Set
import re


def extract_text_from_pdf(file_input: Union[str, Path, BinaryIO]) -> str:
    """Extract raw text from a PDF file path or file-like buffer using pypdf or pdfplumber."""
    text_content = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_input)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_content.append(extracted)
    except Exception as exc:
        # Fallback or alternate handling
        try:
            import pdfplumber
            with pdfplumber.open(file_input) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_content.append(extracted)
        except Exception:
            raise RuntimeError(f"Failed to extract text from PDF: {exc}")

    return "\n".join(text_content).strip()


def extract_text_from_docx(file_input: Union[str, Path, BinaryIO]) -> str:
    """Extract raw text from a DOCX file path or file-like buffer."""
    try:
        import docx
        doc = docx.Document(file_input)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paragraphs).strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to extract text from DOCX: {exc}")


def extract_resume_text(file_input: Union[str, Path, BinaryIO], filename: str = "") -> str:
    """Unified entry point to extract text from PDF, DOCX, or plain text resumes.

    Args:
        file_input: File path or byte stream.
        filename: Optional filename to infer extension if file_input is a stream.

    Returns:
        Extracted plain text string.
    """
    path_str = str(file_input).lower() if isinstance(file_input, (str, Path)) else filename.lower()

    if path_str.endswith(".pdf"):
        return extract_text_from_pdf(file_input)
    elif path_str.endswith(".docx"):
        return extract_text_from_docx(file_input)
    elif path_str.endswith(".txt"):
        if hasattr(file_input, "read"):
            content = file_input.read()
            return content.decode("utf-8", errors="ignore") if isinstance(content, bytes) else str(content)
        with open(file_input, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file format for resume parsing: {path_str}")


def extract_skills_rule_based(text: str, skill_vocabulary: Set[str]) -> Set[str]:
    """Extract matched skills from raw resume text against a known skill vocabulary.

    Uses word boundaries and case-insensitive matching.
    """
    found_skills = set()
    cleaned = text.lower()
    for skill in skill_vocabulary:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, cleaned):
            found_skills.add(skill)
    return found_skills
