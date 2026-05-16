from __future__ import annotations

import re


EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}")
LINKEDIN_PATTERN = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+/?", re.IGNORECASE)
GITHUB_PATTERN = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_-]+/?", re.IGNORECASE)


def extract_personal_info(resume: str) -> dict[str, str | None]:
    lines = [line.strip() for line in resume.splitlines() if line.strip()]
    first_line = lines[0] if lines else ""

    email = _first_match(EMAIL_PATTERN, resume)
    phone = _first_match(PHONE_PATTERN, resume)
    linkedin = _first_match(LINKEDIN_PATTERN, resume)
    github = _first_match(GITHUB_PATTERN, resume)

    return {
        "name": _clean_name(first_line),
        "email": email,
        "phone": phone,
        "linkedin": linkedin,
        "github": github,
    }


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0) if match else None


def _clean_name(text: str) -> str | None:
    if not text:
        return None
    if EMAIL_PATTERN.search(text) or len(text.split()) > 6:
        return None
    return text

