"""Validation helpers for page and excerpt citations in audit findings."""

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


_PAGE_PATTERN = re.compile(r"\b(?:pages?|p(?:age)?\.?)\s*(\d+)", re.IGNORECASE)
_EXCERPT_PATTERN = re.compile(
    r"(?:exact\s+excerpt|excerpt|quote)\s*:\s*[\"“](.*?)[\"”]",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class CitationResult:
    """The verification result for one finding citation."""

    page_numbers: List[int]
    excerpt: Optional[str]
    verified: bool
    message: str


def _normalise_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _page_numbers(evidence_section: str) -> List[int]:
    return [int(match) for match in _PAGE_PATTERN.findall(evidence_section)]


def _excerpt(evidence_section: str) -> Optional[str]:
    match = _EXCERPT_PATTERN.search(evidence_section)
    if match:
        return match.group(1).strip()

    # Accept a concise citation such as "Page 2: payment is due in 30 days"
    # when the model omits quotation marks despite the output instructions.
    fallback = re.search(
        r"\b(?:pages?|p(?:age)?\.?)\s*\d+\s*[-:]\s*(.+)",
        evidence_section,
        re.IGNORECASE | re.DOTALL,
    )
    if fallback:
        excerpt = fallback.group(1).strip()
        excerpt = re.sub(r"^exact\s+excerpt\s*:\s*", "", excerpt, flags=re.IGNORECASE)
        return excerpt.strip(" \"“”")
    return None


def validate_citation(
    evidence_section: str, pages: Sequence[Dict[str, Any]]
) -> CitationResult:
    """Verify the cited pages and exact excerpt against extracted PDF pages."""

    page_numbers = _page_numbers(evidence_section)
    excerpt = _excerpt(evidence_section)
    page_content = {
        int(page["page_number"]): _normalise_text(str(page.get("content", "")))
        for page in pages
        if "page_number" in page
    }

    if not page_numbers:
        return CitationResult(
            page_numbers=[],
            excerpt=excerpt,
            verified=False,
            message="No page number was found in the evidence citation.",
        )

    missing_pages = [page for page in page_numbers if page not in page_content]
    if missing_pages:
        return CitationResult(
            page_numbers=page_numbers,
            excerpt=excerpt,
            verified=False,
            message=f"Cited page(s) not found in the PDF: {', '.join(map(str, missing_pages))}.",
        )

    if not excerpt:
        return CitationResult(
            page_numbers=page_numbers,
            excerpt=None,
            verified=False,
            message="A page was cited, but no quoted excerpt was provided.",
        )

    normalised_excerpt = _normalise_text(excerpt)
    if not any(
        normalised_excerpt in page_content[page_number]
        for page_number in page_numbers
    ):
        return CitationResult(
            page_numbers=page_numbers,
            excerpt=excerpt,
            verified=False,
            message="The cited excerpt was not found on the cited page(s).",
        )

    return CitationResult(
        page_numbers=page_numbers,
        excerpt=excerpt,
        verified=True,
        message="Page and excerpt verified against the extracted PDF text.",
    )


def validate_report_citations(
    findings: Iterable[Any], pages: Sequence[Dict[str, Any]]
) -> List[CitationResult]:
    """Validate every finding in a report, preserving finding order."""

    return [
        validate_citation(finding.evidence_section, pages) for finding in findings
    ]
