"""Markdown report generation for structured audit results."""

from typing import Iterable, Optional

from models.schemas import AuditReport
from services.citation_validator import CitationResult


def _citation_status(result: CitationResult) -> str:
    return "Verified" if result.verified else f"UNVERIFIED — {result.message}"


def generate_markdown_report(
    report: AuditReport,
    citation_results: Optional[Iterable[CitationResult]] = None,
) -> str:
    """Compile an audit report and citation status into downloadable Markdown."""

    results = list(citation_results or [])
    lines = [
        "# VaultDocs AI Contract Audit Report",
        "",
        "> This is an automated triage report, not formal legal advice.",
        "",
        "## Summary",
        "",
        report.summary,
        "",
        "## Findings",
        "",
    ]

    if not report.findings:
        lines.append("No risks were identified for the selected audit focus.")
        lines.append("")

    for index, finding in enumerate(report.findings):
        citation = results[index] if index < len(results) else None
        status = _citation_status(citation) if citation else "UNVERIFIED — Not checked."
        lines.extend(
            [
                f"### {index + 1}. {finding.category} ({finding.priority})",
                "",
                f"**Finding:** {finding.finding}",
                "",
                f"**Evidence:** {finding.evidence_section}",
                "",
                f"**Citation status:** {status}",
                "",
                f"**Why it matters:** {finding.reason}",
                "",
                f"**Question for the freelancer:** {finding.action_question}",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "Generated locally by VaultDocs AI.",
            "",
        ]
    )
    return "\n".join(lines)
