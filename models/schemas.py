"""Pydantic models for structured contract-audit output."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class RiskFinding(BaseModel):
    """A single contract risk finding returned by the local model."""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1)
    priority: str = Field(min_length=1)
    finding: str = Field(min_length=1)
    evidence_section: str = Field(
        min_length=1,
        description='Citation in the form "Page N - exact excerpt: ..."',
    )
    reason: str = Field(min_length=1)
    action_question: str = Field(min_length=1)


class AuditReport(BaseModel):
    """The complete structured audit report."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    findings: List[RiskFinding] = Field(default_factory=list)
