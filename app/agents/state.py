"""Shared LangGraph state + the structured models each agent produces.

The state is a `TypedDict` (LangGraph's native channel container). Fields that
several parallel nodes may write in the same superstep — `errors` and
`token_usage` — are annotated with **reducers** so concurrent updates merge
instead of clobbering each other. Every other field is written by exactly one
node, so last-write semantics are fine.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


def _to_str(value: Any) -> str:
    """Coerce a value some models emit (list/None/number) into a string.

    Open models (via Groq's strict tool validation) sometimes return an array or
    null where we expect a string; normalise instead of failing the whole agent.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return str(value)

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
class FileChange(BaseModel):
    """A changed file as the review pipeline sees it."""

    filename: str
    status: str  # added | modified | removed | renamed | ...
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    patch: str | None = None


# --------------------------------------------------------------------------- #
# Diff Analyzer output
# --------------------------------------------------------------------------- #
class FileDiffFinding(BaseModel):
    """Per-file analysis from the diff analyzer."""

    file: str = ""
    summary: str = Field(default="", description="One-line summary of what changed.")
    intent: str = Field(default="", description="The apparent purpose of the change.")
    risks: list[str] = Field(
        default_factory=list,
        description="Breaking changes, logic errors, perf concerns, removed error handling.",
    )

    @field_validator("file", "summary", "intent", mode="before")
    @classmethod
    def _str(cls, v: Any) -> str:
        return _to_str(v)

    @field_validator("risks", mode="before")
    @classmethod
    def _risks(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v else []
        return [str(x) for x in v]


class DiffFindings(BaseModel):
    """Aggregate output of the diff analyzer."""

    files: list[FileDiffFinding] = Field(default_factory=list)
    overall_summary: str = ""

    @field_validator("overall_summary", mode="before")
    @classmethod
    def _summary(cls, v: Any) -> str:
        return _to_str(v)


# --------------------------------------------------------------------------- #
# Test Suggester output
# --------------------------------------------------------------------------- #
class TestSuggestion(BaseModel):
    """A single proposed test case.

    `input` / `expected_output` accept str | list | None so strict providers
    (Groq) don't reject an array/null, then a validator normalises to a string.
    """

    name: str = ""
    verifies: str = Field(default="", description="What behaviour this test checks.")
    input: str | list | None = Field(default="", description="The input / setup.")
    expected_output: str | list | None = Field(default="", description="Expected result.")
    edge_case: str = Field(
        default="", description="empty | null | boundary | error-path | concurrency | ..."
    )

    @field_validator("name", "verifies", "input", "expected_output", "edge_case", mode="before")
    @classmethod
    def _normalise(cls, v: Any) -> str:
        return _to_str(v)


class TestSuggestions(BaseModel):
    """Aggregate output of the test suggester."""

    suggestions: list[TestSuggestion] = Field(default_factory=list)
    summary: str = ""

    @field_validator("summary", mode="before")
    @classmethod
    def _summary(cls, v: Any) -> str:
        return _to_str(v)


# --------------------------------------------------------------------------- #
# Security Auditor output
# --------------------------------------------------------------------------- #
Severity = Literal["low", "medium", "high", "critical"]
_VALID_SEVERITIES = {"low", "medium", "high", "critical"}


class SecurityFinding(BaseModel):
    """A single security issue.

    Types are permissive (str severity, int|str|None line) so strict providers
    accept model drift; validators normalise them back to expected shapes.
    """

    severity: str = "medium"
    category: str = Field(default="", description="e.g. sql-injection, hardcoded-secret.")
    file: str = ""
    line: int | str | None = None
    description: str = ""
    recommendation: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def _sev(cls, v: Any) -> str:
        s = _to_str(v).lower().strip()
        return s if s in _VALID_SEVERITIES else "medium"

    @field_validator("category", "file", "description", "recommendation", mode="before")
    @classmethod
    def _str(cls, v: Any) -> str:
        return _to_str(v)

    @field_validator("line", mode="before")
    @classmethod
    def _line(cls, v: Any) -> int | None:
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None


class SecurityFindings(BaseModel):
    """Aggregate output of the security auditor."""

    findings: list[SecurityFinding] = Field(default_factory=list)
    summary: str = ""

    @field_validator("summary", mode="before")
    @classmethod
    def _summary(cls, v: Any) -> str:
        return _to_str(v)


# --------------------------------------------------------------------------- #
# Reducers for concurrently-written fields
# --------------------------------------------------------------------------- #
def merge_token_usage(
    left: dict[str, dict], right: dict[str, dict]
) -> dict[str, dict]:
    """Merge per-agent token/cost dicts (each agent owns a distinct key)."""
    merged = dict(left)
    merged.update(right)
    return merged


# --------------------------------------------------------------------------- #
# The shared graph state
# --------------------------------------------------------------------------- #
class PRReviewState(TypedDict, total=False):
    """State threaded through every node of the review graph."""

    # ---- inputs ----
    owner: str
    repo: str
    pr_number: int
    pr_title: str
    pr_body: str
    diff: str
    changed_files: list[FileChange]

    # ---- working memory (each specialist writes its own field) ----
    diff_findings: DiffFindings | None
    test_suggestions: TestSuggestions | None
    security_findings: SecurityFindings | None

    # ---- control ----
    agents_to_run: list[str]
    errors: Annotated[list[str], operator.add]

    # ---- output ----
    final_review: str | None

    # ---- observability ----
    trace_id: str | None
    token_usage: Annotated[dict[str, dict], merge_token_usage]


# Canonical node names (single source of truth).
COORDINATOR = "coordinator"
DIFF_ANALYZER = "diff_analyzer"
TEST_SUGGESTER = "test_suggester"
SECURITY_AUDITOR = "security_auditor"
SUMMARIZER = "summarizer"
ALL_SPECIALISTS = (DIFF_ANALYZER, TEST_SUGGESTER, SECURITY_AUDITOR)


def new_state(
    *,
    owner: str,
    repo: str,
    pr_number: int,
    pr_title: str = "",
    pr_body: str = "",
    diff: str = "",
    changed_files: list[FileChange] | None = None,
) -> PRReviewState:
    """Construct a fully-initialised state with empty reducer channels."""
    return PRReviewState(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        pr_title=pr_title,
        pr_body=pr_body,
        diff=diff,
        changed_files=changed_files or [],
        diff_findings=None,
        test_suggestions=None,
        security_findings=None,
        agents_to_run=[],
        errors=[],
        final_review=None,
        trace_id=None,
        token_usage={},
    )
