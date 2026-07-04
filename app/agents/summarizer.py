"""Summarizer node: writes the single, polished Markdown review comment.

If the LLM call fails, it still returns a review — a deterministic fallback
assembled from whatever structured findings exist — so the pipeline always
produces something to post.
"""

from __future__ import annotations

import logging

from app.agents.context import build_summary_input
from app.agents.state import SUMMARIZER, PRReviewState
from app.guardrails.output_guard import guard_output
from app.llm import call_text
from app.observability.cost import empty_usage

logger = logging.getLogger("app.agents.summarizer")

SYSTEM_PROMPT = (
    "You are the lead reviewer writing the final PR review comment. You are given "
    "structured findings from the diff analyzer, test suggester, and security "
    "auditor. Write ONE clean Markdown review comment that a busy engineer will "
    "read: start with a 2-line verdict (approve / request changes), then a "
    "prioritized list (security first, then correctness, then tests), then a "
    "short 'nice work' note if applicable. Be concise, specific, and kind. No "
    "filler. Never expose raw secrets — reference them as [REDACTED]."
)


def _fallback_review(state: PRReviewState) -> str:
    """Deterministic review assembled without the LLM (used on failure)."""
    lines = ["## Automated PR Review (degraded mode)", ""]
    security = state.get("security_findings")
    if security and security.findings:
        lines.append("### Security")
        for f in security.findings:
            loc = f"{f.file}:{f.line}" if f.line else f.file
            lines.append(f"- **[{f.severity}]** {loc} — {f.description}")
        lines.append("")
    diff = state.get("diff_findings")
    if diff and diff.files:
        lines.append("### Changes")
        for fd in diff.files:
            lines.append(f"- `{fd.file}` — {fd.summary}")
        lines.append("")
    tests = state.get("test_suggestions")
    if tests and tests.suggestions:
        lines.append("### Suggested tests")
        for t in tests.suggestions:
            lines.append(f"- {t.name}: {t.verifies}")
        lines.append("")
    errors = state.get("errors") or []
    if errors:
        lines.append("> Note: some sections were skipped due to internal errors.")
    if len(lines) <= 2:
        lines.append("_No findings were produced for this PR._")
    return "\n".join(lines)


def summarizer_node(state: PRReviewState) -> dict:
    try:
        review, usage = call_text(SYSTEM_PROMPT, build_summary_input(state))
        review = guard_output(review)  # redact secrets, strip leaks, cap length
        logger.info("summarizer produced final review", extra={"chars": len(review)})
        return {"final_review": review, "token_usage": {SUMMARIZER: usage}}
    except Exception as exc:  # noqa: BLE001
        logger.exception("summarizer failed; using deterministic fallback")
        return {
            "final_review": guard_output(_fallback_review(state)),
            "errors": [f"{SUMMARIZER}: {exc}"],
            "token_usage": {SUMMARIZER: empty_usage()},
        }
