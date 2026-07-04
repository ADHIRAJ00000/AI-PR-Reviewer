from __future__ import annotations

import logging

from app.agents.context import build_specialist_input
from app.agents.state import DIFF_ANALYZER, DiffFindings, PRReviewState
from app.llm import call_structured
from app.observability.cost import empty_usage

logger = logging.getLogger("app.agents.diff_analyzer")

SYSTEM_PROMPT = (
    "You are a senior code reviewer. You are given a unified git diff. For each "
    "changed file, produce: (1) a one-line summary of what changed, (2) the "
    "apparent intent, (3) any risky changes (breaking changes, logic errors, "
    "performance concerns, removed error handling). Be specific and cite file + "
    "approximate line. Do NOT comment on style nitpicks. Output ONLY the "
    "structured schema provided."
)


def diff_analyzer_node(state: PRReviewState) -> dict:
    try:
        findings, usage = call_structured(
            SYSTEM_PROMPT, build_specialist_input(state), DiffFindings
        )
        logger.info(
            "diff_analyzer produced findings", extra={"files": len(findings.files)}
        )
        return {"diff_findings": findings, "token_usage": {DIFF_ANALYZER: usage}}
    except Exception as exc:  # noqa: BLE001 - one agent must never crash the review
        logger.exception("diff_analyzer failed")
        return {
            "errors": [f"{DIFF_ANALYZER}: {exc}"],
            "token_usage": {DIFF_ANALYZER: empty_usage()},
        }
