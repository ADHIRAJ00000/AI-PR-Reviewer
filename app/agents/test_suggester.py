"""Test Suggester node: proposes missing test cases for changed code."""

from __future__ import annotations

import logging

from app.agents.context import build_specialist_input
from app.agents.state import TEST_SUGGESTER, PRReviewState, TestSuggestions
from app.llm import call_structured
from app.observability.cost import empty_usage

logger = logging.getLogger("app.agents.test_suggester")

SYSTEM_PROMPT = (
    "You are a test engineer. Given the code diff, identify functions/branches "
    "that lack test coverage. Propose concrete test cases: name, what it "
    "verifies, the input, the expected output, and the edge case it targets "
    "(empty input, null, boundary, error path, concurrency). Prioritize the "
    "highest-risk untested paths. Output ONLY the structured schema."
)


def test_suggester_node(state: PRReviewState) -> dict:
    try:
        suggestions, usage = call_structured(
            SYSTEM_PROMPT, build_specialist_input(state), TestSuggestions
        )
        logger.info(
            "test_suggester produced suggestions",
            extra={"count": len(suggestions.suggestions)},
        )
        return {
            "test_suggestions": suggestions,
            "token_usage": {TEST_SUGGESTER: usage},
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("test_suggester failed")
        return {
            "errors": [f"{TEST_SUGGESTER}: {exc}"],
            "token_usage": {TEST_SUGGESTER: empty_usage()},
        }
