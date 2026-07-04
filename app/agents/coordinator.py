"""Coordinator node: decides which specialists to run for this PR.

In the full pipeline the coordinator also fetches PR data (Phase 6 wires that
via the webhook). Here it operates on the changed files already in state and
sets `agents_to_run`, which drives the conditional fan-out in the graph.
"""

from __future__ import annotations

import logging

from app.agents.state import (
    DIFF_ANALYZER,
    SECURITY_AUDITOR,
    TEST_SUGGESTER,
    FileChange,
    PRReviewState,
)

logger = logging.getLogger("app.agents.coordinator")

# Extensions treated as prose/docs rather than code.
_DOC_EXTENSIONS = (".md", ".markdown", ".rst", ".txt", ".adoc")


def _is_code_file(f: FileChange) -> bool:
    """True unless the file is clearly documentation/prose."""
    return not f.filename.lower().endswith(_DOC_EXTENSIONS)


def decide_agents(changed_files: list[FileChange]) -> list[str]:
    """Choose specialists: diff analyzer always; tests + security only for code."""
    agents = [DIFF_ANALYZER]
    if any(_is_code_file(f) for f in changed_files):
        agents.append(TEST_SUGGESTER)
        agents.append(SECURITY_AUDITOR)
    return agents


def coordinator_node(state: PRReviewState) -> dict:
    agents = decide_agents(state.get("changed_files", []))
    logger.info(
        "coordinator selected specialists",
        extra={"pr_number": state.get("pr_number"), "agents_to_run": agents},
    )
    return {"agents_to_run": agents}
