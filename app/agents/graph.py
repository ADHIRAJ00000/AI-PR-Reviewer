"""Builds and compiles the LangGraph review graph.

Flow:

    START → coordinator ──(conditional fan-out)──▶ diff_analyzer   ─┐
                                                 ├▶ test_suggester ─┤▶ summarizer → END
                                                 └▶ security_auditor┘

The coordinator sets `agents_to_run`; a conditional edge fans out to only those
specialists (in parallel); the summarizer joins whatever ran.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from app.agents.coordinator import coordinator_node, decide_agents  # noqa: F401 (re-export)
from app.agents.diff_analyzer import diff_analyzer_node
from app.agents.security_auditor import security_auditor_node
from app.agents.state import (
    COORDINATOR,
    DIFF_ANALYZER,
    SECURITY_AUDITOR,
    SUMMARIZER,
    TEST_SUGGESTER,
    PRReviewState,
)
from app.agents.summarizer import summarizer_node
from app.agents.test_suggester import test_suggester_node

logger = logging.getLogger("app.agents.graph")


def route_specialists(state: PRReviewState) -> list[str]:
    """Conditional edge: fan out to the selected specialists (parallel)."""
    agents = state.get("agents_to_run") or [DIFF_ANALYZER]
    logger.info("routing to specialists", extra={"agents": agents})
    return agents


def build_graph() -> StateGraph:
    """Assemble the (uncompiled) review StateGraph."""
    graph = StateGraph(PRReviewState)

    graph.add_node(COORDINATOR, coordinator_node)
    graph.add_node(DIFF_ANALYZER, diff_analyzer_node)
    graph.add_node(TEST_SUGGESTER, test_suggester_node)
    graph.add_node(SECURITY_AUDITOR, security_auditor_node)
    graph.add_node(SUMMARIZER, summarizer_node)

    graph.add_edge(START, COORDINATOR)

    graph.add_conditional_edges(
        COORDINATOR,
        route_specialists,
        {
            DIFF_ANALYZER: DIFF_ANALYZER,
            TEST_SUGGESTER: TEST_SUGGESTER,
            SECURITY_AUDITOR: SECURITY_AUDITOR,
        },
    )

    graph.add_edge(DIFF_ANALYZER, SUMMARIZER)
    graph.add_edge(TEST_SUGGESTER, SUMMARIZER)
    graph.add_edge(SECURITY_AUDITOR, SUMMARIZER)

    graph.add_edge(SUMMARIZER, END)
    return graph


def compile_graph():
    """Build and compile the graph, ready to invoke."""
    return build_graph().compile()
