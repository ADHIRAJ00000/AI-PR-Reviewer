"""Tests for the graph structure: compilation, routing, parallel fan-out/in.

End-to-end behaviour of the nodes themselves lives in test_agents.py; here we
use the fake LLM only to exercise routing (which specialists actually run).
"""

from __future__ import annotations

from app.agents.graph import build_graph, compile_graph, decide_agents
from app.agents.state import (
    DIFF_ANALYZER,
    SECURITY_AUDITOR,
    TEST_SUGGESTER,
    FileChange,
    new_state,
)


def _file(name: str) -> FileChange:
    return FileChange(filename=name, status="modified", additions=1, changes=1)


def test_graph_compiles():
    assert compile_graph() is not None


def test_build_graph_returns_stategraph():
    assert build_graph() is not None


def test_decide_agents_runs_all_for_code():
    agents = decide_agents([_file("app/main.py")])
    assert set(agents) == {DIFF_ANALYZER, TEST_SUGGESTER, SECURITY_AUDITOR}


def test_decide_agents_skips_specialists_for_docs_only():
    agents = decide_agents([_file("README.md"), _file("docs/guide.rst")])
    assert agents == [DIFF_ANALYZER]  # security + tests skipped


def test_code_pr_runs_all_specialists(fake_llm):
    fake_llm()
    result = compile_graph().invoke(
        new_state(owner="o", repo="r", pr_number=1, changed_files=[_file("svc.py")])
    )
    assert result["diff_findings"] is not None
    assert result["test_suggestions"] is not None
    assert result["security_findings"] is not None
    assert result["final_review"] is not None


def test_docs_only_pr_skips_security_and_tests(fake_llm):
    fake_llm()
    result = compile_graph().invoke(
        new_state(owner="o", repo="r", pr_number=2, changed_files=[_file("README.md")])
    )
    assert result["agents_to_run"] == [DIFF_ANALYZER]
    assert result["diff_findings"] is not None      # ran
    assert result["security_findings"] is None       # skipped
    assert result["test_suggestions"] is None        # skipped
    assert result["final_review"] is not None        # still completes


def test_summarizer_joins_once_on_parallel_fanin(fake_llm):
    """3 parallel specialists → summarizer runs once (single usage key, no dup)."""
    fake_llm()
    result = compile_graph().invoke(
        new_state(owner="o", repo="r", pr_number=3, changed_files=[_file("a.py")])
    )
    assert "summarizer" in result["token_usage"]
    assert result["errors"] == []
