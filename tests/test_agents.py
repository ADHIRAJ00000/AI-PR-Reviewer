"""Tests for the specialist agents and the full graph, using the fake LLM.

Verifies: each specialist writes typed findings + token usage; a failing
specialist is isolated (error recorded, review still completes); the summarizer
produces a final review; and the end-to-end graph populates everything.
"""

from __future__ import annotations

from app.agents.diff_analyzer import diff_analyzer_node
from app.agents.graph import compile_graph
from app.agents.security_auditor import security_auditor_node
from app.agents.state import (
    DiffFindings,
    FileChange,
    SecurityFindings,
)
from app.agents.state import TestSuggestions as TestSuggestionsModel
from app.agents.state import new_state
from app.agents.summarizer import summarizer_node
# Aliased so pytest doesn't try to collect the `test_`-prefixed agent fn as a test.
from app.agents.test_suggester import test_suggester_node as run_test_suggester


def _code_state():
    return new_state(
        owner="octo",
        repo="hello",
        pr_number=1,
        pr_title="Add login",
        diff="diff --git a/auth.py b/auth.py\n+def login(): ...",
        changed_files=[FileChange(filename="auth.py", status="modified", changes=5)],
    )


# --------------------------------------------------------------------------- #
# Individual specialists
# --------------------------------------------------------------------------- #
def test_diff_analyzer_writes_typed_findings_and_usage(fake_llm):
    fake_llm()
    out = diff_analyzer_node(_code_state())
    assert isinstance(out["diff_findings"], DiffFindings)
    assert out["token_usage"]["diff_analyzer"]["total"] == 180
    assert out["token_usage"]["diff_analyzer"]["cost_usd"] > 0


def test_security_auditor_writes_typed_findings(fake_llm):
    fake_llm()
    out = security_auditor_node(_code_state())
    assert isinstance(out["security_findings"], SecurityFindings)


def test_test_suggester_writes_typed_findings(fake_llm):
    fake_llm()
    out = run_test_suggester(_code_state())
    assert isinstance(out["test_suggestions"], TestSuggestionsModel)


def test_specialist_failure_is_isolated(fake_llm):
    """A failing LLM call → error recorded + zeroed usage, no exception."""
    fake_llm(fail_schemas={"SecurityFindings"})
    out = security_auditor_node(_code_state())
    assert "security_findings" not in out
    assert out["errors"] and "security_auditor" in out["errors"][0]
    assert out["token_usage"]["security_auditor"]["total"] == 0


def test_summarizer_produces_review(fake_llm):
    fake_llm(text="## Verdict\n\nApprove.")
    out = summarizer_node(_code_state())
    assert "Approve" in out["final_review"]
    assert out["token_usage"]["summarizer"]["total"] == 180


def test_summarizer_falls_back_when_llm_fails(fake_llm):
    """If the summarizer LLM fails, a deterministic review is still returned."""
    fake_llm(fail_text=True)
    state = _code_state()
    state["security_findings"] = SecurityFindings(summary="none", findings=[])
    out = summarizer_node(state)
    assert out["final_review"]  # non-empty fallback
    assert "summarizer" in out["errors"][0]


# --------------------------------------------------------------------------- #
# End-to-end graph
# --------------------------------------------------------------------------- #
def test_full_graph_populates_all_sections(fake_llm):
    fake_llm()
    result = compile_graph().invoke(_code_state())

    assert isinstance(result["diff_findings"], DiffFindings)
    assert isinstance(result["test_suggestions"], TestSuggestionsModel)
    assert isinstance(result["security_findings"], SecurityFindings)
    assert result["final_review"]
    assert result["errors"] == []
    # Token usage accumulated for every agent that ran (3 specialists + summary).
    assert set(result["token_usage"]) == {
        "diff_analyzer",
        "test_suggester",
        "security_auditor",
        "summarizer",
    }


def test_full_graph_completes_when_one_specialist_fails(fake_llm):
    """One specialist failing must NOT crash the review."""
    fake_llm(fail_schemas={"SecurityFindings"})
    result = compile_graph().invoke(_code_state())

    assert result["security_findings"] is None       # failed
    assert isinstance(result["diff_findings"], DiffFindings)  # others ran
    assert result["final_review"]                    # review still produced
    assert any("security_auditor" in e for e in result["errors"])
