"""Tests for the eval harness: scoring math, fixture parsing, offline run."""

from __future__ import annotations

import json

from app.agents.graph import compile_graph
from app.agents.state import SecurityFinding
from evals.run_evals import (
    FIXTURES_DIR,
    GOLDEN_PATH,
    build_scorecard,
    format_scorecard,
    parse_changed_files,
    run_fixture,
    score_fixture,
)
from evals.scoring import count_false_positives, finding_matches, security_recall


def _finding(category, file, severity="high", desc="issue"):
    return SecurityFinding(
        severity=severity, category=category, file=file,
        description=desc, recommendation="fix it",
    )


# --------------------------------------------------------------------------- #
# Matching + recall
# --------------------------------------------------------------------------- #
def test_finding_matches_on_keyword_and_file():
    f = _finding("SQL Injection", "app/db.py", desc="string-built query")
    assert finding_matches(f, {"category_keywords": ["sql", "injection"], "file": "db.py"})


def test_finding_does_not_match_wrong_keyword():
    f = _finding("Style", "app/db.py")
    assert not finding_matches(f, {"category_keywords": ["sql"], "file": "db.py"})


def test_finding_does_not_match_wrong_file():
    f = _finding("SQL Injection", "app/other.py")
    assert not finding_matches(f, {"category_keywords": ["sql"], "file": "db.py"})


def test_security_recall_counts_caught():
    findings = [_finding("SQL Injection", "db.py", desc="sql injection risk")]
    expected = [
        {"category_keywords": ["sql", "injection"], "file": "db.py"},
        {"category_keywords": ["xss"], "file": "web.py"},
    ]
    assert security_recall(findings, expected) == (1, 2)


def test_count_false_positives_respects_severity():
    findings = [
        _finding("x", "a.py", severity="high"),
        _finding("y", "a.py", severity="low"),
    ]
    assert count_false_positives(findings, min_severity="medium") == 1


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_build_scorecard_aggregates():
    rows = [
        {"name": "a", "kind": "insecure", "caught": 1, "total": 1, "false_positives": 0, "judge_score": 5},
        {"name": "b", "kind": "insecure", "caught": 0, "total": 1, "false_positives": 0, "judge_score": 3},
        {"name": "c", "kind": "clean", "caught": 0, "total": 0, "false_positives": 1, "judge_score": 4},
        {"name": "d", "kind": "clean", "caught": 0, "total": 0, "false_positives": 0, "judge_score": 5},
    ]
    agg = build_scorecard(rows)
    assert agg["security_recall_pct"] == 50.0        # 1 of 2
    assert agg["false_positive_findings"] == 1
    assert agg["false_positive_rate_pct"] == 50.0    # 1 of 2 clean flagged
    assert agg["judge_faithfulness_avg"] == 4.25


# --------------------------------------------------------------------------- #
# Fixtures + golden set
# --------------------------------------------------------------------------- #
def test_parse_changed_files():
    diff = (FIXTURES_DIR / "sql_injection.diff").read_text()
    files = parse_changed_files(diff)
    assert [f.filename for f in files] == ["app/db.py"]


def test_every_golden_fixture_file_exists():
    golden = json.loads(GOLDEN_PATH.read_text())
    assert len(golden) >= 8
    for name in golden:
        assert (FIXTURES_DIR / name).exists(), f"missing fixture: {name}"


# --------------------------------------------------------------------------- #
# The harness runs end-to-end offline (with the fake LLM)
# --------------------------------------------------------------------------- #
def test_harness_runs_offline(fake_llm):
    fake_llm()
    golden = json.loads(GOLDEN_PATH.read_text())
    graph = compile_graph()
    rows = []
    for name, meta in list(golden.items())[:3]:
        diff = (FIXTURES_DIR / name).read_text()
        state = run_fixture(graph, name, diff)
        rows.append(score_fixture(name, meta, state, use_judge=True))
    agg = build_scorecard(rows)
    out = format_scorecard(agg, rows)
    assert "SCORECARD" in out
    assert "Security recall" in out
