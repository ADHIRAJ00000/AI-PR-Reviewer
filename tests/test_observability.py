"""Tests for stats aggregation, the /stats endpoint, and tracing no-op."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.observability.stats import StatsCollector, stats
from app.observability.tracing import build_trace, flush
from app.review import run_review
from tests.test_review import FakeGitHub


# --------------------------------------------------------------------------- #
# Stats math
# --------------------------------------------------------------------------- #
def test_stats_averages():
    c = StatsCollector()
    c.record(tokens=1000, cost_usd=0.02, latency_s=3.0)
    c.record(tokens=3000, cost_usd=0.06, latency_s=5.0)
    snap = c.snapshot()
    assert snap["prs_reviewed"] == 2
    assert snap["total_tokens"] == 4000
    assert snap["avg_tokens_per_pr"] == 2000.0
    assert snap["avg_cost_per_pr_usd"] == 0.04
    assert snap["avg_latency_seconds"] == 4.0


def test_stats_empty_snapshot():
    c = StatsCollector()
    snap = c.snapshot()
    assert snap["prs_reviewed"] == 0
    assert snap["avg_cost_per_pr_usd"] == 0.0


# --------------------------------------------------------------------------- #
# Tracing is a no-op when Langfuse keys are absent
# --------------------------------------------------------------------------- #
def test_build_trace_disabled_has_no_callbacks():
    config, handler = build_trace("pr-review x/y#1", metadata={"pr_number": 1})
    assert handler is None
    assert "callbacks" not in config
    assert config["run_name"] == "pr-review x/y#1"
    assert config["metadata"] == {"pr_number": 1}
    flush(handler)  # must not raise


# --------------------------------------------------------------------------- #
# /stats endpoint
# --------------------------------------------------------------------------- #
def test_stats_endpoint_reflects_recorded_reviews():
    stats.reset()
    stats.record(tokens=5000, cost_usd=0.05, latency_s=4.0)
    with TestClient(app) as client:
        resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["prs_reviewed"] == 1
    assert body["total_tokens"] == 5000
    assert body["avg_cost_per_pr_usd"] == 0.05


# --------------------------------------------------------------------------- #
# The runner records stats end-to-end
# --------------------------------------------------------------------------- #
async def test_run_review_records_stats(fake_llm):
    fake_llm()
    stats.reset()
    await run_review("octo", "hello", 5, client=FakeGitHub())
    snap = stats.snapshot()
    assert snap["prs_reviewed"] == 1
    assert snap["total_tokens"] > 0
    assert snap["avg_latency_seconds"] >= 0.0
