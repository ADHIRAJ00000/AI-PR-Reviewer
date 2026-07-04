"""In-process aggregate stats for the /stats endpoint.

A single-process, thread-safe accumulator. (For multi-instance deployments this
would move to Redis/DB — kept in-memory here to stay dependency-light.)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class _Totals:
    prs_reviewed: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_s: float = 0.0


class StatsCollector:
    def __init__(self) -> None:
        self._t = _Totals()
        self._lock = threading.Lock()

    def record(self, *, tokens: int, cost_usd: float, latency_s: float) -> None:
        with self._lock:
            self._t.prs_reviewed += 1
            self._t.total_tokens += tokens
            self._t.total_cost_usd += cost_usd
            self._t.total_latency_s += latency_s

    def snapshot(self) -> dict:
        with self._lock:
            n = self._t.prs_reviewed
            return {
                "prs_reviewed": n,
                "total_tokens": self._t.total_tokens,
                "total_cost_usd": round(self._t.total_cost_usd, 6),
                "avg_tokens_per_pr": round(self._t.total_tokens / n, 1) if n else 0.0,
                "avg_cost_per_pr_usd": round(self._t.total_cost_usd / n, 6) if n else 0.0,
                "avg_latency_seconds": round(self._t.total_latency_s / n, 3) if n else 0.0,
            }

    def reset(self) -> None:
        with self._lock:
            self._t = _Totals()


# Process-wide singleton.
stats = StatsCollector()
