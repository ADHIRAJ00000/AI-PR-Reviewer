"""End-to-end review runner: fetch PR → run graph → post review comment.

This is what the webhook background task calls. It's the single place that
stitches the GitHub client, the agent graph, and the cost logging together.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from app.agents.graph import compile_graph
from app.agents.state import FileChange, new_state
from app.config import get_settings
from app.github.client import GitHubClient
from app.observability.stats import stats
from app.observability.tracing import build_trace, flush

logger = logging.getLogger("app.review")


@lru_cache
def _graph():
    """Compile the graph once and reuse it."""
    return compile_graph()


async def run_review(
    owner: str,
    repo: str,
    pr_number: int,
    *,
    client: GitHubClient | None = None,
) -> dict:
    """Fetch the PR, run the multi-agent review, and post the result.

    `client` may be injected for testing; otherwise one is built from settings.
    Returns the final graph state.
    """
    gh = client or GitHubClient(token=get_settings().GITHUB_TOKEN)
    owns_client = client is None
    try:
        pr = await gh.get_pull_request(owner, repo, pr_number)
        diff = await gh.get_pr_diff(owner, repo, pr_number)
        files = await gh.get_pr_files(owner, repo, pr_number)

        state = new_state(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            pr_title=pr.title,
            pr_body=pr.body,
            diff=diff,
            changed_files=[FileChange(**f.model_dump()) for f in files],
        )

        # One Langfuse trace per PR; each agent's LLM call is a child span.
        config, handler = build_trace(
            run_name=f"pr-review {owner}/{repo}#{pr_number}",
            metadata={"owner": owner, "repo": repo, "pr_number": pr_number},
        )
        started = time.perf_counter()
        try:
            result = await _graph().ainvoke(state, config=config)
        finally:
            flush(handler)
        latency = time.perf_counter() - started

        review = result.get("final_review") or "_No review was generated._"
        await gh.post_review_comment(owner, repo, pr_number, review)

        _record_and_log(owner, repo, pr_number, result, latency)
        return result
    except Exception:  # noqa: BLE001 - background task must not crash silently
        logger.exception(
            "review run failed",
            extra={"owner": owner, "repo": repo, "pr_number": pr_number},
        )
        raise
    finally:
        if owns_client:
            await gh.aclose()


def _record_and_log(
    owner: str, repo: str, pr_number: int, result: dict, latency_s: float
) -> None:
    """Update aggregate stats and emit the per-PR cost summary line."""
    usage = result.get("token_usage", {})
    total_tokens = sum(u.get("total", 0) for u in usage.values())
    total_cost = sum(u.get("cost_usd", 0.0) for u in usage.values())

    stats.record(tokens=total_tokens, cost_usd=total_cost, latency_s=latency_s)

    logger.info(
        "PR #%s reviewed — %d agents, %d tokens, $%.4f (%.1fs)",
        pr_number,
        len(usage),
        total_tokens,
        total_cost,
        latency_s,
        extra={
            "repo": f"{owner}/{repo}",
            "pr_number": pr_number,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "latency_seconds": round(latency_s, 3),
            "errors": result.get("errors", []),
        },
    )
