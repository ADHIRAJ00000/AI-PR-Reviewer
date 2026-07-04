"""LangChain tools that expose the GitHub client to tool-calling agents.

Each tool:
  * has a crisp, API-doc-style description (the LLM decides usage from it),
  * takes typed arguments via a Pydantic `args_schema`,
  * handles its own errors and ALWAYS returns a structured envelope:
        {"ok": bool, "data": <result> | None, "error": <str> | None}
    so the agent can reason about success/failure instead of seeing a raised
    exception.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import get_settings
from app.github.client import GitHubClient, GitHubError

logger = logging.getLogger("app.tools.github")


# --------------------------------------------------------------------------- #
# Structured envelope helpers
# --------------------------------------------------------------------------- #
def _ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None}


def _err(message: str) -> dict[str, Any]:
    return {"ok": False, "data": None, "error": message}


def _make_client() -> GitHubClient:
    """Build a client from the configured token (one per tool invocation)."""
    return GitHubClient(token=get_settings().GITHUB_TOKEN)


# --------------------------------------------------------------------------- #
# Argument schemas
# --------------------------------------------------------------------------- #
class PRRef(BaseModel):
    """Reference to a specific pull request."""

    owner: str = Field(..., description="Repository owner or org, e.g. 'octocat'.")
    repo: str = Field(..., description="Repository name, e.g. 'Hello-World'.")
    pr_number: int = Field(..., description="The pull request number, e.g. 42.")


class PostReviewInput(PRRef):
    """Arguments for posting the final review comment."""

    body: str = Field(..., description="Markdown body of the review comment to post.")


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@tool(args_schema=PRRef)
async def fetch_pr_diff(owner: str, repo: str, pr_number: int) -> dict[str, Any]:
    """Fetch the raw unified git diff for a pull request.

    Use this to read exactly what code changed. Returns the full diff text as a
    single string under `data`. Returns `ok: false` with an `error` if the PR is
    not found or GitHub is unreachable.
    """
    try:
        async with _make_client() as gh:
            diff = await gh.get_pr_diff(owner, repo, pr_number)
        return _ok(diff)
    except GitHubError as exc:
        logger.warning("fetch_pr_diff failed", extra={"error": exc.message})
        return _err(exc.message)
    except Exception as exc:  # noqa: BLE001 - tools must never raise to the agent
        logger.exception("fetch_pr_diff unexpected error")
        return _err(f"unexpected error: {exc!s}")


@tool(args_schema=PRRef)
async def fetch_pr_files(owner: str, repo: str, pr_number: int) -> dict[str, Any]:
    """List the files changed in a pull request with their patch hunks.

    Returns a list under `data`; each item has `filename`, `status`, `additions`,
    `deletions`, `changes`, and `patch` (the per-file diff, may be null for binary
    or very large files). Prefer this over the raw diff when you need per-file
    structure. Returns `ok: false` with an `error` on failure.
    """
    try:
        async with _make_client() as gh:
            files = await gh.get_pr_files(owner, repo, pr_number)
        return _ok([f.model_dump() for f in files])
    except GitHubError as exc:
        logger.warning("fetch_pr_files failed", extra={"error": exc.message})
        return _err(exc.message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("fetch_pr_files unexpected error")
        return _err(f"unexpected error: {exc!s}")


@tool(args_schema=PostReviewInput)
async def post_pr_review(
    owner: str, repo: str, pr_number: int, body: str
) -> dict[str, Any]:
    """Post the final review as a single Markdown comment on the pull request.

    Call this exactly once, at the end, with the fully composed review. Returns
    the created comment's `id` and `html_url` under `data`. Returns `ok: false`
    with an `error` if posting fails (e.g. missing write permission).
    """
    try:
        async with _make_client() as gh:
            comment = await gh.post_review_comment(owner, repo, pr_number, body)
        return _ok(comment.model_dump())
    except GitHubError as exc:
        logger.warning("post_pr_review failed", extra={"error": exc.message})
        return _err(exc.message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("post_pr_review unexpected error")
        return _err(f"unexpected error: {exc!s}")


# Convenient registry for binding to an LLM.
GITHUB_TOOLS = [fetch_pr_diff, fetch_pr_files, post_pr_review]
