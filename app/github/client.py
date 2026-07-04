"""Async GitHub API client — the only component that talks to GitHub.

Design goals:
  * Typed Pydantic models out, never raw dicts.
  * Exponential-backoff retry on 5xx / network errors (max 3 attempts).
  * Rate-limit awareness via `X-RateLimit-Remaining` / `X-RateLimit-Reset`.
  * A single typed `GitHubError` on failure — never swallow errors silently.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from types import TracebackType
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("app.github.client")

GITHUB_API_URL = "https://api.github.com"
API_VERSION = "2022-11-28"

# Back off proactively once remaining quota drops to/below this.
RATE_LIMIT_FLOOR = 5


class GitHubError(RuntimeError):
    """Raised when a GitHub request ultimately fails.

    Attributes:
        status_code: HTTP status (0 for network/transport errors).
        message: Human-readable description.
    """

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(f"GitHub API error [{status_code}]: {message}")


# --------------------------------------------------------------------------- #
# Typed response models
# --------------------------------------------------------------------------- #
class PullRequest(BaseModel):
    """Minimal PR metadata the review pipeline needs."""

    number: int
    title: str
    body: str = ""
    author: str
    state: str
    base_sha: str
    head_sha: str
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0


class PRFile(BaseModel):
    """A single changed file in a PR."""

    filename: str
    status: str  # added | modified | removed | renamed | ...
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    # `patch` may be absent for binary files or very large diffs.
    patch: str | None = None


class IssueComment(BaseModel):
    """A posted issue/PR comment."""

    id: int
    html_url: str
    body: str


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class GitHubClient:
    """Thin, resilient async wrapper over the GitHub REST API."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = GITHUB_API_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": "autonomous-pr-reviewer",
            },
        )

    # ---- lifecycle ---- #
    async def __aenter__(self) -> "GitHubClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ---- core request with retry + rate-limit handling ---- #
    async def _request(
        self,
        method: str,
        path: str,
        *,
        accept: str = "application/vnd.github+json",
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Perform a request, retrying transient failures with backoff."""
        headers = {"Accept": accept}
        last_error: str = "unknown error"

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.request(
                    method, path, headers=headers, json=json
                )
            except httpx.TransportError as exc:  # network / connect / timeout
                last_error = f"network error: {exc!s}"
                logger.warning(
                    "github request transport error",
                    extra={"path": path, "attempt": attempt, "error": str(exc)},
                )
                if attempt < self._max_retries:
                    await self._backoff(attempt)
                    continue
                raise GitHubError(0, last_error) from exc

            await self._handle_rate_limit(response)

            if response.status_code >= 500:
                last_error = f"server error {response.status_code}"
                logger.warning(
                    "github request 5xx",
                    extra={"path": path, "attempt": attempt,
                           "status": response.status_code},
                )
                if attempt < self._max_retries:
                    await self._backoff(attempt)
                    continue
                raise GitHubError(response.status_code, response.text[:500])

            if response.status_code >= 400:
                # Client errors are not retried (except rate limit handled above).
                message = self._error_message(response)
                raise GitHubError(response.status_code, message)

            return response

        # Loop exits only via return/raise above; this is defensive.
        raise GitHubError(0, last_error)

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter: ~0.5s, 1s, 2s ..."""
        delay = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
        await asyncio.sleep(delay)

    async def _handle_rate_limit(self, response: httpx.Response) -> None:
        """Log low quota and sleep until reset if we've hit zero."""
        remaining_raw = response.headers.get("X-RateLimit-Remaining")
        if remaining_raw is None:
            return
        try:
            remaining = int(remaining_raw)
        except ValueError:
            return

        if remaining > RATE_LIMIT_FLOOR:
            return

        reset_raw = response.headers.get("X-RateLimit-Reset")
        logger.warning(
            "github rate limit low",
            extra={"remaining": remaining, "reset": reset_raw},
        )

        # Only actually wait when we're fully exhausted and got throttled.
        if remaining <= 0 and response.status_code in (403, 429) and reset_raw:
            try:
                wait = max(0.0, float(reset_raw) - time.time()) + 1.0
            except ValueError:
                return
            # Cap the wait so a bad header can't hang the worker forever.
            wait = min(wait, 60.0)
            logger.warning("github rate limit sleeping", extra={"seconds": wait})
            await asyncio.sleep(wait)

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        """Extract a readable message from a GitHub error body."""
        try:
            data = response.json()
            if isinstance(data, dict) and "message" in data:
                return str(data["message"])
        except Exception:  # noqa: BLE001 - body may not be JSON
            pass
        return response.text[:500] or response.reason_phrase

    # ---- public API ---- #
    async def get_pull_request(
        self, owner: str, repo: str, pr_number: int
    ) -> PullRequest:
        """Fetch PR metadata (title, body, author, base/head SHA)."""
        resp = await self._request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}")
        data = resp.json()
        return PullRequest(
            number=data["number"],
            title=data.get("title") or "",
            body=data.get("body") or "",
            author=(data.get("user") or {}).get("login", "unknown"),
            state=data.get("state", "unknown"),
            base_sha=data["base"]["sha"],
            head_sha=data["head"]["sha"],
            additions=data.get("additions", 0),
            deletions=data.get("deletions", 0),
            changed_files=data.get("changed_files", 0),
        )

    async def get_pr_diff(self, owner: str, repo: str, pr_number: int) -> str:
        """Fetch the raw unified diff text for a PR."""
        resp = await self._request(
            "GET",
            f"/repos/{owner}/{repo}/pulls/{pr_number}",
            accept="application/vnd.github.v3.diff",
        )
        return resp.text

    async def get_pr_files(
        self, owner: str, repo: str, pr_number: int
    ) -> list[PRFile]:
        """Fetch changed files (with patch hunks), paginating fully."""
        files: list[PRFile] = []
        page = 1
        while True:
            resp = await self._request(
                "GET",
                f"/repos/{owner}/{repo}/pulls/{pr_number}/files"
                f"?per_page=100&page={page}",
            )
            batch = resp.json()
            if not batch:
                break
            for item in batch:
                files.append(
                    PRFile(
                        filename=item["filename"],
                        status=item.get("status", "modified"),
                        additions=item.get("additions", 0),
                        deletions=item.get("deletions", 0),
                        changes=item.get("changes", 0),
                        patch=item.get("patch"),
                    )
                )
            if len(batch) < 100:
                break
            page += 1
        return files

    async def post_review_comment(
        self, owner: str, repo: str, pr_number: int, body: str
    ) -> IssueComment:
        """Post a single issue comment on the PR (the final review)."""
        resp = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
            json={"body": body},
        )
        data = resp.json()
        return IssueComment(
            id=data["id"], html_url=data["html_url"], body=data["body"]
        )

    async def post_inline_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        *,
        commit_sha: str,
        path: str,
        line: int,
        body: str,
        side: Literal["LEFT", "RIGHT"] = "RIGHT",
    ) -> IssueComment:
        """Post a line-level review comment on a specific file/line (optional)."""
        resp = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls/{pr_number}/comments",
            json={
                "body": body,
                "commit_id": commit_sha,
                "path": path,
                "line": line,
                "side": side,
            },
        )
        data = resp.json()
        return IssueComment(
            id=data["id"], html_url=data["html_url"], body=data["body"]
        )
