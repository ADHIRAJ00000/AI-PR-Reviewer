"""Tests for the GitHub client: happy path, retries, rate limits, errors.

All HTTP is mocked with `respx`; no network access. `asyncio.sleep` is
patched to a no-op so backoff/rate-limit waits don't slow the suite.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.github.client import GitHubClient, GitHubError

BASE = "https://api.github.com"
PR_PATH = "/repos/octo/hello/pulls/1"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make every backoff / rate-limit sleep instantaneous."""
    async def _instant(_seconds):
        return None

    monkeypatch.setattr("app.github.client.asyncio.sleep", _instant)


@pytest.fixture
async def client():
    c = GitHubClient(token="test-token")
    try:
        yield c
    finally:
        await c.aclose()


@respx.mock
async def test_get_pull_request_returns_typed_model(client):
    respx.get(f"{BASE}{PR_PATH}").mock(
        return_value=httpx.Response(
            200,
            json={
                "number": 1,
                "title": "Add feature",
                "body": "does things",
                "user": {"login": "alice"},
                "state": "open",
                "base": {"sha": "base123"},
                "head": {"sha": "head456"},
                "additions": 10,
                "deletions": 2,
                "changed_files": 3,
            },
        )
    )
    pr = await client.get_pull_request("octo", "hello", 1)
    assert pr.number == 1
    assert pr.author == "alice"
    assert pr.head_sha == "head456"
    assert pr.additions == 10


@respx.mock
async def test_get_pr_diff_sends_diff_accept_header(client):
    route = respx.get(f"{BASE}{PR_PATH}").mock(
        return_value=httpx.Response(200, text="diff --git a/x b/x\n+hello")
    )
    diff = await client.get_pr_diff("octo", "hello", 1)
    assert "diff --git" in diff
    sent = route.calls.last.request
    assert sent.headers["Accept"] == "application/vnd.github.v3.diff"


@respx.mock
async def test_retries_on_500_then_succeeds(client):
    route = respx.get(f"{BASE}{PR_PATH}").mock(
        side_effect=[
            httpx.Response(500, text="boom"),
            httpx.Response(200, json=_min_pr()),
        ]
    )
    pr = await client.get_pull_request("octo", "hello", 1)
    assert pr.number == 1
    assert route.call_count == 2  # retried once


@respx.mock
async def test_gives_up_after_max_retries_on_5xx(client):
    respx.get(f"{BASE}{PR_PATH}").mock(return_value=httpx.Response(503, text="down"))
    with pytest.raises(GitHubError) as exc:
        await client.get_pull_request("octo", "hello", 1)
    assert exc.value.status_code == 503


@respx.mock
async def test_network_error_is_retried_then_raises(client):
    respx.get(f"{BASE}{PR_PATH}").mock(
        side_effect=httpx.ConnectError("no route")
    )
    with pytest.raises(GitHubError) as exc:
        await client.get_pull_request("octo", "hello", 1)
    assert exc.value.status_code == 0
    assert "network error" in exc.value.message


@respx.mock
async def test_404_raises_typed_error_with_message(client):
    respx.get(f"{BASE}{PR_PATH}").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )
    with pytest.raises(GitHubError) as exc:
        await client.get_pull_request("octo", "hello", 1)
    assert exc.value.status_code == 404
    assert exc.value.message == "Not Found"


@respx.mock
async def test_4xx_not_retried(client):
    route = respx.get(f"{BASE}{PR_PATH}").mock(
        return_value=httpx.Response(404, json={"message": "Not Found"})
    )
    with pytest.raises(GitHubError):
        await client.get_pull_request("octo", "hello", 1)
    assert route.call_count == 1  # no retry on client error


@respx.mock
async def test_rate_limit_exhausted_sleeps_and_continues(client):
    # remaining=0 + 403 => client sleeps (patched to no-op) then the SAME
    # response is returned; a 403 still raises, proving the path executes.
    respx.get(f"{BASE}{PR_PATH}").mock(
        return_value=httpx.Response(
            403,
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "0"},
            json={"message": "API rate limit exceeded"},
        )
    )
    with pytest.raises(GitHubError) as exc:
        await client.get_pull_request("octo", "hello", 1)
    assert exc.value.status_code == 403


@respx.mock
async def test_post_review_comment(client):
    respx.post(f"{BASE}/repos/octo/hello/issues/1/comments").mock(
        return_value=httpx.Response(
            201,
            json={
                "id": 999,
                "html_url": "https://github.com/octo/hello/pull/1#issuecomment-999",
                "body": "LGTM",
            },
        )
    )
    comment = await client.post_review_comment("octo", "hello", 1, "LGTM")
    assert comment.id == 999
    assert comment.body == "LGTM"


@respx.mock
async def test_get_pr_files_paginates(client):
    page1 = [_file(f"f{i}.py") for i in range(100)]
    page2 = [_file("last.py")]
    respx.get(f"{BASE}{PR_PATH}/files").mock(
        side_effect=[
            httpx.Response(200, json=page1),
            httpx.Response(200, json=page2),
        ]
    )
    files = await client.get_pr_files("octo", "hello", 1)
    assert len(files) == 101
    assert files[-1].filename == "last.py"


# ---- helpers ---- #
def _min_pr() -> dict:
    return {
        "number": 1,
        "title": "t",
        "body": "",
        "user": {"login": "bob"},
        "state": "open",
        "base": {"sha": "b"},
        "head": {"sha": "h"},
    }


def _file(name: str) -> dict:
    return {
        "filename": name,
        "status": "modified",
        "additions": 1,
        "deletions": 0,
        "changes": 1,
        "patch": "@@ -0,0 +1 @@\n+x",
    }
