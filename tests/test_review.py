"""Test the end-to-end review runner with a fake GitHub client + fake LLM."""

from __future__ import annotations

from app.github.client import IssueComment, PRFile, PullRequest
from app.review import run_review


class FakeGitHub:
    """Records the posted review; returns canned PR data."""

    def __init__(self):
        self.posted: str | None = None
        self.closed = False

    async def get_pull_request(self, owner, repo, pr_number):
        return PullRequest(
            number=pr_number, title="Add login", body="adds auth",
            author="alice", state="open", base_sha="base", head_sha="head",
        )

    async def get_pr_diff(self, owner, repo, pr_number):
        return "diff --git a/auth.py b/auth.py\n+def login(): ..."

    async def get_pr_files(self, owner, repo, pr_number):
        return [PRFile(filename="auth.py", status="modified",
                       additions=3, deletions=0, changes=3, patch="@@")]

    async def post_review_comment(self, owner, repo, pr_number, body):
        self.posted = body
        return IssueComment(id=1, html_url="https://gh/c/1", body=body)

    async def aclose(self):
        self.closed = True


async def test_run_review_posts_generated_review(fake_llm):
    fake_llm(text="## Verdict\n\nRequest changes — hardcoded secret.")
    gh = FakeGitHub()

    result = await run_review("octo", "hello", 42, client=gh)

    # The graph's final review was posted verbatim.
    assert gh.posted is not None
    assert "Request changes" in gh.posted
    assert result["final_review"] == gh.posted
    # Cost tracking populated for all agents that ran.
    assert set(result["token_usage"]) == {
        "diff_analyzer", "test_suggester", "security_auditor", "summarizer",
    }
    # Injected client is not closed by the runner (caller owns it).
    assert gh.closed is False


async def test_run_review_posts_even_when_summarizer_fails(fake_llm):
    """Summarizer LLM failure → deterministic fallback still gets posted."""
    fake_llm(fail_text=True)
    gh = FakeGitHub()

    result = await run_review("octo", "hello", 43, client=gh)

    assert gh.posted  # a review was still posted
    assert result["final_review"] == gh.posted
