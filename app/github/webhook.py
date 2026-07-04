"""GitHub webhook: HMAC signature verification + payload parsing.

Verifying `X-Hub-Signature-256` ensures the request genuinely came from GitHub
(the shared secret), not a spoofed caller — a hard requirement for a public
endpoint that triggers real work.
"""

from __future__ import annotations

import hashlib
import hmac

from pydantic import BaseModel

# GitHub PR actions we actually review.
RELEVANT_ACTIONS = frozenset({"opened", "synchronize"})


class PullRequestEvent(BaseModel):
    """Normalised view of a `pull_request` webhook payload."""

    action: str
    owner: str
    repo: str
    pr_number: int
    head_sha: str
    title: str = ""
    body: str = ""


def verify_signature(secret: str, body: bytes, signature_header: str | None) -> bool:
    """Constant-time verify GitHub's `X-Hub-Signature-256` header."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    expected = f"sha256={digest}"
    return hmac.compare_digest(expected, signature_header)


def _resolve_owner_repo(repository: dict) -> tuple[str, str]:
    """Extract owner/name from a repository payload (full_name or nested)."""
    full = repository.get("full_name", "")
    if "/" in full:
        owner, _, name = full.partition("/")
        return owner, name
    owner = (repository.get("owner") or {}).get("login", "")
    return owner, repository.get("name", "")


def parse_pull_request_event(payload: dict) -> PullRequestEvent | None:
    """Parse a webhook payload into a PullRequestEvent, or None if not actionable.

    Returns None for actions other than opened/synchronize, or non-PR payloads.
    """
    action = payload.get("action")
    if action not in RELEVANT_ACTIONS:
        return None
    pr = payload.get("pull_request")
    if not pr:
        return None
    owner, name = _resolve_owner_repo(payload.get("repository") or {})
    if not owner or not name:
        return None
    return PullRequestEvent(
        action=action,
        owner=owner,
        repo=name,
        pr_number=pr["number"],
        head_sha=pr["head"]["sha"],
        title=pr.get("title") or "",
        body=pr.get("body") or "",
    )
