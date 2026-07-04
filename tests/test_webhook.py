"""Tests for webhook signature verification, parsing, and the endpoint."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.github.webhook import parse_pull_request_event, verify_signature
from app.main import app

SECRET = "topsecret"


def _sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# --------------------------------------------------------------------------- #
# Signature verification
# --------------------------------------------------------------------------- #
def test_verify_signature_valid():
    body = b'{"a":1}'
    assert verify_signature(SECRET, body, _sign(SECRET, body)) is True


def test_verify_signature_wrong_secret():
    body = b'{"a":1}'
    assert verify_signature(SECRET, body, _sign("other", body)) is False


def test_verify_signature_missing_or_malformed():
    body = b'{"a":1}'
    assert verify_signature(SECRET, body, None) is False
    assert verify_signature(SECRET, body, "md5=deadbeef") is False


# --------------------------------------------------------------------------- #
# Payload parsing
# --------------------------------------------------------------------------- #
def _pr_payload(action="opened", sha="abc123"):
    return {
        "action": action,
        "pull_request": {"number": 7, "title": "T", "body": "B", "head": {"sha": sha}},
        "repository": {"full_name": "octo/hello"},
    }


def test_parse_opened_event():
    ev = parse_pull_request_event(_pr_payload("opened"))
    assert ev is not None
    assert (ev.owner, ev.repo, ev.pr_number, ev.head_sha) == ("octo", "hello", 7, "abc123")


def test_parse_synchronize_event():
    assert parse_pull_request_event(_pr_payload("synchronize")) is not None


def test_parse_ignores_other_actions():
    assert parse_pull_request_event(_pr_payload("closed")) is None


def test_parse_ignores_non_pr_payload():
    assert parse_pull_request_event({"action": "opened", "zen": "x"}) is None


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(monkeypatch):
    # Pin the webhook secret and stub out the heavy paths.
    settings = get_settings()
    monkeypatch.setattr(settings, "GITHUB_WEBHOOK_SECRET", SECRET)

    calls: list[tuple] = []

    async def _fake_run_review(owner, repo, pr_number):
        calls.append((owner, repo, pr_number))

    async def _claim_true(_key):
        return True

    monkeypatch.setattr("app.main.run_review", _fake_run_review)
    monkeypatch.setattr("app.main.claim_review", _claim_true)

    with TestClient(app) as c:
        c.review_calls = calls  # type: ignore[attr-defined]
        yield c


def _post(client, payload, secret=SECRET):
    body = json.dumps(payload).encode()
    return client.post(
        "/webhook/github",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(secret, body),
            "Content-Type": "application/json",
        },
    )


def test_webhook_rejects_invalid_signature(client):
    body = json.dumps(_pr_payload()).encode()
    resp = client.post(
        "/webhook/github",
        content=body,
        headers={"X-Hub-Signature-256": _sign("wrong", body)},
    )
    assert resp.status_code == 401
    assert client.review_calls == []


def test_webhook_accepts_and_queues_review(client):
    resp = _post(client, _pr_payload("opened", sha="sha1"))
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    assert client.review_calls == [("octo", "hello", 7)]


def test_webhook_ignores_irrelevant_action(client):
    resp = _post(client, _pr_payload("closed"))
    assert resp.status_code == 200
    assert resp.json()["status"] == "ignored"
    assert client.review_calls == []


def test_webhook_dedupes_duplicate_sha(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "GITHUB_WEBHOOK_SECRET", SECRET)
    calls: list = []

    async def _fake_run_review(*a):
        calls.append(a)

    async def _claim_false(_key):  # already claimed → duplicate
        return False

    monkeypatch.setattr("app.main.run_review", _fake_run_review)
    monkeypatch.setattr("app.main.claim_review", _claim_false)

    with TestClient(app) as c:
        resp = _post(c, _pr_payload("synchronize", sha="dup"))
    assert resp.status_code == 200
    assert resp.json()["status"] == "duplicate"
    assert calls == []  # review NOT queued
