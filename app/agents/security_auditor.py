"""Security Auditor node: scans the diff for security issues."""

from __future__ import annotations

import logging

from app.agents.context import build_specialist_input
from app.agents.state import SECURITY_AUDITOR, PRReviewState, SecurityFindings
from app.llm import call_structured
from app.observability.cost import empty_usage

logger = logging.getLogger("app.agents.security_auditor")

SYSTEM_PROMPT = (
    "You are an application security engineer. Scan the diff for: hardcoded "
    "secrets/API keys, SQL/command/code injection, unsafe deserialization, "
    "missing input validation, insecure crypto, path traversal, SSRF, and risky "
    "dependency additions. For each issue give severity (low/medium/high/"
    "critical), category, file, line, description, and a concrete fix. If none "
    "found, say so explicitly. Do NOT invent issues. Output ONLY the structured "
    "schema."
)


def security_auditor_node(state: PRReviewState) -> dict:
    try:
        findings, usage = call_structured(
            SYSTEM_PROMPT, build_specialist_input(state), SecurityFindings
        )
        logger.info(
            "security_auditor produced findings",
            extra={"count": len(findings.findings)},
        )
        return {
            "security_findings": findings,
            "token_usage": {SECURITY_AUDITOR: usage},
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("security_auditor failed")
        return {
            "errors": [f"{SECURITY_AUDITOR}: {exc}"],
            "token_usage": {SECURITY_AUDITOR: empty_usage()},
        }
