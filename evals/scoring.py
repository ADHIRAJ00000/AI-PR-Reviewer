"""Pure scoring functions for the eval harness (no I/O, fully unit-testable).

Metrics:
  * Security recall — of the planted vulnerabilities, how many did the auditor
    catch?
  * False-positive rate — how often did the auditor flag an issue on a
    fixture that has none?
"""

from __future__ import annotations

from app.agents.state import SecurityFinding

_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def finding_matches(finding: SecurityFinding, expected: dict) -> bool:
    """True if a finding plausibly corresponds to an expected vulnerability.

    Match rule: at least one expected keyword appears in the finding's category
    or description, AND (if the finding names a file) it matches the expected
    file.
    """
    text = f"{finding.category} {finding.description}".lower()
    keywords = [k.lower() for k in expected.get("category_keywords", [])]
    if not any(k in text for k in keywords):
        return False
    expected_file = (expected.get("file") or "").lower()
    if expected_file and finding.file and expected_file not in finding.file.lower():
        return False
    return True


def security_recall(
    findings: list[SecurityFinding], expected: list[dict]
) -> tuple[int, int]:
    """Return (caught, total) planted vulnerabilities."""
    caught = sum(
        1 for exp in expected if any(finding_matches(f, exp) for f in findings)
    )
    return caught, len(expected)


def count_false_positives(
    findings: list[SecurityFinding], *, min_severity: str = "medium"
) -> int:
    """Count findings at/above `min_severity` (used on no-vuln fixtures)."""
    threshold = _SEVERITY_ORDER[min_severity]
    return sum(
        1 for f in findings if _SEVERITY_ORDER.get(f.severity, 0) >= threshold
    )


def build_scorecard(rows: list[dict]) -> dict:
    """Aggregate per-fixture rows into headline metrics.

    Each row: {name, kind, caught, total, false_positives, judge_score|None}.
    """
    insecure = [r for r in rows if r["kind"] == "insecure"]
    no_vuln = [r for r in rows if r["kind"] != "insecure"]

    caught = sum(r["caught"] for r in insecure)
    planted = sum(r["total"] for r in insecure)

    fp_findings = sum(r["false_positives"] for r in no_vuln)
    fixtures_with_fp = sum(1 for r in no_vuln if r["false_positives"] > 0)

    judge_scores = [r["judge_score"] for r in rows if r.get("judge_score") is not None]

    return {
        "fixtures": len(rows),
        "planted_vulnerabilities": planted,
        "vulnerabilities_caught": caught,
        "security_recall_pct": round(100 * caught / planted, 1) if planted else 0.0,
        "false_positive_findings": fp_findings,
        "clean_fixtures": len(no_vuln),
        "false_positive_rate_pct": (
            round(100 * fixtures_with_fp / len(no_vuln), 1) if no_vuln else 0.0
        ),
        "judge_faithfulness_avg": (
            round(sum(judge_scores) / len(judge_scores), 2) if judge_scores else None
        ),
        "judge_faithfulness_pct": (
            round(100 * (sum(judge_scores) / len(judge_scores)) / 5, 1)
            if judge_scores
            else None
        ),
    }
