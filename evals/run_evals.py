"""Run the review graph over the fixture PRs and print a scorecard.

Usage:  python evals/run_evals.py
Requires a real LLM_API_KEY for meaningful numbers (it runs the actual agents
and an LLM-as-judge). The scoring math is unit-tested independently.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

# Make `app` importable when run as a script from the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.agents.graph import compile_graph  # noqa: E402
from app.agents.state import FileChange, new_state  # noqa: E402
from evals.judge import judge_summary  # noqa: E402
from evals.scoring import build_scorecard, count_false_positives, security_recall  # noqa: E402

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"
GOLDEN_PATH = pathlib.Path(__file__).parent / "golden_set.json"


def parse_changed_files(diff: str) -> list[FileChange]:
    """Derive changed files from a unified diff's `+++ b/<path>` lines."""
    files: list[FileChange] = []
    for match in re.finditer(r"^\+\+\+ b/(.+)$", diff, flags=re.M):
        path = match.group(1).strip()
        if path and path != "/dev/null":
            files.append(FileChange(filename=path, status="modified", changes=1))
    return files


def run_fixture(graph, name: str, diff: str) -> dict:
    """Run the compiled graph on one fixture diff; return final state."""
    state = new_state(
        owner="evals",
        repo="fixtures",
        pr_number=0,
        pr_title=name,
        diff=diff,
        changed_files=parse_changed_files(diff),
    )
    return graph.invoke(state)


def score_fixture(name: str, meta: dict, state: dict, *, use_judge: bool = True) -> dict:
    """Score one fixture's result into a scorecard row."""
    sec = state.get("security_findings")
    findings = sec.findings if sec else []
    expected = meta.get("expected_vulnerabilities", [])

    if meta["kind"] == "insecure":
        caught, total = security_recall(findings, expected)
        false_positives = 0
    else:
        caught, total = 0, 0
        false_positives = count_false_positives(findings)

    judge_score = None
    if use_judge:
        diff_findings = state.get("diff_findings")
        summary = diff_findings.overall_summary if diff_findings else ""
        verdict = judge_summary(summary, meta["intent"])
        if verdict is not None:
            judge_score = verdict.score

    return {
        "name": name,
        "kind": meta["kind"],
        "caught": caught,
        "total": total,
        "false_positives": false_positives,
        "judge_score": judge_score,
    }


def format_scorecard(agg: dict, rows: list[dict]) -> str:
    """Render a human-readable scorecard table."""
    lines = ["", "=" * 68, "  PR-REVIEWER EVAL SCORECARD", "=" * 68, ""]
    lines.append(f"  {'fixture':<28}{'kind':<10}{'recall':<10}{'FP':<6}judge")
    lines.append("  " + "-" * 62)
    for r in rows:
        recall = f"{r['caught']}/{r['total']}" if r["kind"] == "insecure" else "—"
        judge = str(r["judge_score"]) if r["judge_score"] is not None else "—"
        lines.append(
            f"  {r['name']:<28}{r['kind']:<10}{recall:<10}"
            f"{r['false_positives']:<6}{judge}"
        )
    lines.append("  " + "-" * 62)
    lines.append("")
    lines.append(
        f"  Security recall:        {agg['vulnerabilities_caught']}/"
        f"{agg['planted_vulnerabilities']}  "
        f"({agg['security_recall_pct']}%)"
    )
    lines.append(
        f"  False-positive rate:    {agg['false_positive_findings']} findings on "
        f"{agg['clean_fixtures']} clean fixtures  ({agg['false_positive_rate_pct']}%)"
    )
    judge_pct = agg["judge_faithfulness_pct"]
    lines.append(
        "  Diff-analysis (judge):  "
        + (f"{agg['judge_faithfulness_avg']}/5  ({judge_pct}% faithful)"
           if judge_pct is not None else "n/a")
    )
    lines.append("=" * 68)
    return "\n".join(lines)


def main() -> None:
    golden = json.loads(GOLDEN_PATH.read_text())
    graph = compile_graph()
    rows: list[dict] = []
    for name, meta in golden.items():
        diff = (FIXTURES_DIR / name).read_text()
        state = run_fixture(graph, name, diff)
        row = score_fixture(name, meta, state, use_judge=True)
        rows.append(row)
        print(f"  scored {name} ...")
    agg = build_scorecard(rows)
    print(format_scorecard(agg, rows))


if __name__ == "__main__":
    main()
