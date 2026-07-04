"""LLM-as-judge: grade whether a diff-analysis summary is faithful to the
ground-truth intent of a change. Uses a separate structured LLM call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

JUDGE_SYSTEM = (
    "You are an impartial grader (LLM-as-judge). You are given the TRUE intent "
    "of a code change and a reviewer's summary of that change. Decide whether "
    "the summary faithfully and accurately captures the intent. Score from 1 "
    "(wrong or misleading) to 5 (fully faithful and specific). Do not reward "
    "verbosity. Output ONLY the structured schema."
)


class JudgeVerdict(BaseModel):
    """Grader output. Defaults keep offline/mocked runs valid."""

    faithful: bool = False
    score: int = Field(default=0, ge=0, le=5)
    reasoning: str = ""


def judge_summary(summary: str, ground_truth_intent: str) -> JudgeVerdict | None:
    """Grade a reviewer summary; returns None if the judge call fails."""
    # Imported lazily so scoring/tests don't require the LLM layer.
    from app.llm import call_structured

    human = (
        f"TRUE intent of the change:\n{ground_truth_intent}\n\n"
        f"Reviewer's summary:\n{summary or '(empty)'}"
    )
    try:
        verdict, _usage = call_structured(JUDGE_SYSTEM, human, JudgeVerdict)
        return verdict
    except Exception:  # noqa: BLE001 - a judge failure shouldn't abort the eval
        return None
