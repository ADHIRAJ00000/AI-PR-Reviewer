"""Output guardrails applied to the final review comment before posting.

  * Redact any secret-looking strings that slipped through.
  * Strip lines that look like the model leaking its own system prompt.
  * Cap the comment length so a runaway generation can't post a wall of text.

Schema enforcement for structured agent outputs lives in `app.llm`
(`call_structured` validates and retries once); this module handles the final
human-facing text.
"""

from __future__ import annotations

from app.guardrails.input_guard import redact_secrets

OUTPUT_MAX_CHARS = 8_000

# Phrases lifted from the agents' system prompts; if they appear in output the
# model is echoing its instructions — drop those lines.
_PROMPT_LEAK_MARKERS = (
    "you are the lead reviewer",
    "you are a senior code reviewer",
    "you are a test engineer",
    "you are an application security engineer",
    "output only the structured schema",
)


def _strip_prompt_leak(text: str) -> str:
    kept = [
        line
        for line in text.splitlines()
        if not any(marker in line.lower() for marker in _PROMPT_LEAK_MARKERS)
    ]
    return "\n".join(kept)


def guard_output(text: str, *, max_chars: int = OUTPUT_MAX_CHARS) -> str:
    """Sanitize the final review comment for posting."""
    text = redact_secrets(text)
    text = _strip_prompt_leak(text)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n_[review truncated to length limit]_"
    return text
