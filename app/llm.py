"""LLM factory + structured/text call helpers.

Central place that builds the chat model (provider factory) and runs calls with
structured output, capturing token usage for cost tracking. Agents call
`call_structured` / `call_text` and never touch the provider SDK directly.

Reliability on free tiers: Groq counts `prompt + max_tokens` against the
per-minute token budget, so a small over-provisioned `max_tokens` used to make
three parallel agents blow the limit. We keep `max_tokens` tight, retry on
transient rate limits (respecting the retry hint), and — if one model's quota is
exhausted for the day — fall back to another model that has its own quota.

Note on models: Claude Sonnet 5 / Opus 4.8 use adaptive thinking and reject
`temperature` / `top_p` / `budget_tokens`, so we never pass sampling params.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.config import ConfigError, get_settings
from app.observability.cost import usage_record

logger = logging.getLogger("app.llm")

TModel = TypeVar("TModel", bound=BaseModel)
TResult = TypeVar("TResult")

# The agents emit a handful of short findings; they never need a large budget.
# Keeping this tight is what keeps three parallel agents under the per-minute
# token limit on Groq's free tier.
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT = 90.0

# Rate-limit resilience.
_MAX_ATTEMPTS_PER_MODEL = 3
_MAX_BACKOFF_S = 20.0  # a longer wait means a daily cap — switch models instead.

# Fallback chains: models with independent quotas, ordered by capability. If the
# configured model is rate-limited, we try the next one that isn't the same.
_GROQ_FALLBACKS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
]


class StructuredOutputError(RuntimeError):
    """Raised when the model does not return a valid structured object."""


def get_chat_model(
    *, model: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS
) -> BaseChatModel:
    """Build a chat model for the configured provider."""
    settings = get_settings()
    if settings.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or settings.LLM_MODEL,  # type: ignore[call-arg]
            api_key=settings.LLM_API_KEY,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            timeout=DEFAULT_TIMEOUT,
            max_retries=2,
            stop=None,
        )
    if settings.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        # Groq serves open models (Llama, etc.) with tool-calling + structured
        # output. temperature=0 keeps reviews deterministic-ish.
        return ChatGroq(
            model=model or settings.LLM_MODEL,  # type: ignore[call-arg]
            api_key=settings.LLM_API_KEY,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=0,
            timeout=DEFAULT_TIMEOUT,
            max_retries=0,  # we handle retries/fallback ourselves
        )
    raise ConfigError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER!r}")


def _model_chain() -> list[str]:
    """The configured model first, then any provider fallbacks with own quota."""
    settings = get_settings()
    chain = [settings.LLM_MODEL]
    if settings.LLM_PROVIDER == "groq":
        for m in _GROQ_FALLBACKS:
            if m not in chain:
                chain.append(m)
    return chain


def _is_rate_limit(exc: BaseException) -> bool:
    """True for provider rate-limit / quota errors (HTTP 429)."""
    text = str(exc).lower()
    return (
        exc.__class__.__name__ == "RateLimitError"
        or "rate_limit" in text
        or "429" in text
        or "too many requests" in text
    )


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Parse the wait hint from a rate-limit error (e.g. 'try again in 12.4s').

    Returns seconds to wait, or None if no hint was found. A per-minute (TPM)
    limit yields a few seconds; a per-day (TPD) limit yields minutes/hours.
    """
    m = re.search(
        r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", str(exc), re.IGNORECASE
    )
    if not m:
        return None
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = float(m.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _run_resilient(call: Callable[[str], TResult]) -> TResult:
    """Run `call(model_name)`, retrying on rate limits and falling back models.

    Short waits (a per-minute limit) are slept out and retried on the same model.
    A long wait (a daily cap) skips straight to the next model in the chain.
    Non-rate-limit errors propagate immediately so the agent's own error
    handling (graceful degradation) still applies.
    """
    last_exc: BaseException | None = None
    for model_name in _model_chain():
        for attempt in range(1, _MAX_ATTEMPTS_PER_MODEL + 1):
            try:
                return call(model_name)
            except Exception as exc:  # noqa: BLE001 - inspected below
                if not _is_rate_limit(exc):
                    raise
                last_exc = exc
                wait = _retry_after_seconds(exc)
                if wait is not None and wait <= _MAX_BACKOFF_S and attempt < _MAX_ATTEMPTS_PER_MODEL:
                    logger.warning(
                        "rate limited; backing off",
                        extra={"model": model_name, "attempt": attempt, "sleep_s": round(wait, 1)},
                    )
                    time.sleep(wait + 0.5)
                    continue
                logger.warning(
                    "rate limited; switching model",
                    extra={"model": model_name, "wait_s": wait},
                )
                break  # move to the next model in the chain
    assert last_exc is not None
    raise last_exc


def _usage_from_raw(raw: Any, model_name: str) -> dict:
    """Extract a token/cost record from a raw AIMessage's usage metadata."""
    meta = getattr(raw, "usage_metadata", None) or {}
    return usage_record(
        model_name,
        int(meta.get("input_tokens", 0) or 0),
        int(meta.get("output_tokens", 0) or 0),
    )


def call_structured(
    system_prompt: str, human_content: str, schema: type[TModel]
) -> tuple[TModel, dict]:
    """Invoke the model and force a validated `schema` instance.

    Returns (parsed_model, usage_record). Raises StructuredOutputError if the
    model's output can't be coerced to the schema. Transparently retries on
    rate limits and falls back across models.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    def _call(model_name: str) -> tuple[TModel, dict]:
        structured = get_chat_model(model=model_name).with_structured_output(
            schema, include_raw=True
        )
        last_error: object = None
        for attempt in range(2):  # retry once on malformed (non-rate-limit) output
            result: dict = structured.invoke(messages)
            parsed = result.get("parsed")
            if parsed is not None:
                return parsed, _usage_from_raw(result.get("raw"), model_name)  # type: ignore[return-value]
            last_error = result.get("parsing_error")
            logger.warning(
                "structured output invalid; retrying",
                extra={"schema": schema.__name__, "attempt": attempt + 1},
            )
        raise StructuredOutputError(
            f"model did not return valid {schema.__name__} after retry: {last_error}"
        )

    return _run_resilient(_call)


def call_text(system_prompt: str, human_content: str) -> tuple[str, dict]:
    """Invoke the model for a plain-text (Markdown) response + usage."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    def _call(model_name: str) -> tuple[str, dict]:
        resp = get_chat_model(model=model_name).invoke(messages)
        content = resp.content
        text = content if isinstance(content, str) else str(content)
        return text, _usage_from_raw(resp, model_name)

    return _run_resilient(_call)
