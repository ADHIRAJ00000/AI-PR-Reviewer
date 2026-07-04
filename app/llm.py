"""LLM factory + structured/text call helpers.

Central place that builds the chat model (provider factory) and runs calls with
structured output, capturing token usage for cost tracking. Agents call
`call_structured` / `call_text` and never touch the provider SDK directly.

Note on models: Claude Sonnet 5 / Opus 4.8 use adaptive thinking and reject
`temperature` / `top_p` / `budget_tokens`, so we never pass sampling params.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.config import ConfigError, get_settings
from app.observability.cost import usage_record

logger = logging.getLogger("app.llm")

TModel = TypeVar("TModel", bound=BaseModel)

# Bounded output so a runaway response can't hang the worker or blow cost.
DEFAULT_MAX_TOKENS = 8000
DEFAULT_TIMEOUT = 90.0


class StructuredOutputError(RuntimeError):
    """Raised when the model does not return a valid structured object."""


def get_chat_model(*, model: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS) -> BaseChatModel:
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
            max_retries=2,
        )
    raise ConfigError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER!r}")


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
    model's output can't be coerced to the schema.
    """
    settings = get_settings()
    model = get_chat_model()
    structured = model.with_structured_output(schema, include_raw=True)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]

    # Enforce the schema; retry once on malformed output before giving up.
    last_error: object = None
    for attempt in range(2):
        result: dict = structured.invoke(messages)
        parsed = result.get("parsed")
        if parsed is not None:
            usage = _usage_from_raw(result.get("raw"), settings.LLM_MODEL)
            return parsed, usage  # type: ignore[return-value]
        last_error = result.get("parsing_error")
        logger.warning(
            "structured output invalid; retrying",
            extra={"schema": schema.__name__, "attempt": attempt + 1},
        )

    raise StructuredOutputError(
        f"model did not return valid {schema.__name__} after retry: {last_error}"
    )


def call_text(system_prompt: str, human_content: str) -> tuple[str, dict]:
    """Invoke the model for a plain-text (Markdown) response + usage."""
    settings = get_settings()
    model = get_chat_model()
    resp = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
    )
    content = resp.content
    text = content if isinstance(content, str) else str(content)
    usage = _usage_from_raw(resp, settings.LLM_MODEL)
    return text, usage
