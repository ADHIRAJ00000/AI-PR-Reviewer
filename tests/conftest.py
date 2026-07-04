"""Shared test fixtures — notably a fake chat model so agents run without a key.

`fake_llm` monkeypatches `app.llm.get_chat_model` to return a FakeChatModel that
mimics `with_structured_output(schema, include_raw=True)` and `invoke(...)`,
including token `usage_metadata`, so the whole graph runs offline.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

DEFAULT_USAGE = {"input_tokens": 120, "output_tokens": 60, "total_tokens": 180}


class _FakeStructuredRunnable:
    """Stands in for model.with_structured_output(schema, include_raw=True)."""

    def __init__(self, schema: type[BaseModel], usage: dict, fail: bool):
        self._schema = schema
        self._usage = usage
        self._fail = fail

    def invoke(self, _messages):
        if self._fail:
            raise RuntimeError("simulated structured-output failure")
        # All aggregate schemas have all-default fields → valid empty instance.
        return {
            "raw": AIMessage(content="", usage_metadata=self._usage),
            "parsed": self._schema(),
            "parsing_error": None,
        }


class FakeChatModel:
    """Minimal chat model: schema-aware structured output + text invoke."""

    def __init__(
        self,
        *,
        text: str = "## Review\n\nLGTM — no blocking issues.",
        fail_schemas: set[str] | None = None,
        fail_text: bool = False,
        usage: dict | None = None,
    ):
        self._text = text
        self._fail_schemas = fail_schemas or set()
        self._fail_text = fail_text
        self._usage = usage or DEFAULT_USAGE

    def with_structured_output(self, schema: type[BaseModel], include_raw: bool = False):
        return _FakeStructuredRunnable(
            schema, self._usage, fail=schema.__name__ in self._fail_schemas
        )

    def invoke(self, _messages):
        if self._fail_text:
            raise RuntimeError("simulated text-generation failure")
        return AIMessage(content=self._text, usage_metadata=self._usage)


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a FakeChatModel; returns a callable to customise failure modes."""

    def _install(**kwargs) -> FakeChatModel:
        model = FakeChatModel(**kwargs)
        monkeypatch.setattr("app.llm.get_chat_model", lambda *a, **k: model)
        return model

    return _install
