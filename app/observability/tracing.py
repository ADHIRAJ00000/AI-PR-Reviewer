"""Langfuse tracing setup.

Each PR review becomes one trace; every LLM call inside the graph becomes a
child span (LangGraph propagates the callback into each node's model call). If
Langfuse keys aren't configured, tracing is a no-op — the pipeline runs
unchanged. Token/cost accounting works independently via `state['token_usage']`.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger("app.observability.tracing")


def _build_handler() -> Any | None:
    """Create a Langfuse LangChain callback handler, or None if disabled."""
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    except Exception as exc:  # noqa: BLE001 - never let tracing break the review
        logger.warning("langfuse init failed; tracing disabled", extra={"error": str(exc)})
        return None


def build_trace(run_name: str, metadata: dict | None = None) -> tuple[dict, Any | None]:
    """Return (runnable_config, handler).

    Pass `config` to `graph.ainvoke(state, config=config)`. Call `handler.flush()`
    when done (via `flush(handler)`) so spans are sent before the process idles.
    """
    handler = _build_handler()
    config: dict = {"run_name": run_name}
    if metadata:
        config["metadata"] = metadata
    if handler is not None:
        config["callbacks"] = [handler]
    return config, handler


def flush(handler: Any | None) -> None:
    """Flush buffered spans to Langfuse (no-op if disabled)."""
    if handler is None:
        return
    try:
        handler.flush()
    except Exception as exc:  # noqa: BLE001
        logger.warning("langfuse flush failed", extra={"error": str(exc)})
