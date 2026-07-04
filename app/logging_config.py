"""Structured (JSON) logging with a per-request correlation id.

The `request_id` is stored in a ContextVar so any log record emitted while
handling a request automatically carries it, without threading it through
every function call.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

# Correlation id for the in-flight request (or background task).
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_ctx.get(),
            "message": record.getMessage(),
        }
        # Attach any structured `extra={...}` fields the caller supplied.
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            if key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# Standard LogRecord attributes we do not want duplicated into the payload.
_RESERVED = frozenset(
    vars(logging.makeLogRecord({})).keys()
    | {"message", "asctime", "taskName"}
)


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger to emit JSON to stdout (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Quiet noisy third-party loggers a touch.
    logging.getLogger("uvicorn.access").handlers.clear()
