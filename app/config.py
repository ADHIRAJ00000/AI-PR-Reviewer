"""Application configuration.

Loads and validates all environment variables through a single Pydantic
`Settings` object. Required variables that are missing produce a clear,
human-readable error at startup instead of a deep stack trace.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Literal

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from env / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ---- GitHub (required) ----
    GITHUB_TOKEN: str = Field(..., min_length=1, description="GitHub API token.")
    GITHUB_WEBHOOK_SECRET: str = Field(
        ..., min_length=1, description="Secret used to verify webhook signatures."
    )

    # ---- LLM (required) ----
    LLM_API_KEY: str = Field(..., min_length=1, description="LLM provider API key.")
    LLM_PROVIDER: Literal["anthropic", "groq"] = "anthropic"
    LLM_MODEL: str = "claude-sonnet-5"

    # ---- Langfuse (optional; required from Phase 8) ----
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # ---- Redis (optional) ----
    REDIS_URL: str = "redis://localhost:6379/0"

    # ---- App ----
    LOG_LEVEL: str = "INFO"

    @property
    def langfuse_enabled(self) -> bool:
        """True only when both Langfuse keys are present."""
        return bool(self.LANGFUSE_PUBLIC_KEY and self.LANGFUSE_SECRET_KEY)


class ConfigError(RuntimeError):
    """Raised when configuration is invalid or incomplete."""


def _format_validation_error(exc: ValidationError) -> str:
    """Turn a Pydantic ValidationError into a readable multi-line message."""
    lines = ["Configuration error — the app cannot start:"]
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err["loc"])
        if err["type"] == "missing":
            lines.append(f"  - Missing required environment variable: {field}")
        else:
            lines.append(f"  - {field}: {err['msg']}")
    lines.append("")
    lines.append("Fix: copy `.env.example` to `.env` and fill in the values above.")
    return "\n".join(lines)


@lru_cache
def get_settings() -> Settings:
    """Return the singleton Settings, failing loudly on invalid config."""
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        message = _format_validation_error(exc)
        # Print to stderr so it is visible even before logging is configured.
        print(message, file=sys.stderr)
        raise ConfigError(message) from exc
