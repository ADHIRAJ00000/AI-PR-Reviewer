"""Token → USD cost accounting.

Pricing is per 1M tokens (input, output), sourced from Anthropic's published
rates. Sonnet 5 has introductory pricing through 2026-08-31; we use the standard
sticker rate here to avoid over-reporting savings. Unknown models fall back to
the configured default's tier.
"""

from __future__ import annotations

# (input $/1M, output $/1M)
PRICING: dict[str, tuple[float, float]] = {
    # --- Anthropic ---
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # --- Groq (approx on-demand rates; free tier available) ---
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "moonshotai/kimi-k2-instruct": (1.0, 3.0),
    "qwen/qwen3-32b": (0.29, 0.59),
}

_FALLBACK = (3.0, 15.0)  # sonnet-tier


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a single call, rounded to 6 dp."""
    inp_rate, out_rate = PRICING.get(model, _FALLBACK)
    cost = (input_tokens / 1_000_000) * inp_rate + (output_tokens / 1_000_000) * out_rate
    return round(cost, 6)


def usage_record(model: str, input_tokens: int, output_tokens: int) -> dict:
    """Build a per-agent token/cost record for `state['token_usage']`."""
    return {
        "prompt": input_tokens,
        "completion": output_tokens,
        "total": input_tokens + output_tokens,
        "cost_usd": compute_cost(model, input_tokens, output_tokens),
    }


def empty_usage() -> dict:
    """Zeroed record used when an agent fails before any LLM call succeeds."""
    return {"prompt": 0, "completion": 0, "total": 0, "cost_usd": 0.0}
