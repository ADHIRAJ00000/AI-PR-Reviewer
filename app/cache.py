"""Redis-backed dedup so the same PR head SHA isn't reviewed twice.

Fail-open: if Redis is unreachable, we log and allow the review to proceed
rather than blocking the pipeline on a cache outage.
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis
from redis.exceptions import RedisError

from app.config import get_settings

logger = logging.getLogger("app.cache")

# Keep dedup keys for a day — long enough to absorb webhook retries.
CLAIM_TTL_SECONDS = 86_400

_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Lazily build a shared async Redis client."""
    global _client
    if _client is None:
        _client = aioredis.from_url(
            get_settings().REDIS_URL, decode_responses=True
        )
    return _client


async def claim_review(key: str) -> bool:
    """Atomically claim a review slot.

    Returns True if this is the first claim (proceed), False if already claimed
    (duplicate). On Redis errors, fails open (returns True) so reviews still run.
    """
    try:
        # SET key 1 NX EX ttl → returns True only if the key was newly set.
        was_set = await get_redis().set(key, "1", nx=True, ex=CLAIM_TTL_SECONDS)
        return bool(was_set)
    except RedisError as exc:
        logger.warning("redis unavailable; skipping dedup", extra={"error": str(exc)})
        return True


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
