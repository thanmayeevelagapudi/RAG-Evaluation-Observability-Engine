"""Feedback collector — writes signals to a Redis FIFO buffer.

Hot path: record_explicit / record_implicit are called at query time and must
return immediately. The updater drains the buffer asynchronously every 60 s.
No synchronous Postgres write in the hot path.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from retrieval.base import QueryType

_BUFFER_KEY = "feedback:buffer"


class FeedbackCollector:
    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    async def record_explicit(
        self,
        query: str,
        strategy: str,
        query_type: QueryType,
        positive: bool,
        was_exploration: bool,
        session_id: str | None = None,
    ) -> None:
        """Record a thumbs-up (positive=True) or thumbs-down (positive=False) signal."""
        await self._push({
            "query":           query,
            "strategy":        strategy,
            "query_type":      query_type.value,
            "signal_type":     "explicit",
            "signal_value":    1.0 if positive else 0.0,
            "was_exploration": was_exploration,
            "session_id":      session_id,
            "created_at":      datetime.now(UTC).isoformat(),
        })

    async def record_implicit(
        self,
        query: str,
        strategy: str,
        query_type: QueryType,
        signal_value: float,
        was_exploration: bool,
        session_id: str | None = None,
    ) -> None:
        """Record an inferred signal (0.0–1.0) from user behaviour."""
        await self._push({
            "query":           query,
            "strategy":        strategy,
            "query_type":      query_type.value,
            "signal_type":     "implicit",
            "signal_value":    max(0.0, min(1.0, signal_value)),
            "was_exploration": was_exploration,
            "session_id":      session_id,
            "created_at":      datetime.now(UTC).isoformat(),
        })

    async def _push(self, event: dict) -> None:
        await self._redis.rpush(_BUFFER_KEY, json.dumps(event))
