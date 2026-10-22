"""Query intent classifier — maps raw query strings to QueryType via Claude + Redis cache.

Classification adds ~300-500 ms latency; Redis caching on exact query hash eliminates
repeat cost within the TTL window.
"""

from __future__ import annotations

import hashlib
import logging

import anthropic

from config.settings import get_settings
from retrieval.base import QueryType

log = logging.getLogger(__name__)

_CLASSIFY_PROMPT = """\
You are a query intent classifier for a retrieval system.
Classify the query into exactly one of these categories:

FACTUAL     - Has a specific, verifiable answer. "What year was X founded?"
SEMANTIC    - Requires conceptual understanding. "What does async programming feel like?"
MULTI_HOP   - Requires connecting multiple facts. "Which companies that X acquired also partner with Y?"
ENTITY      - About a specific named entity. "Who is the CEO of Stripe?"
ABSTRACTIVE - Requires synthesis across sources. "Summarise the key arguments about X"

Examples:
Query: "When was Python first released?" → FACTUAL
Query: "Why do people find functional programming elegant?" → SEMANTIC
Query: "Which papers cited by Attention Is All You Need also influenced BERT?" → MULTI_HOP
Query: "What is Anthropic's mission?" → ENTITY
Query: "What are the main schools of thought on inflation?" → ABSTRACTIVE

Classify this query. Reply with ONLY the category name, nothing else.
Query: "{query}"\
"""

_MODEL = "claude-3-5-haiku-20241022"

_LABEL_MAP: dict[str, QueryType] = {
    "FACTUAL": QueryType.FACTUAL,
    "SEMANTIC": QueryType.SEMANTIC,
    "MULTI_HOP": QueryType.MULTI_HOP,
    "ENTITY": QueryType.ENTITY,
    "ABSTRACTIVE": QueryType.ABSTRACTIVE,
}


class QueryClassifier:
    def __init__(
        self,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
        redis_client=None,
    ) -> None:
        self._client = anthropic_client or anthropic.AsyncAnthropic()
        self._redis = redis_client
        self._settings = get_settings()

    async def classify(self, query: str) -> QueryType:
        cache_key = f"classifier:{hashlib.md5(query.encode()).hexdigest()}"

        if self._redis is not None:
            cached = await self._redis.get(cache_key)
            if cached is not None:
                try:
                    return QueryType(cached.decode())
                except ValueError:
                    pass  # stale/corrupt cache entry — fall through to API call

        try:
            result = await self._call_claude(query)
        except Exception as exc:
            log.warning("Classifier API call failed (%s); returning UNKNOWN", exc)
            return QueryType.UNKNOWN

        if self._redis is not None:
            await self._redis.setex(
                cache_key, self._settings.classifier_cache_ttl, result.value
            )

        return result

    async def _call_claude(self, query: str) -> QueryType:
        message = await self._client.messages.create(
            model=_MODEL,
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(query=query)}],
        )
        raw = message.content[0].text.strip().upper()
        # Default to SEMANTIC for unrecognized labels — better than UNKNOWN for routing
        return _LABEL_MAP.get(raw, QueryType.SEMANTIC)
