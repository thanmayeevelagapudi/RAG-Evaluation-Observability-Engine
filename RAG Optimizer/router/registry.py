"""Strategy performance registry — Postgres (asyncpg) + Redis cache.

Stores the latest windowed score and sample count per (strategy, query_type) pair, as written
by the feedback updater. Redis is the read path; Postgres is the source of truth.
"""

from __future__ import annotations

import json
import logging

import asyncpg

from config.settings import get_settings
from retrieval.base import QueryType

log = logging.getLogger(__name__)

# Cold-start priority ordering — drives pseudo-scores before real data accumulates.
# Derived from each strategy's supported_query_types defined in Phase 2.
COLD_START_DEFAULTS: dict[QueryType, list[str]] = {
    QueryType.FACTUAL:     ["bm25", "hybrid", "parent_child", "dense", "multi_query", "hyde"],
    QueryType.SEMANTIC:    ["dense", "hybrid", "hyde", "multi_query", "bm25", "parent_child"],
    QueryType.MULTI_HOP:   ["multi_query", "hybrid", "parent_child", "hyde", "dense", "bm25"],
    QueryType.ENTITY:      ["bm25", "dense", "hybrid", "parent_child", "multi_query", "hyde"],
    QueryType.ABSTRACTIVE: ["hyde", "multi_query", "hybrid", "dense", "bm25", "parent_child"],
    QueryType.UNKNOWN:     ["hybrid", "dense", "bm25", "multi_query", "hyde", "parent_child"],
}

_ALL_STRATEGIES = ["dense", "bm25", "hybrid", "hyde", "multi_query", "parent_child"]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS strategy_weights (
    id           SERIAL PRIMARY KEY,
    strategy     VARCHAR(50)  NOT NULL,
    query_type   VARCHAR(50)  NOT NULL,
    avg_score    FLOAT        NOT NULL DEFAULT 0.0,
    sample_count INT          NOT NULL DEFAULT 0,
    updated_at   TIMESTAMP    NOT NULL DEFAULT NOW(),
    UNIQUE (strategy, query_type)
);
"""

_SET_WEIGHT = """
INSERT INTO strategy_weights (strategy, query_type, avg_score, sample_count, updated_at)
VALUES ($1, $2, $3, $4, NOW())
ON CONFLICT (strategy, query_type) DO UPDATE SET
    avg_score    = EXCLUDED.avg_score,
    sample_count = EXCLUDED.sample_count,
    updated_at   = NOW();
"""


class StrategyRegistry:
    def __init__(self, db_pool: asyncpg.Pool | None = None, redis_client=None) -> None:
        self._pool = db_pool
        self._redis = redis_client
        self._settings = get_settings()

    async def initialise(self) -> None:
        """Create the connection pool and ensure the table exists. Call once at startup."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._settings.postgres_dsn)
        await self._pool.execute(_CREATE_TABLE)

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    async def get_scores(self, query_type: QueryType) -> dict[str, float]:
        scores, _ = await self.get_weights(query_type)
        return scores

    async def get_weights(
        self, query_type: QueryType
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Return (scores, sample_counts) for all strategies, Redis-cached."""
        cache_key = f"registry:scores:{query_type.value}"

        if self._redis is not None:
            cached = await self._redis.get(cache_key)
            if cached is not None:
                data = json.loads(cached)
                return data["scores"], data["counts"]

        scores, counts = await self._fetch_from_db(query_type)

        if self._redis is not None:
            payload = json.dumps({"scores": scores, "counts": counts})
            await self._redis.setex(cache_key, self._settings.registry_cache_ttl, payload)

        return scores, counts

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    async def set_score(
        self, strategy: str, query_type: QueryType, score: float, sample_count: int
    ) -> None:
        """Overwrite the windowed aggregate for (strategy, query_type) and bust the Redis cache.

        Idempotent, so repeated cycles or several replicas cannot inflate sample_count.
        """
        if self._pool is None:
            log.warning("No DB pool — score update for %s/%s skipped", strategy, query_type)
            return
        await self._pool.execute(_SET_WEIGHT, strategy, query_type.value, score, sample_count)
        if self._redis is not None:
            await self._redis.delete(f"registry:scores:{query_type.value}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _fetch_from_db(
        self, query_type: QueryType
    ) -> tuple[dict[str, float], dict[str, int]]:
        if self._pool is None:
            return self._cold_start_scores(query_type)

        rows = await self._pool.fetch(
            "SELECT strategy, avg_score, sample_count FROM strategy_weights "
            "WHERE query_type = $1",
            query_type.value,
        )
        db_scores = {r["strategy"]: r["avg_score"] for r in rows}
        db_counts = {r["strategy"]: r["sample_count"] for r in rows}

        cold_order = COLD_START_DEFAULTS.get(
            query_type, COLD_START_DEFAULTS[QueryType.UNKNOWN]
        )
        n = len(_ALL_STRATEGIES)
        min_s = self._settings.bandit_min_samples

        scores: dict[str, float] = {}
        counts: dict[str, int] = {}
        for strategy in _ALL_STRATEGIES:
            count = db_counts.get(strategy, 0)
            counts[strategy] = count
            if count < min_s:
                # Position-based pseudo-score so exploitation respects cold-start priors
                pos = cold_order.index(strategy) if strategy in cold_order else n
                scores[strategy] = (n - pos) / n
            else:
                scores[strategy] = db_scores.get(strategy, 0.0)

        return scores, counts

    def _cold_start_scores(
        self, query_type: QueryType
    ) -> tuple[dict[str, float], dict[str, int]]:
        cold_order = COLD_START_DEFAULTS.get(
            query_type, COLD_START_DEFAULTS[QueryType.UNKNOWN]
        )
        n = len(_ALL_STRATEGIES)
        scores = {
            s: (n - cold_order.index(s)) / n if s in cold_order else 0.0
            for s in _ALL_STRATEGIES
        }
        return scores, {s: 0 for s in _ALL_STRATEGIES}
