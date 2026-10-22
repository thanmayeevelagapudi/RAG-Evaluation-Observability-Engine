"""Feedback updater — the closed loop that makes the router self-improve.

Every 60 s:
  1. Drain Redis buffer → Postgres feedback_events
  2. Aggregate ragas_scores + judge_scores + feedback_events (7-day window)
  3. Compute combined score per (strategy, query_type)
  4. registry.set_score() → strategy_weights table (window aggregate + real sample count)
     → bandit adapts next query

was_exploration dampening: exploration runs receive 50 % weight so a single
bad exploratory result doesn't permanently tank a strategy's standing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import asyncpg

from retrieval.base import QueryType
from router.registry import StrategyRegistry

log = logging.getLogger(__name__)

_BUFFER_KEY = "feedback:buffer"
_MAX_FLUSH_PER_CYCLE = 10_000  # prevents endless lpop loops if buffer grew while updater was down

_INSERT_FEEDBACK = """
INSERT INTO feedback_events
    (query, strategy, query_type, signal_type, signal_value, was_exploration, session_id, created_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
"""

_RAGAS_AGG = """
SELECT strategy, query_type,
       AVG(composite)                                          AS ragas_avg,
       COUNT(*)                                                AS total_count,
       COUNT(CASE WHEN was_exploration THEN 1 END)::FLOAT      AS explore_count
FROM ragas_scores
WHERE created_at > $1
GROUP BY strategy, query_type
"""

_JUDGE_AGG = """
SELECT strategy, query_type,
       AVG(composite) AS judge_avg,
       COUNT(*)       AS n
FROM judge_scores
WHERE created_at > $1
GROUP BY strategy, query_type
"""

_FEEDBACK_AGG = """
SELECT strategy, query_type,
       AVG(signal_value) AS feedback_avg,
       COUNT(*)          AS n
FROM feedback_events
WHERE created_at > $1
GROUP BY strategy, query_type
"""


@dataclass
class AggregatedScore:
    strategy:             str
    query_type:           str
    ragas_avg:            float | None = None
    judge_avg:            float | None = None
    feedback_avg:         float | None = None
    was_exploration_ratio: float = 0.0
    sample_count:         int = 0


@dataclass
class UpdateResult:
    flushed:   int
    updated:   int
    timestamp: datetime
    errors:    list[str] = field(default_factory=list)


class FeedbackUpdater:
    def __init__(
        self,
        db_pool: asyncpg.Pool | None,
        redis_client,
        registry: StrategyRegistry,
        metrics=None,  # optional PrometheusMetrics — avoids circular import
    ) -> None:
        self._pool = db_pool
        self._redis = redis_client
        self._registry = registry
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_once(self) -> UpdateResult:
        """One full update cycle: flush → aggregate → update weights."""
        errors: list[str] = []
        flushed = updated = 0

        try:
            flushed = await self._flush_buffer()
        except Exception as exc:
            errors.append(f"flush: {exc}")
            log.error("Buffer flush failed: %s", exc)

        try:
            scores = await self._aggregate_scores()
            updated = await self._update_weights(scores)
        except Exception as exc:
            errors.append(f"update: {exc}")
            log.error("Weight update failed: %s", exc)

        return UpdateResult(flushed=flushed, updated=updated, timestamp=datetime.now(UTC), errors=errors)

    async def start_background_loop(self, interval_seconds: int = 60) -> None:
        """Run update cycle on a fixed schedule. Call from FastAPI lifespan."""
        while True:
            try:
                result = await self.run_once()
                log.info(
                    "Feedback update: flushed=%d updated=%d errors=%d",
                    result.flushed, result.updated, len(result.errors),
                )
            except Exception as exc:
                log.error("Feedback updater loop error: %s", exc)
            await asyncio.sleep(interval_seconds)

    # ------------------------------------------------------------------
    # Step 1 — flush Redis buffer
    # ------------------------------------------------------------------

    async def _flush_buffer(self) -> int:
        if self._pool is None:
            # No DB available — do NOT pop events (they would be silently lost)
            return 0

        events: list[dict] = []
        for _ in range(_MAX_FLUSH_PER_CYCLE):
            raw = await self._redis.lpop(_BUFFER_KEY)
            if raw is None:
                break
            events.append(json.loads(raw))

        if not events:
            return 0

        rows = [_event_to_row(e) for e in events]
        await self._pool.executemany(_INSERT_FEEDBACK, rows)
        return len(events)

    # ------------------------------------------------------------------
    # Step 2 — aggregate scores from all three tables
    # ------------------------------------------------------------------

    async def _aggregate_scores(
        self, lookback_days: int = 7
    ) -> dict[tuple[str, str], AggregatedScore]:
        if self._pool is None:
            return {}

        # Naive UTC datetime: asyncpg sends as OID 1114 (timestamp without time zone)
        # which matches the TIMESTAMP column type in ragas_scores / judge_scores / feedback_events
        cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=lookback_days)
        merged: dict[tuple[str, str], AggregatedScore] = {}

        for row in await self._pool.fetch(_RAGAS_AGG, cutoff):
            key = (row["strategy"], row["query_type"])
            ratio = row["explore_count"] / row["total_count"] if row["total_count"] else 0.0
            merged[key] = AggregatedScore(
                strategy=row["strategy"],
                query_type=row["query_type"],
                ragas_avg=row["ragas_avg"],
                was_exploration_ratio=ratio,
                sample_count=int(row["total_count"]),
            )

        for row in await self._pool.fetch(_JUDGE_AGG, cutoff):
            key = (row["strategy"], row["query_type"])
            n = int(row["n"])
            if key in merged:
                merged[key].judge_avg = row["judge_avg"]
                merged[key].sample_count = max(merged[key].sample_count, n)
            else:
                merged[key] = AggregatedScore(
                    strategy=row["strategy"],
                    query_type=row["query_type"],
                    judge_avg=row["judge_avg"],
                    sample_count=n,
                )

        for row in await self._pool.fetch(_FEEDBACK_AGG, cutoff):
            key = (row["strategy"], row["query_type"])
            n = int(row["n"])
            if key in merged:
                merged[key].feedback_avg = row["feedback_avg"]
                merged[key].sample_count = max(merged[key].sample_count, n)
            else:
                merged[key] = AggregatedScore(
                    strategy=row["strategy"],
                    query_type=row["query_type"],
                    feedback_avg=row["feedback_avg"],
                    sample_count=n,
                )

        return merged

    # ------------------------------------------------------------------
    # Step 3 — compute combined score + update registry
    # ------------------------------------------------------------------

    async def _update_weights(
        self, scores: dict[tuple[str, str], AggregatedScore]
    ) -> int:
        updated = 0
        for (strategy, qt_str), agg in scores.items():
            combined = self._compute_combined_score(
                ragas_avg=agg.ragas_avg,
                judge_avg=agg.judge_avg,
                feedback_avg=agg.feedback_avg,
                was_exploration_ratio=agg.was_exploration_ratio,
            )
            if combined <= 0.0:
                continue
            try:
                qt = QueryType(qt_str)
            except ValueError:
                log.warning("Unknown query_type '%s' in aggregated scores — skipped", qt_str)
                continue
            await self._registry.set_score(strategy, qt, combined, agg.sample_count)
            if self._metrics is not None:
                self._metrics.update_strategy_score(strategy, qt_str, combined)
            updated += 1
        return updated

    def _compute_combined_score(
        self,
        ragas_avg: float | None,
        judge_avg: float | None,
        feedback_avg: float | None,
        was_exploration_ratio: float,
    ) -> float:
        """Weighted average of three signal sources with exploration dampening.

        Weights: RAGAS 0.50 · judge 0.30 · feedback 0.20
        Missing signals are excluded and remaining weights re-normalised to 1.0.
        Exploration dampening: RAGAS contribution scaled by (1 − 0.5 × ratio)
        so bad luck during exploration doesn't permanently tank a strategy.
        """
        scores: list[float] = []
        weights: list[float] = []

        if ragas_avg is not None:
            effective = ragas_avg * (1.0 - 0.5 * was_exploration_ratio)
            scores.append(effective)
            weights.append(0.50)

        if judge_avg is not None:
            scores.append(judge_avg)
            weights.append(0.30)

        if feedback_avg is not None:
            scores.append(feedback_avg)
            weights.append(0.20)

        if not scores:
            return 0.0

        total_weight = sum(weights)
        return sum(s * w / total_weight for s, w in zip(scores, weights))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event_to_row(e: dict) -> tuple:
    """Convert a JSON event dict to a positional tuple for asyncpg executemany."""
    created_at_str = e.get("created_at")
    try:
        created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(UTC)
    except ValueError:
        created_at = datetime.now(UTC)
    # asyncpg rejects tz-aware datetimes for TIMESTAMP (without time zone) columns.
    if created_at.tzinfo is not None:
        created_at = created_at.astimezone(UTC).replace(tzinfo=None)
    return (
        e["query"],
        e["strategy"],
        e["query_type"],
        e["signal_type"],
        float(e["signal_value"]),
        bool(e["was_exploration"]),
        e.get("session_id"),
        created_at,
    )
