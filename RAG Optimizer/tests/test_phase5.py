"""Phase 5 unit tests — FeedbackCollector, FeedbackUpdater, _compute_combined_score.

All Postgres and Redis calls are mocked. No live infrastructure required.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from feedback.collector import FeedbackCollector
from feedback.updater import (
    AggregatedScore,
    FeedbackUpdater,
    UpdateResult,
    _event_to_row,
)
from retrieval.base import QueryType
from router.registry import StrategyRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_redis(buffer: list[bytes | None] | None = None) -> AsyncMock:
    """Mock Redis whose lpop returns each item in buffer then None forever."""
    redis = AsyncMock()
    if buffer is None:
        # Empty buffer by default
        redis.lpop = AsyncMock(return_value=None)
    else:
        # Exhaust the buffer then return None
        redis.lpop = AsyncMock(side_effect=buffer + [None])
    redis.rpush = AsyncMock(return_value=1)
    redis.llen = AsyncMock(return_value=0)
    return redis


def _make_pool(fetch_side_effect: list | None = None) -> AsyncMock:
    pool = AsyncMock()
    pool.executemany = AsyncMock()
    if fetch_side_effect is not None:
        pool.fetch = AsyncMock(side_effect=fetch_side_effect)
    else:
        pool.fetch = AsyncMock(return_value=[])
    return pool


def _make_registry() -> AsyncMock:
    reg = AsyncMock(spec=StrategyRegistry)
    reg.set_score = AsyncMock()
    return reg


def _json_event(strategy="dense", query_type="semantic", signal_value=0.8) -> bytes:
    return json.dumps({
        "query": "test query",
        "strategy": strategy,
        "query_type": query_type,
        "signal_type": "explicit",
        "signal_value": signal_value,
        "was_exploration": False,
        "session_id": None,
        "created_at": datetime.now(UTC).isoformat(),
    }).encode()


# ---------------------------------------------------------------------------
# FeedbackCollector
# ---------------------------------------------------------------------------

class TestFeedbackCollector:
    @pytest.mark.asyncio
    async def test_record_explicit_positive_pushes_1_0(self):
        redis = _make_redis()
        collector = FeedbackCollector(redis_client=redis)
        await collector.record_explicit("q", "hybrid", QueryType.SEMANTIC, positive=True, was_exploration=False)
        redis.rpush.assert_called_once()
        event = json.loads(redis.rpush.call_args[0][1])
        assert event["signal_value"] == 1.0
        assert event["signal_type"] == "explicit"

    @pytest.mark.asyncio
    async def test_record_explicit_negative_pushes_0_0(self):
        redis = _make_redis()
        collector = FeedbackCollector(redis_client=redis)
        await collector.record_explicit("q", "bm25", QueryType.FACTUAL, positive=False, was_exploration=True)
        event = json.loads(redis.rpush.call_args[0][1])
        assert event["signal_value"] == 0.0

    @pytest.mark.asyncio
    async def test_record_implicit_pushes_clamped_value(self):
        redis = _make_redis()
        collector = FeedbackCollector(redis_client=redis)
        await collector.record_implicit("q", "dense", QueryType.SEMANTIC, signal_value=0.7, was_exploration=False)
        event = json.loads(redis.rpush.call_args[0][1])
        assert event["signal_value"] == pytest.approx(0.7)
        assert event["signal_type"] == "implicit"

    @pytest.mark.asyncio
    async def test_record_implicit_clamps_above_1(self):
        redis = _make_redis()
        collector = FeedbackCollector(redis_client=redis)
        await collector.record_implicit("q", "dense", QueryType.SEMANTIC, signal_value=1.5, was_exploration=False)
        event = json.loads(redis.rpush.call_args[0][1])
        assert event["signal_value"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_record_stores_was_exploration_flag(self):
        redis = _make_redis()
        collector = FeedbackCollector(redis_client=redis)
        await collector.record_explicit("q", "hyde", QueryType.ABSTRACTIVE, positive=True, was_exploration=True)
        event = json.loads(redis.rpush.call_args[0][1])
        assert event["was_exploration"] is True

    @pytest.mark.asyncio
    async def test_each_call_increments_buffer(self):
        redis = _make_redis()
        collector = FeedbackCollector(redis_client=redis)
        await collector.record_explicit("q1", "dense", QueryType.SEMANTIC, True, False)
        await collector.record_explicit("q2", "bm25",  QueryType.FACTUAL,  True, False)
        assert redis.rpush.call_count == 2


# ---------------------------------------------------------------------------
# FeedbackUpdater — _flush_buffer
# ---------------------------------------------------------------------------

class TestFlushBuffer:
    @pytest.mark.asyncio
    async def test_returns_zero_on_empty_redis(self):
        updater = FeedbackUpdater(
            db_pool=_make_pool(), redis_client=_make_redis(), registry=_make_registry()
        )
        count = await updater._flush_buffer()
        assert count == 0

    @pytest.mark.asyncio
    async def test_pops_all_events_and_returns_count(self):
        events = [_json_event(), _json_event(), _json_event()]
        redis = _make_redis(buffer=events)
        pool = _make_pool()
        updater = FeedbackUpdater(db_pool=pool, redis_client=redis, registry=_make_registry())
        count = await updater._flush_buffer()
        assert count == 3

    @pytest.mark.asyncio
    async def test_bulk_inserts_to_db(self):
        events = [_json_event(strategy="dense"), _json_event(strategy="bm25")]
        redis = _make_redis(buffer=events)
        pool = _make_pool()
        updater = FeedbackUpdater(db_pool=pool, redis_client=redis, registry=_make_registry())
        await updater._flush_buffer()
        pool.executemany.assert_called_once()
        rows = pool.executemany.call_args[0][1]
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_no_pool_skips_pop_to_prevent_data_loss(self):
        """Events must NOT be popped when DB is unavailable — popping without writing = data loss."""
        events = [_json_event()]
        redis = _make_redis(buffer=events)
        updater = FeedbackUpdater(db_pool=None, redis_client=redis, registry=_make_registry())
        count = await updater._flush_buffer()
        assert count == 0               # nothing flushed
        redis.lpop.assert_not_called()  # Redis queue untouched

    @pytest.mark.asyncio
    async def test_flush_stops_at_batch_cap(self):
        """_flush_buffer must not loop indefinitely — it stops at _MAX_FLUSH_PER_CYCLE."""
        from feedback.updater import _MAX_FLUSH_PER_CYCLE
        # Simulate buffer larger than cap: cap+5 events then None
        events = [_json_event() for _ in range(_MAX_FLUSH_PER_CYCLE + 5)]
        redis = _make_redis(buffer=events)
        pool = _make_pool()
        updater = FeedbackUpdater(db_pool=pool, redis_client=redis, registry=_make_registry())
        count = await updater._flush_buffer()
        assert count == _MAX_FLUSH_PER_CYCLE  # capped, not over

    @pytest.mark.asyncio
    async def test_no_pool_skips_pop_entirely(self):
        """Events must NOT be popped from Redis when DB is unavailable (data loss prevention)."""
        events = [_json_event(), _json_event()]
        redis = _make_redis(buffer=events)
        updater = FeedbackUpdater(db_pool=None, redis_client=redis, registry=_make_registry())
        count = await updater._flush_buffer()
        assert count == 0                   # nothing processed
        redis.lpop.assert_not_called()      # Redis never touched


# ---------------------------------------------------------------------------
# FeedbackUpdater — _compute_combined_score
# ---------------------------------------------------------------------------

class TestComputeCombinedScore:
    def _updater(self) -> FeedbackUpdater:
        return FeedbackUpdater(db_pool=None, redis_client=AsyncMock(), registry=_make_registry())

    def test_only_ragas_normalises_to_full_weight(self):
        updater = self._updater()
        score = updater._compute_combined_score(
            ragas_avg=0.7, judge_avg=None, feedback_avg=None, was_exploration_ratio=0.0
        )
        assert score == pytest.approx(0.7)

    def test_exploration_dampening_at_ratio_1(self):
        updater = self._updater()
        # ragas_avg=0.8, exploration_ratio=1.0 → effective=0.8*(1-0.5)=0.4
        score = updater._compute_combined_score(
            ragas_avg=0.8, judge_avg=None, feedback_avg=None, was_exploration_ratio=1.0
        )
        assert score == pytest.approx(0.4)

    def test_exploration_dampening_partial(self):
        updater = self._updater()
        # exploration_ratio=0.5 → effective=0.8*(1-0.25)=0.6
        score = updater._compute_combined_score(
            ragas_avg=0.8, judge_avg=None, feedback_avg=None, was_exploration_ratio=0.5
        )
        assert score == pytest.approx(0.6)

    def test_all_three_signals_correct_weighted_average(self):
        updater = self._updater()
        # (0.8*0.5 + 0.7*0.3 + 0.6*0.2) / 1.0 = (0.40 + 0.21 + 0.12) = 0.73
        score = updater._compute_combined_score(
            ragas_avg=0.8, judge_avg=0.7, feedback_avg=0.6, was_exploration_ratio=0.0
        )
        assert score == pytest.approx(0.73)

    def test_missing_judge_renormalises_weights(self):
        updater = self._updater()
        # weights: ragas=0.5, feedback=0.2, total=0.7
        # score = (0.8*0.5 + 0.6*0.2) / 0.7 = (0.4 + 0.12) / 0.7 ≈ 0.743
        score = updater._compute_combined_score(
            ragas_avg=0.8, judge_avg=None, feedback_avg=0.6, was_exploration_ratio=0.0
        )
        expected = (0.8 * 0.5 + 0.6 * 0.2) / 0.7
        assert score == pytest.approx(expected, rel=1e-3)

    def test_all_none_returns_zero(self):
        updater = self._updater()
        score = updater._compute_combined_score(
            ragas_avg=None, judge_avg=None, feedback_avg=None, was_exploration_ratio=0.0
        )
        assert score == 0.0

    def test_exploration_scores_contribute_less_than_exploit(self):
        updater = self._updater()
        exploit = updater._compute_combined_score(0.8, None, None, was_exploration_ratio=0.0)
        explore  = updater._compute_combined_score(0.8, None, None, was_exploration_ratio=1.0)
        assert explore < exploit


# ---------------------------------------------------------------------------
# FeedbackUpdater — run_once
# ---------------------------------------------------------------------------

class TestRunOnce:
    @pytest.mark.asyncio
    async def test_returns_update_result(self):
        redis = _make_redis(buffer=[_json_event()])
        pool = _make_pool(fetch_side_effect=[[], [], []])
        updater = FeedbackUpdater(db_pool=pool, redis_client=redis, registry=_make_registry())
        result = await updater.run_once()
        assert isinstance(result, UpdateResult)

    @pytest.mark.asyncio
    async def test_flushed_count_matches_buffer_size(self):
        events = [_json_event(), _json_event()]
        redis = _make_redis(buffer=events)
        pool = _make_pool(fetch_side_effect=[[], [], []])
        updater = FeedbackUpdater(db_pool=pool, redis_client=redis, registry=_make_registry())
        result = await updater.run_once()
        assert result.flushed == 2

    @pytest.mark.asyncio
    async def test_errors_list_empty_on_clean_run(self):
        redis = _make_redis()
        pool = _make_pool(fetch_side_effect=[[], [], []])
        updater = FeedbackUpdater(db_pool=pool, redis_client=redis, registry=_make_registry())
        result = await updater.run_once()
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_flush_error_captured_not_raised(self):
        redis = AsyncMock()
        redis.lpop = AsyncMock(side_effect=RuntimeError("Redis down"))
        pool = _make_pool(fetch_side_effect=[[], [], []])
        updater = FeedbackUpdater(db_pool=pool, redis_client=redis, registry=_make_registry())
        result = await updater.run_once()
        assert result.errors
        assert "flush" in result.errors[0]

    @pytest.mark.asyncio
    async def test_registry_updated_when_aggregation_has_scores(self):
        redis = _make_redis()
        ragas_row = MagicMock()
        ragas_row.__getitem__ = lambda self, k: {
            "strategy": "dense", "query_type": "semantic",
            "ragas_avg": 0.75, "total_count": 10.0, "explore_count": 2.0
        }[k]
        pool = _make_pool(fetch_side_effect=[[ragas_row], [], []])
        registry = _make_registry()
        updater = FeedbackUpdater(db_pool=pool, redis_client=redis, registry=registry)
        result = await updater.run_once()
        assert result.updated == 1
        registry.set_score.assert_called_once_with(
            "dense", QueryType.SEMANTIC, pytest.approx(0.75 * (1 - 0.5 * 0.2), rel=1e-2), 10
        )

    @pytest.mark.asyncio
    async def test_repeated_cycles_write_the_same_window_count(self):
        # Regression: counts used to grow by one per 60 s cycle, not per evaluated query.
        ragas_row = {
            "strategy": "dense", "query_type": "semantic",
            "ragas_avg": 0.75, "total_count": 3, "explore_count": 0,
        }
        pool = _make_pool(fetch_side_effect=[[ragas_row], [], []] * 2)
        registry = _make_registry()
        updater = FeedbackUpdater(db_pool=pool, redis_client=_make_redis(), registry=registry)
        await updater.run_once()
        await updater.run_once()
        assert [c.args[3] for c in registry.set_score.call_args_list] == [3, 3]

    @pytest.mark.asyncio
    async def test_judge_only_cell_still_reports_its_sample_count(self):
        judge_row = {"strategy": "bm25", "query_type": "factual", "judge_avg": 0.7, "n": 7}
        pool = _make_pool(fetch_side_effect=[[], [judge_row], []])
        updater = FeedbackUpdater(db_pool=pool, redis_client=_make_redis(), registry=_make_registry())
        scores = await updater._aggregate_scores()
        assert scores[("bm25", "factual")].sample_count == 7


# ---------------------------------------------------------------------------
# _event_to_row helper
# ---------------------------------------------------------------------------

class TestEventToRow:
    def test_returns_8_tuple(self):
        e = json.loads(_json_event())
        row = _event_to_row(e)
        assert len(row) == 8

    def test_signal_value_is_float(self):
        e = json.loads(_json_event(signal_value=0.8))
        row = _event_to_row(e)
        assert isinstance(row[4], float)

    def test_was_exploration_is_bool(self):
        e = json.loads(_json_event())
        row = _event_to_row(e)
        assert isinstance(row[5], bool)

    def test_invalid_created_at_falls_back_to_now(self):
        e = json.loads(_json_event())
        e["created_at"] = "not-a-date"
        row = _event_to_row(e)
        assert isinstance(row[7], datetime)

    def test_created_at_is_naive_utc_for_timestamp_columns(self):
        row = _event_to_row(json.loads(_json_event()))  # ISO string carries +00:00
        assert row[7].tzinfo is None

    def test_fallback_created_at_is_naive_utc(self):
        e = json.loads(_json_event())
        e["created_at"] = "not-a-date"
        assert _event_to_row(e)[7].tzinfo is None
