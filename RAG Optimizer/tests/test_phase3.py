"""Phase 3 unit tests — classifier, registry, bandit, router.

All external dependencies (Anthropic, Postgres, Redis) are mocked.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from retrieval.base import QueryType, RetrievedDoc, RetrievalResult
from router.bandit import EpsilonGreedyBandit, Selection
from router.classifier import QueryClassifier
from router.registry import COLD_START_DEFAULTS, StrategyRegistry, _ALL_STRATEGIES
from router.router import AdaptiveRouter, RouterResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_result(strategy: str = "dense") -> RetrievalResult:
    return RetrievalResult(
        query="q",
        docs=[RetrievedDoc(doc_id="d1", score=0.9, text="t", rank=1)],
        strategy=strategy,
        query_type=QueryType.SEMANTIC,
        latency_ms=10.0,
    )


def _mock_retriever(strategy: str = "dense") -> AsyncMock:
    r = AsyncMock()
    r.retrieve = AsyncMock(return_value=_make_result(strategy))
    r.warm_up = AsyncMock()
    return r


def _mock_anthropic_client(label: str) -> AsyncMock:
    client = AsyncMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=label)]
    client.messages.create = AsyncMock(return_value=msg)
    return client


def _mock_redis(cached_value: str | None = None) -> AsyncMock:
    redis = AsyncMock()
    redis.get = AsyncMock(
        return_value=cached_value.encode() if cached_value else None
    )
    redis.setex = AsyncMock()
    redis.delete = AsyncMock()
    return redis


# ---------------------------------------------------------------------------
# QueryClassifier
# ---------------------------------------------------------------------------

class TestQueryClassifier:
    @pytest.mark.asyncio
    async def test_factual_query_classification(self):
        clf = QueryClassifier(anthropic_client=_mock_anthropic_client("FACTUAL"))
        result = await clf.classify("When was Python first released?")
        assert result == QueryType.FACTUAL

    @pytest.mark.asyncio
    async def test_semantic_query_classification(self):
        clf = QueryClassifier(anthropic_client=_mock_anthropic_client("SEMANTIC"))
        result = await clf.classify("Why do people find functional programming elegant?")
        assert result == QueryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_entity_query_classification(self):
        clf = QueryClassifier(anthropic_client=_mock_anthropic_client("ENTITY"))
        result = await clf.classify("What is Anthropic's mission?")
        assert result == QueryType.ENTITY

    @pytest.mark.asyncio
    async def test_unknown_on_api_failure(self):
        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=RuntimeError("rate limit"))
        clf = QueryClassifier(anthropic_client=client)
        result = await clf.classify("test query")
        assert result == QueryType.UNKNOWN

    @pytest.mark.asyncio
    async def test_unrecognized_label_defaults_to_semantic(self):
        clf = QueryClassifier(anthropic_client=_mock_anthropic_client("GARBAGE_LABEL"))
        result = await clf.classify("some query")
        assert result == QueryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_cache_hit_skips_claude(self):
        redis = _mock_redis(cached_value="factual")
        client = AsyncMock()
        client.messages.create = AsyncMock()
        clf = QueryClassifier(anthropic_client=client, redis_client=redis)
        result = await clf.classify("cached query")
        assert result == QueryType.FACTUAL
        client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_stores_result_in_redis(self):
        redis = _mock_redis(cached_value=None)
        clf = QueryClassifier(
            anthropic_client=_mock_anthropic_client("SEMANTIC"), redis_client=redis
        )
        await clf.classify("new query")
        redis.setex.assert_called_once()
        # Verify the stored value is the QueryType's string value
        stored_value = redis.setex.call_args[0][2]
        assert stored_value == QueryType.SEMANTIC.value

    @pytest.mark.asyncio
    async def test_corrupt_cache_falls_through_to_api(self):
        redis = _mock_redis(cached_value="NOT_A_VALID_QUERY_TYPE")
        client = _mock_anthropic_client("FACTUAL")
        clf = QueryClassifier(anthropic_client=client, redis_client=redis)
        result = await clf.classify("q")
        assert result == QueryType.FACTUAL
        client.messages.create.assert_called_once()


# ---------------------------------------------------------------------------
# StrategyRegistry
# ---------------------------------------------------------------------------

class TestStrategyRegistry:
    def _make_registry(self, rows=None, redis_client=None) -> StrategyRegistry:
        pool = AsyncMock()
        pool.fetch = AsyncMock(return_value=rows or [])
        pool.execute = AsyncMock()
        return StrategyRegistry(db_pool=pool, redis_client=redis_client)

    @pytest.mark.asyncio
    async def test_cold_start_no_db_returns_all_strategies(self):
        reg = StrategyRegistry(db_pool=None)
        scores = await reg.get_scores(QueryType.SEMANTIC)
        assert set(scores.keys()) == set(_ALL_STRATEGIES)

    @pytest.mark.asyncio
    async def test_cold_start_scores_respect_ordering(self):
        reg = StrategyRegistry(db_pool=None)
        scores = await reg.get_scores(QueryType.SEMANTIC)
        # "dense" is first for SEMANTIC — should have the highest pseudo-score
        assert scores["dense"] > scores["bm25"]

    @pytest.mark.asyncio
    async def test_cold_start_sample_counts_are_zero(self):
        reg = StrategyRegistry(db_pool=None)
        _, counts = await reg.get_weights(QueryType.FACTUAL)
        assert all(c == 0 for c in counts.values())

    @pytest.mark.asyncio
    async def test_db_scores_used_after_min_samples(self):
        rows = [
            {"strategy": "dense", "avg_score": 0.75, "sample_count": 15},
            {"strategy": "bm25",  "avg_score": 0.60, "sample_count": 12},
        ]
        reg = self._make_registry(rows=rows)
        scores = await reg.get_scores(QueryType.SEMANTIC)
        assert scores["dense"] == pytest.approx(0.75)
        assert scores["bm25"] == pytest.approx(0.60)

    @pytest.mark.asyncio
    async def test_set_score_overwrites_the_window_aggregate(self):
        reg = self._make_registry()
        await reg.set_score("dense", QueryType.SEMANTIC, 0.8, 42)
        reg._pool.execute.assert_called_once()
        sql, *params = reg._pool.execute.call_args[0]
        assert params == ["dense", "semantic", 0.8, 42]
        assert "EXCLUDED.sample_count" in sql  # overwrite, never increment

    @pytest.mark.asyncio
    async def test_set_score_invalidates_redis_cache(self):
        redis = _mock_redis()
        reg = self._make_registry(redis_client=redis)
        await reg.set_score("dense", QueryType.SEMANTIC, 0.8, 42)
        redis.delete.assert_called_once_with("registry:scores:semantic")

    @pytest.mark.asyncio
    async def test_redis_cache_hit_skips_db(self):
        payload = json.dumps({
            "scores": {"dense": 0.8},
            "counts": {"dense": 20},
        })
        redis = _mock_redis(cached_value=payload)
        pool = AsyncMock()
        pool.fetch = AsyncMock()
        reg = StrategyRegistry(db_pool=pool, redis_client=redis)
        scores = await reg.get_scores(QueryType.SEMANTIC)
        pool.fetch.assert_not_called()
        assert scores["dense"] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_set_score_no_pool_logs_warning(self, caplog):
        import logging
        reg = StrategyRegistry(db_pool=None)
        with caplog.at_level(logging.WARNING, logger="router.registry"):
            await reg.set_score("dense", QueryType.SEMANTIC, 0.5, 3)
        assert "skipped" in caplog.text


# ---------------------------------------------------------------------------
# EpsilonGreedyBandit
# ---------------------------------------------------------------------------

class TestEpsilonGreedyBandit:
    def _scores(self) -> dict[str, float]:
        return {"dense": 0.9, "bm25": 0.7, "hybrid": 0.8}

    def _counts(self, value: int = 20) -> dict[str, int]:
        return {"dense": value, "bm25": value, "hybrid": value}

    def test_epsilon_zero_always_exploits_best(self):
        bandit = EpsilonGreedyBandit(epsilon=0.0, min_samples=0)
        for _ in range(20):
            sel = bandit.select(self._scores(), self._counts())
            assert sel.strategy == "dense"
            assert sel.was_exploration is False

    def test_epsilon_one_always_explores(self):
        bandit = EpsilonGreedyBandit(epsilon=1.0, min_samples=0)
        strategies = {bandit.select(self._scores(), self._counts()).strategy for _ in range(50)}
        assert len(strategies) > 1  # must explore more than just the best

    def test_exploration_flag_true_when_epsilon_fires(self):
        bandit = EpsilonGreedyBandit(epsilon=1.0, min_samples=0)
        sel = bandit.select(self._scores(), self._counts())
        assert sel.was_exploration is True

    def test_cold_strategy_forces_exploration(self):
        counts = {"dense": 20, "bm25": 5, "hybrid": 20}  # bm25 is cold
        bandit = EpsilonGreedyBandit(epsilon=0.0, min_samples=10)
        for _ in range(20):
            sel = bandit.select(self._scores(), counts)
            assert sel.strategy == "bm25"
            assert sel.was_exploration is True

    def test_all_cold_picks_from_cold_set(self):
        bandit = EpsilonGreedyBandit(epsilon=0.0, min_samples=100)
        selections = {bandit.select(self._scores(), self._counts(0)).strategy for _ in range(50)}
        assert selections <= set(self._scores().keys())

    def test_empty_scores_raises_value_error(self):
        bandit = EpsilonGreedyBandit()
        with pytest.raises(ValueError, match="empty"):
            bandit.select({}, {})

    def test_exploit_does_not_pick_second_best(self):
        scores = {"good": 1.0, "bad": 0.0}
        counts = {"good": 50, "bad": 50}
        bandit = EpsilonGreedyBandit(epsilon=0.0, min_samples=10)
        assert bandit.select(scores, counts).strategy == "good"


# ---------------------------------------------------------------------------
# AdaptiveRouter
# ---------------------------------------------------------------------------

class TestAdaptiveRouter:
    def _make_router(
        self,
        query_type: QueryType = QueryType.SEMANTIC,
        selected_strategy: str = "dense",
        was_exploration: bool = False,
        reranker=None,
    ) -> AdaptiveRouter:
        clf = AsyncMock()
        clf.classify = AsyncMock(return_value=query_type)

        registry = AsyncMock()
        registry.get_weights = AsyncMock(
            return_value=({"dense": 0.9, "bm25": 0.7}, {"dense": 20, "bm25": 20})
        )

        bandit = MagicMock()
        bandit.select = MagicMock(
            return_value=Selection(strategy=selected_strategy, was_exploration=was_exploration)
        )

        strategies = {
            "dense": _mock_retriever("dense"),
            "bm25": _mock_retriever("bm25"),
        }

        return AdaptiveRouter(
            strategies=strategies,
            reranker=reranker,
            classifier=clf,
            registry=registry,
            bandit=bandit,
        )

    @pytest.mark.asyncio
    async def test_force_strategy_bypasses_classifier(self):
        router = self._make_router()
        result = await router.route("q", force_strategy="bm25")
        assert result.selected_strategy == "bm25"
        router._classifier.classify.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_strategy_bypasses_bandit(self):
        router = self._make_router()
        await router.route("q", force_strategy="dense")
        router._bandit.select.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_strategy_sets_unknown_query_type(self):
        router = self._make_router()
        result = await router.route("q", force_strategy="dense")
        assert result.query_type == QueryType.UNKNOWN

    @pytest.mark.asyncio
    async def test_normal_route_uses_classifier(self):
        router = self._make_router(query_type=QueryType.FACTUAL)
        result = await router.route("What year was Python released?")
        assert result.query_type == QueryType.FACTUAL

    @pytest.mark.asyncio
    async def test_was_exploration_propagated_from_bandit(self):
        router = self._make_router(was_exploration=True)
        result = await router.route("q")
        assert result.was_exploration is True

    @pytest.mark.asyncio
    async def test_was_exploration_false_for_exploit(self):
        router = self._make_router(was_exploration=False)
        result = await router.route("q")
        assert result.was_exploration is False

    @pytest.mark.asyncio
    async def test_reranker_called_when_provided(self):
        reranker = AsyncMock()
        reranker.rerank = AsyncMock(
            return_value=[RetrievedDoc(doc_id="d1", score=0.95, text="t", rank=1)]
        )
        router = self._make_router(reranker=reranker)
        result = await router.route("q")
        assert result.reranked is True
        reranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_reranker_skipped_when_none(self):
        router = self._make_router(reranker=None)
        result = await router.route("q")
        assert result.reranked is False

    @pytest.mark.asyncio
    async def test_rerank_false_skips_reranker_and_overfetch(self):
        reranker = AsyncMock()
        reranker.rerank = AsyncMock()
        router = self._make_router(reranker=reranker)
        result = await router.route("q", top_k=4, force_strategy="dense", rerank=False)
        reranker.rerank.assert_not_called()
        assert result.reranked is False
        router._strategies["dense"].retrieve.assert_called_once_with("q", top_k=4)

    @pytest.mark.asyncio
    async def test_warm_up_calls_all_strategies(self):
        router = self._make_router()
        await router.warm_up()
        for retriever in router._strategies.values():
            retriever.warm_up.assert_called_once()

    @pytest.mark.asyncio
    async def test_router_result_has_positive_latencies(self):
        router = self._make_router()
        result = await router.route("q")
        assert result.classifier_latency_ms >= 0
        assert result.router_latency_ms > 0

    @pytest.mark.asyncio
    async def test_retrieval_result_is_returned(self):
        router = self._make_router(selected_strategy="dense")
        result = await router.route("q")
        assert isinstance(result.retrieval_result, RetrievalResult)
        assert result.retrieval_result.strategy == "dense"

    @pytest.mark.asyncio
    async def test_force_strategy_unknown_raises_value_error(self):
        router = self._make_router()
        with pytest.raises(ValueError, match="Unknown strategy"):
            await router.route("q", force_strategy="nonexistent")

    @pytest.mark.asyncio
    async def test_warm_up_removes_failed_strategy(self):
        """A strategy that fails warm_up is removed from the active set, not propagated."""
        good = _mock_retriever("dense")
        bad = AsyncMock()
        bad.warm_up = AsyncMock(side_effect=ConnectionError("Qdrant unreachable"))
        router = AdaptiveRouter(strategies={"dense": good, "failing": bad})
        await router.warm_up()
        assert "failing" not in router._strategies
        assert "dense" in router._strategies

    @pytest.mark.asyncio
    async def test_warm_up_all_fail_raises_runtime_error(self):
        bad1 = AsyncMock()
        bad1.warm_up = AsyncMock(side_effect=ConnectionError("down"))
        bad2 = AsyncMock()
        bad2.warm_up = AsyncMock(side_effect=ConnectionError("down"))
        router = AdaptiveRouter(strategies={"a": bad1, "b": bad2})
        with pytest.raises(RuntimeError, match="All retrieval strategies"):
            await router.warm_up()
