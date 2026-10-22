"""Phase 4 unit tests — compressor, generator, RAGAS scorer, LLM judge, CI gate, BEIR eval.

All external API calls are mocked. No live Qdrant/ES/Anthropic required.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmarks.beir_eval import BEIREvaluator, MetricsResult, check_regression
from benchmarks.llm_judge import JudgeScores, LLMJudge, _composite, _clamp
from benchmarks.ragas_scorer import RAGASScores, RAGASScorer
from retrieval.base import QueryType, RetrievedDoc, RetrievalResult
from retrieval.compressor.contextual_compressor import (
    CompressedDoc,
    ContextualCompressor,
    _approx_tokens,
    _first_two_sentences,
)
from router.router import AdaptiveRouter, RouterResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(doc_id: str = "d1", text: str = "Some text here.", score: float = 0.9) -> RetrievedDoc:
    return RetrievedDoc(doc_id=doc_id, score=score, text=text, rank=1)


def _make_retrieval_result(docs=None, strategy="dense") -> RetrievalResult:
    return RetrievalResult(
        query="q",
        docs=docs or [_make_doc()],
        strategy=strategy,
        query_type=QueryType.SEMANTIC,
        latency_ms=10.0,
    )


def _make_router_result(strategy="dense") -> RouterResult:
    return RouterResult(
        retrieval_result=_make_retrieval_result(strategy=strategy),
        selected_strategy=strategy,
        query_type=QueryType.SEMANTIC,
        was_exploration=False,
        classifier_latency_ms=0.0,
        router_latency_ms=10.0,
        reranked=False,
    )


def _mock_anthropic(response_text: str) -> AsyncMock:
    client = AsyncMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=response_text)]
    client.messages.create = AsyncMock(return_value=msg)
    return client


# ---------------------------------------------------------------------------
# ContextualCompressor helpers
# ---------------------------------------------------------------------------

class TestApproxTokens:
    def test_empty_string(self):
        assert _approx_tokens("") == 1  # max(1, ...)

    def test_ten_words(self):
        assert _approx_tokens("a " * 10) == pytest.approx(13, abs=1)


class TestFirstTwoSentences:
    def test_returns_two_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _first_two_sentences(text)
        assert "First sentence" in result
        assert "Second sentence" in result
        assert "Third" not in result

    def test_no_periods_returns_truncated(self):
        text = "word " * 200
        result = _first_two_sentences(text)
        assert len(result) <= 410


# ---------------------------------------------------------------------------
# ContextualCompressor
# ---------------------------------------------------------------------------

class TestContextualCompressor:
    @pytest.mark.asyncio
    async def test_short_doc_skips_compression(self):
        doc = _make_doc(text="Short text.")  # under any reasonable max_tokens
        client = AsyncMock()
        client.messages.create = AsyncMock()
        compressor = ContextualCompressor(anthropic_client=client)
        results = await compressor.compress("q", [doc], max_tokens_per_doc=500)
        client.messages.create.assert_not_called()
        assert results[0].compressed_text == "Short text."
        assert results[0].compression_ratio == 1.0

    @pytest.mark.asyncio
    async def test_empty_claude_response_uses_fallback(self):
        long_text = "word " * 300
        doc = _make_doc(text=long_text)
        compressor = ContextualCompressor(anthropic_client=_mock_anthropic(""))
        results = await compressor.compress("q", [doc], max_tokens_per_doc=50)
        assert results[0].compressed_text  # not empty
        assert results[0].compressed_text != long_text

    @pytest.mark.asyncio
    async def test_api_error_uses_fallback(self):
        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
        doc = _make_doc(text="word " * 300)
        compressor = ContextualCompressor(anthropic_client=client)
        results = await compressor.compress("q", [doc], max_tokens_per_doc=50)
        assert results[0].compressed_text  # fallback is non-empty

    @pytest.mark.asyncio
    async def test_compression_ratio_in_valid_range(self):
        doc = _make_doc(text="word " * 300)
        compressor = ContextualCompressor(anthropic_client=_mock_anthropic("compressed text here"))
        results = await compressor.compress("q", [doc], max_tokens_per_doc=50)
        assert 0.0 < results[0].compression_ratio <= 1.0

    @pytest.mark.asyncio
    async def test_all_docs_processed_concurrently(self):
        docs = [_make_doc(f"d{i}", "Short.") for i in range(5)]
        client = AsyncMock()
        client.messages.create = AsyncMock()
        compressor = ContextualCompressor(anthropic_client=client)
        results = await compressor.compress("q", docs, max_tokens_per_doc=500)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# RAGASScorer
# ---------------------------------------------------------------------------

class TestRAGASScorer:
    @pytest.mark.asyncio
    async def test_faithfulness_all_claims_supported(self):
        scorer = RAGASScorer()
        scorer._extract_claims = AsyncMock(return_value=["claim 1", "claim 2"])
        scorer._verify_claim = AsyncMock(return_value=1.0)
        scorer._answer_relevancy = AsyncMock(return_value=0.9)
        scorer._context_precision = AsyncMock(return_value=0.8)
        result = await scorer.score("q", "answer", ["ctx"])
        assert result.faithfulness == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_faithfulness_no_claims_supported(self):
        scorer = RAGASScorer()
        scorer._extract_claims = AsyncMock(return_value=["claim 1", "claim 2"])
        scorer._verify_claim = AsyncMock(return_value=0.0)
        scorer._answer_relevancy = AsyncMock(return_value=0.5)
        scorer._context_precision = AsyncMock(return_value=0.5)
        result = await scorer.score("q", "answer", ["ctx"])
        assert result.faithfulness == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_empty_claims_gives_vacuous_truth(self):
        scorer = RAGASScorer()
        scorer._extract_claims = AsyncMock(return_value=[])
        scorer._answer_relevancy = AsyncMock(return_value=0.7)
        scorer._context_precision = AsyncMock(return_value=0.6)
        result = await scorer.score("q", "answer", ["ctx"])
        assert result.faithfulness == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_returns_valid_scores_on_api_error(self):
        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=RuntimeError("down"))
        scorer = RAGASScorer(anthropic_client=client)
        result = await scorer.score("q", "answer", ["ctx"])
        assert isinstance(result, RAGASScores)
        assert 0.0 <= result.faithfulness <= 1.0

    @pytest.mark.asyncio
    async def test_composite_is_mean_of_three_metrics(self):
        scores = RAGASScores(
            faithfulness=0.8,
            answer_relevancy=0.6,
            context_precision=1.0,
        )
        assert scores.composite == pytest.approx((0.8 + 0.6 + 1.0) / 3)

    @pytest.mark.asyncio
    async def test_context_precision_all_useful(self):
        scorer = RAGASScorer()
        scorer._is_chunk_useful = AsyncMock(return_value=1.0)
        scorer._extract_claims = AsyncMock(return_value=[])
        scorer._answer_relevancy = AsyncMock(return_value=0.5)
        result = await scorer.score("q", "a", ["chunk1", "chunk2"])
        assert result.context_precision == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------

class TestLLMJudge:
    def _judge_response(self, r=4, co=3, g=4, cl=3, reasoning="ok") -> str:
        return json.dumps({
            "relevance": r, "completeness": co,
            "groundedness": g, "clarity": cl, "reasoning": reasoning,
        })

    @pytest.mark.asyncio
    async def test_valid_response_parsed_correctly(self):
        judge = LLMJudge(anthropic_client=_mock_anthropic(self._judge_response(4, 3, 4, 3)))
        scores = await judge.judge("q", "answer", ["ctx"])
        assert scores.relevance == 4
        assert scores.completeness == 3

    @pytest.mark.asyncio
    async def test_parse_failure_returns_zero_composite(self):
        judge = LLMJudge(anthropic_client=_mock_anthropic("not json at all"))
        scores = await judge.judge("q", "answer", ["ctx"])
        assert scores.composite == 0.0
        assert scores.reasoning == "parse_error"

    @pytest.mark.asyncio
    async def test_json_fence_stripped_before_parse(self):
        fenced = "```json\n" + self._judge_response() + "\n```"
        judge = LLMJudge(anthropic_client=_mock_anthropic(fenced))
        scores = await judge.judge("q", "answer", ["ctx"])
        assert scores.relevance > 0

    def test_composite_formula_matches_weights(self):
        composite = _composite(5, 5, 5, 5)
        assert composite == pytest.approx(1.0)
        composite_low = _composite(1, 1, 1, 1)
        assert composite_low == pytest.approx(0.2)

    def test_clamp_bounds(self):
        assert _clamp(0) == 1
        assert _clamp(6) == 5
        assert _clamp(3) == 3

    def test_out_of_range_scores_clamped(self):
        judge = LLMJudge.__new__(LLMJudge)
        scores = judge._parse(
            json.dumps({"relevance": 10, "completeness": -1, "groundedness": 3, "clarity": 3, "reasoning": "x"}),
            "q",
        )
        assert scores.relevance == 5
        assert scores.completeness == 1

    @pytest.mark.asyncio
    async def test_api_error_returns_zero(self):
        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=RuntimeError("down"))
        judge = LLMJudge(anthropic_client=client)
        scores = await judge.judge("q", "a", ["ctx"])
        assert scores.composite == 0.0


# ---------------------------------------------------------------------------
# CI regression gate
# ---------------------------------------------------------------------------

class TestCheckRegression:
    def _results(
        self,
        ndcg: dict[str, float],
        latency_ms: dict[str, float] | None = None,
    ) -> dict[str, MetricsResult]:
        latency_ms = latency_ms or {}
        return {
            s: MetricsResult(metrics={"ndcg@10": v}, avg_latency_ms=latency_ms.get(s, 0.0))
            for s, v in ndcg.items()
        }

    def test_regression_detected_on_large_drop(self):
        current  = self._results({"dense": 0.250, "bm25": 0.310})
        baseline = self._results({"dense": 0.320, "bm25": 0.310})  # dense dropped 0.07
        passed, msg = check_regression(current, baseline)
        assert not passed
        assert "dense" in msg

    def test_no_regression_when_within_threshold(self):
        current  = self._results({"dense": 0.299, "bm25": 0.310})
        baseline = self._results({"dense": 0.300, "bm25": 0.310})  # only 0.001 drop
        passed, _ = check_regression(current, baseline)
        assert passed

    def test_improvement_is_not_regression(self):
        current  = self._results({"hybrid": 0.400})
        baseline = self._results({"hybrid": 0.350})
        passed, _ = check_regression(current, baseline)
        assert passed

    def test_strategy_not_in_baseline_is_skipped(self):
        current  = self._results({"new_strategy": 0.10})
        baseline = self._results({"bm25": 0.30})
        passed, _ = check_regression(current, baseline)
        assert passed

    def test_regression_message_contains_strategy_and_delta(self):
        current  = self._results({"bm25": 0.250})
        baseline = self._results({"bm25": 0.320})
        _, msg = check_regression(current, baseline)
        assert "bm25" in msg
        assert "0.070" in msg

    def test_latency_regression_detected_when_over_threshold(self):
        current = self._results({"dense": 0.320}, latency_ms={"dense": 130.0})
        baseline = self._results({"dense": 0.320}, latency_ms={"dense": 100.0})
        passed, msg = check_regression(
            current,
            baseline,
            threshold=0.02,
            latency_regression_pct=20.0,
        )
        assert not passed
        assert "latency" in msg.lower()
        assert "30.0%" in msg

    def test_latency_regression_ignored_when_disabled(self):
        current = self._results({"dense": 0.320}, latency_ms={"dense": 150.0})
        baseline = self._results({"dense": 0.320}, latency_ms={"dense": 100.0})
        passed, _ = check_regression(
            current,
            baseline,
            threshold=0.02,
            latency_regression_pct=None,
        )
        assert passed


# ---------------------------------------------------------------------------
# BEIREvaluator
# ---------------------------------------------------------------------------

class TestBEIREvaluator:
    def _make_mock_router(self, doc_id: str = "MED-10") -> MagicMock:
        router = MagicMock()
        router._strategies = {"dense": MagicMock(), "bm25": MagicMock()}
        router.route = AsyncMock(
            return_value=RouterResult(
                retrieval_result=_make_retrieval_result(
                    docs=[RetrievedDoc(doc_id=doc_id, score=0.9, text="t", rank=1)],
                ),
                selected_strategy="dense",
                query_type=QueryType.SEMANTIC,
                was_exploration=False,
                classifier_latency_ms=0.0,
                router_latency_ms=5.0,
                reranked=False,
            )
        )
        return router

    @pytest.mark.asyncio
    async def test_evaluate_from_data_returns_all_strategies(self):
        router = self._make_mock_router()
        evaluator = BEIREvaluator(router=router)
        qrels   = {"q1": {"MED-10": 2}}
        queries = {"q1": "what causes inflammation?"}
        corpus  = {"MED-10": {"text": "Inflammation text"}}
        results = await evaluator.evaluate_from_data(
            corpus=corpus, queries=queries, qrels=qrels,
            strategies=["dense", "bm25"], k_values=[10],
        )
        assert "dense" in results
        assert "bm25" in results

    @pytest.mark.asyncio
    async def test_evaluate_uses_force_strategy(self):
        router = self._make_mock_router(doc_id="MED-10")
        evaluator = BEIREvaluator(router=router)
        qrels   = {"q1": {"MED-10": 1}}
        queries = {"q1": "test query"}
        corpus  = {"MED-10": {"text": "t"}}
        await evaluator.evaluate_from_data(
            corpus=corpus, queries=queries, qrels=qrels,
            strategies=["dense"], k_values=[10],
        )
        # route must be called with force_strategy="dense"
        call_kwargs = router.route.call_args[1]
        assert call_kwargs.get("force_strategy") == "dense"

    @pytest.mark.asyncio
    async def test_perfect_retrieval_gives_ndcg_one(self):
        router = self._make_mock_router(doc_id="MED-10")
        evaluator = BEIREvaluator(router=router)
        qrels   = {"q1": {"MED-10": 1}}
        queries = {"q1": "test query"}
        corpus  = {"MED-10": {"text": "t"}}
        results = await evaluator.evaluate_from_data(
            corpus=corpus, queries=queries, qrels=qrels,
            strategies=["dense"], k_values=[1],
        )
        assert results["dense"].metrics["ndcg@1"] == pytest.approx(1.0)

    def test_metrics_result_primary(self):
        m = MetricsResult(metrics={"ndcg@10": 0.35, "mrr": 0.45})
        assert m.primary() == pytest.approx(0.35)

    def test_metrics_result_round_trips(self):
        m = MetricsResult(metrics={"ndcg@10": 0.35}, avg_latency_ms=123.0)
        assert MetricsResult.from_dict(m.to_dict()).primary() == pytest.approx(0.35)

    @pytest.mark.asyncio
    async def test_retrieves_to_the_deepest_metric_cutoff(self):
        # Regression: route() defaults to 5 docs, which silently truncated nDCG@10 and Recall@100.
        router = self._make_mock_router()
        evaluator = BEIREvaluator(router=router)
        data = dict(
            corpus={"MED-10": {"text": "t"}}, queries={"q1": "test"}, qrels={"q1": {"MED-10": 1}}
        )
        await evaluator.evaluate_from_data(**data, strategies=["dense"])
        assert router.route.call_args[1]["top_k"] == 100
        await evaluator.evaluate_from_data(**data, strategies=["dense"], k_values=[10])
        assert router.route.call_args[1]["top_k"] == 10

    @pytest.mark.asyncio
    async def test_benchmark_measures_first_stage_retrieval_only(self):
        router = self._make_mock_router()
        await BEIREvaluator(router=router).evaluate_from_data(
            corpus={}, queries={"q1": "test"}, qrels={"q1": {"MED-10": 1}}, strategies=["dense"]
        )
        assert router.route.call_args[1]["rerank"] is False

    @pytest.mark.asyncio
    async def test_collects_per_query_ndcg_and_latency_percentiles(self):
        router = self._make_mock_router(doc_id="MED-10")
        results = await BEIREvaluator(router=router).evaluate_from_data(
            corpus={}, queries={"q1": "test"}, qrels={"q1": {"MED-10": 1}},
            strategies=["dense"], k_values=[10],
        )
        dense = results["dense"]
        assert dense.per_query_ndcg10 == {"q1": pytest.approx(1.0)}
        assert dense.p50_latency_ms == pytest.approx(10.0)
        assert dense.p95_latency_ms == pytest.approx(10.0)

    def test_baseline_json_keeps_percentiles_but_not_per_query_scores(self):
        m = MetricsResult(
            metrics={"ndcg@10": 0.3}, p50_latency_ms=12.0, p95_latency_ms=40.0,
            per_query_ndcg10={"q1": 0.3},
        )
        assert "per_query_ndcg10" not in m.to_dict()
        assert MetricsResult.from_dict(m.to_dict()).p95_latency_ms == pytest.approx(40.0)
