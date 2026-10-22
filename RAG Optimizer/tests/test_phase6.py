"""Phase 6 unit tests — schemas, routes, tracer, metrics.

Routes are tested with a minimal FastAPI app (no lifespan) where all
dependencies are overridden. No live infra required.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from api import routes
from api.dependencies import (
    get_collector,
    get_compressor,
    get_db_pool,
    get_generator,
    get_judge,
    get_metrics,
    get_redis_client,
    get_registry,
    get_router,
    get_scorer,
    get_tracer,
    verify_api_key,
)
from api.schemas import (
    BenchmarkRequest,
    DocumentResponse,
    FeedbackRequest,
    QueryRequest,
    QueryResponse,
    StrategyScoresResponse,
)
from benchmarks.ragas_scorer import RAGASScores
from observability.metrics_collector import PrometheusMetrics
from observability.tracer import LangfuseTracer, TraceHandle, _NoOpTraceHandle
from retrieval.base import QueryType, RetrievedDoc, RetrievalResult
from retrieval.compressor.contextual_compressor import CompressedDoc
from router.router import RouterResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _router_result(strategy: str = "hybrid") -> RouterResult:
    return RouterResult(
        retrieval_result=RetrievalResult(
            query="test query",
            docs=[RetrievedDoc(doc_id="d1", score=0.9, text="relevant", rank=1)],
            strategy=strategy,
            query_type=QueryType.SEMANTIC,
            latency_ms=100.0,
        ),
        selected_strategy=strategy,
        query_type=QueryType.SEMANTIC,
        was_exploration=False,
        classifier_latency_ms=50.0,
        router_latency_ms=150.0,
        reranked=False,
    )


def _compressed_doc(text: str = "compressed") -> CompressedDoc:
    return CompressedDoc(
        original=RetrievedDoc(doc_id="d1", score=0.9, text="original", rank=1),
        compressed_text=text,
        original_tokens=100,
        compressed_tokens=50,
        compression_ratio=0.5,
    )


@pytest.fixture
def api_client() -> TestClient:
    """Minimal FastAPI app with all dependencies mocked — no lifespan runs."""
    mock_router = AsyncMock()
    mock_router.route = AsyncMock(return_value=_router_result())

    mock_compressor = AsyncMock()
    mock_compressor.compress = AsyncMock(return_value=[_compressed_doc()])

    mock_generator = AsyncMock()
    mock_generator.generate = AsyncMock(return_value="The answer is X.")

    mock_scorer = AsyncMock()
    mock_scorer.score = AsyncMock(
        return_value=RAGASScores(faithfulness=0.9, answer_relevancy=0.8, context_precision=0.7)
    )

    mock_judge = AsyncMock()
    mock_judge.judge = AsyncMock(return_value=MagicMock(
        relevance=4,
        completeness=4,
        groundedness=4,
        clarity=4,
        composite=0.8,
        reasoning="ok",
    ))

    mock_collector = AsyncMock()
    mock_collector.record_explicit = AsyncMock()

    mock_registry = AsyncMock()
    mock_registry.get_weights = AsyncMock(
        return_value=({"dense": 0.8, "bm25": 0.7}, {"dense": 15, "bm25": 12})
    )

    mock_trace = MagicMock()
    mock_trace.trace_id = "trace-abc"
    mock_tracer = MagicMock()
    mock_tracer.create_trace = MagicMock(return_value=mock_trace)

    mock_metrics = MagicMock()
    mock_metrics.record_query = MagicMock()

    app = FastAPI()
    app.include_router(routes.router, prefix="/api/v1")
    app.dependency_overrides[get_router]       = lambda: mock_router
    app.dependency_overrides[get_compressor]   = lambda: mock_compressor
    app.dependency_overrides[get_generator]    = lambda: mock_generator
    app.dependency_overrides[get_scorer]       = lambda: mock_scorer
    app.dependency_overrides[get_judge]        = lambda: mock_judge
    app.dependency_overrides[get_collector]    = lambda: mock_collector
    app.dependency_overrides[get_registry]     = lambda: mock_registry
    app.dependency_overrides[get_tracer]       = lambda: mock_tracer
    app.dependency_overrides[get_metrics]      = lambda: mock_metrics
    app.dependency_overrides[get_db_pool]      = lambda: None
    app.dependency_overrides[get_redis_client] = lambda: None
    app.dependency_overrides[verify_api_key]   = lambda: None  # bypass auth in tests

    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_query_request_defaults(self):
        req = QueryRequest(query="test")
        assert req.top_k == 5
        assert req.force_strategy is None
        assert req.session_id is None

    def test_empty_query_rejected(self):
        import pytest
        with pytest.raises(Exception):  # pydantic ValidationError
            QueryRequest(query="")

    def test_query_too_long_rejected(self):
        import pytest
        with pytest.raises(Exception):
            QueryRequest(query="x" * 2001)

    def test_top_k_bounds(self):
        import pytest
        with pytest.raises(Exception):
            QueryRequest(query="test", top_k=0)
        with pytest.raises(Exception):
            QueryRequest(query="test", top_k=21)

    def test_feedback_request_required_fields(self):
        req = FeedbackRequest(
            query="q", strategy="dense", query_type="semantic", positive=True
        )
        assert req.was_exploration is False

    def test_benchmark_max_queries_capped(self):
        import pytest
        with pytest.raises(Exception):
            BenchmarkRequest(max_queries=501)

    def test_benchmark_request_defaults(self):
        req = BenchmarkRequest()
        assert req.dataset == "nfcorpus"
        assert req.max_queries == 50
        assert 10 in req.k_values

    def test_query_response_optional_fields(self):
        resp = QueryResponse(
            query="q", answer="a", documents=[], strategy_used="dense",
            query_type="semantic", was_exploration=False, latency_ms=100.0,
        )
        assert resp.evaluation_status == "queued"
        assert resp.trace_id is None

    def test_document_response_metadata_defaults_empty(self):
        doc = DocumentResponse(doc_id="d1", text="t", score=0.9, rank=1, strategy="dense")
        assert doc.metadata == {}


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_returns_200_with_answer(self, api_client):
        with patch("api.routes.asyncio.create_task", side_effect=lambda c: c.close()):
            resp = api_client.post("/api/v1/query", json={"query": "What is AI?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "The answer is X."
        assert body["strategy_used"] == "hybrid"
        assert body["query_type"] == "semantic"

    def test_response_includes_documents(self, api_client):
        with patch("api.routes.asyncio.create_task", side_effect=lambda c: c.close()):
            resp = api_client.post("/api/v1/query", json={"query": "test"})
        docs = resp.json()["documents"]
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "d1"

    def test_background_scoring_task_is_created(self, api_client):
        with patch("api.routes.asyncio.create_task", side_effect=lambda c: c.close()) as mock_task:
            api_client.post("/api/v1/query", json={"query": "test"})
        mock_task.assert_called_once()

    def test_force_strategy_passed_to_router(self, api_client):
        mock_router = api_client.app.dependency_overrides[get_router]()
        with patch("api.routes.asyncio.create_task", side_effect=lambda c: c.close()):
            api_client.post(
                "/api/v1/query",
                json={"query": "test", "force_strategy": "bm25"},
            )
        call_kwargs = mock_router.route.call_args[1]
        assert call_kwargs.get("force_strategy") == "bm25"

    def test_trace_id_omitted_when_include_traces_false(self, api_client):
        with patch("api.routes.asyncio.create_task", side_effect=lambda c: c.close()):
            resp = api_client.post("/api/v1/query", json={"query": "test"})
        assert resp.status_code == 200
        assert resp.json()["trace_id"] is None

    def test_trace_id_present_when_include_traces_true(self, api_client):
        with patch("api.routes.asyncio.create_task", side_effect=lambda c: c.close()):
            resp = api_client.post(
                "/api/v1/query",
                json={"query": "test", "include_traces": True},
            )
        assert resp.status_code == 200
        assert resp.json()["trace_id"] == "trace-abc"

    def test_evaluation_skipped_when_sampled_out(self, api_client):
        sampled_out = MagicMock(
            eval_sample_rate=0.0,
            input_cost_per_1k_tokens_usd=0.003,
            output_cost_per_1k_tokens_usd=0.015,
        )
        with patch("api.routes.get_settings", return_value=sampled_out), \
             patch("api.routes.asyncio.create_task") as mock_task:
            resp = api_client.post("/api/v1/query", json={"query": "test"})
        assert resp.status_code == 200
        assert resp.json()["evaluation_status"] == "skipped"
        mock_task.assert_not_called()


class TestScoreAndStore:
    @pytest.mark.asyncio
    async def test_persists_query_type_with_both_score_rows(self):
        scorer = AsyncMock()
        scorer.score = AsyncMock(
            return_value=RAGASScores(faithfulness=0.9, answer_relevancy=0.8, context_precision=0.7)
        )
        judge = AsyncMock()
        judge.judge = AsyncMock(return_value=MagicMock(
            relevance=4, completeness=4, groundedness=4, clarity=4, composite=0.8, reasoning="ok",
        ))
        pool = AsyncMock()

        await routes._score_and_store("q", "a", ["ctx"], _router_result(), scorer, judge, pool)

        assert pool.execute.await_count == 2
        for call in pool.execute.await_args_list:
            sql, *params = call.args
            assert "query_type" in sql
            assert "semantic" in params


class TestFeedbackEndpoint:
    def test_returns_204(self, api_client):
        resp = api_client.post("/api/v1/feedback", json={
            "query": "q", "strategy": "hybrid", "query_type": "semantic", "positive": True,
        })
        assert resp.status_code == 204

    def test_invalid_query_type_returns_422(self, api_client):
        resp = api_client.post("/api/v1/feedback", json={
            "query": "q", "strategy": "hybrid", "query_type": "INVALID", "positive": True,
        })
        assert resp.status_code == 422


class TestStrategyScoresEndpoint:
    def test_returns_scores_for_query_type(self, api_client):
        resp = api_client.get("/api/v1/strategies/scores?query_type=semantic")
        assert resp.status_code == 200
        body = resp.json()
        assert "scores" in body
        assert "sample_counts" in body
        assert "using_cold_start" in body

    def test_using_cold_start_false_when_counts_sufficient(self, api_client):
        # registry mock returns counts of 15 and 12, both >= min_samples=10
        resp = api_client.get("/api/v1/strategies/scores?query_type=semantic")
        assert resp.json()["using_cold_start"] is False


    def test_invalid_force_strategy_returns_422(self, api_client):
        """Router raises ValueError for unknown strategy — route must return 422."""
        from router.router import AdaptiveRouter
        mock_router = api_client.app.dependency_overrides[get_router]()
        mock_router.route = AsyncMock(side_effect=ValueError("Unknown strategy 'bad'"))
        with patch("api.routes.asyncio.create_task", side_effect=lambda c: c.close()):
            resp = api_client.post(
                "/api/v1/query",
                json={"query": "test", "force_strategy": "bad"},
            )
        assert resp.status_code == 422

    def test_empty_query_rejected_by_schema(self, api_client):
        resp = api_client.post("/api/v1/query", json={"query": ""})
        assert resp.status_code == 422

    def test_generator_failure_returns_503(self, api_client):
        """LLM outage must return 503, not a raw 500 with a traceback."""
        mock_gen = api_client.app.dependency_overrides[get_generator]()
        mock_gen.generate = AsyncMock(side_effect=RuntimeError("Anthropic down"))
        with patch("api.routes.asyncio.create_task", side_effect=lambda c: c.close()):
            resp = api_client.post("/api/v1/query", json={"query": "test"})
        assert resp.status_code == 503


class TestHealthEndpoint:
    def test_health_returns_status_field(self, api_client):
        with patch("api.routes._ping_qdrant", AsyncMock(return_value="unavailable")), \
             patch("api.routes._ping_es",     AsyncMock(return_value="unavailable")):
            resp = api_client.get("/api/v1/health")
        assert resp.status_code == 200
        assert "status" in resp.json()

    def test_degraded_when_service_down(self, api_client):
        with patch("api.routes._ping_qdrant", AsyncMock(return_value="unavailable")), \
             patch("api.routes._ping_es",     AsyncMock(return_value="unavailable")), \
             patch("api.routes._ping_postgres", AsyncMock(return_value="unavailable")), \
             patch("api.routes._ping_redis",    AsyncMock(return_value="unavailable")):
            resp = api_client.get("/api/v1/health")
        assert resp.json()["status"] == "degraded"


# ---------------------------------------------------------------------------
# LangfuseTracer
# ---------------------------------------------------------------------------

class TestLangfuseTracer:
    def test_returns_noop_when_not_configured(self):
        tracer = LangfuseTracer()  # no keys in test env
        handle = tracer.create_trace("test query")
        assert isinstance(handle, _NoOpTraceHandle)

    def test_noop_trace_id_is_none(self):
        handle = _NoOpTraceHandle()
        assert handle.trace_id is None

    def test_noop_methods_do_not_raise(self):
        handle = _NoOpTraceHandle()
        handle.log_retrieval(MagicMock())
        handle.log_generation("q", "a", ["ctx"], "model")
        handle.log_scores(None, None)
        handle.end()

    def test_trace_handle_stores_trace_id(self):
        mock_trace = MagicMock()
        mock_trace.id = "trace-123"
        handle = TraceHandle(mock_trace)
        assert handle.trace_id == "trace-123"

    def test_trace_handle_log_retrieval_calls_span(self):
        mock_trace = MagicMock()
        handle = TraceHandle(mock_trace)
        handle.log_retrieval(_router_result())
        mock_trace.span.assert_called_once()

    def test_trace_handle_log_generation_calls_generation(self):
        mock_trace = MagicMock()
        handle = TraceHandle(mock_trace)
        handle.log_generation("q", "a", ["ctx"], "claude-3-5-sonnet")
        mock_trace.generation.assert_called_once()

    def test_span_failure_does_not_raise(self):
        mock_trace = MagicMock()
        mock_trace.span = MagicMock(side_effect=RuntimeError("Langfuse down"))
        handle = TraceHandle(mock_trace)
        handle.log_retrieval(_router_result())  # must not raise


# ---------------------------------------------------------------------------
# PrometheusMetrics
# ---------------------------------------------------------------------------

class TestPrometheusMetrics:
    def _metrics(self) -> PrometheusMetrics:
        return PrometheusMetrics(registry=CollectorRegistry())

    def test_record_query_increments_counter(self):
        metrics = self._metrics()
        result = _router_result()
        metrics.record_query(result, e2e_ms=200.0)
        val = metrics.queries_total.labels(
            strategy="hybrid", query_type="semantic", was_exploration="False"
        )._value.get()
        assert val == 1.0

    def test_update_strategy_score_sets_gauge(self):
        metrics = self._metrics()
        metrics.update_strategy_score("dense", "semantic", 0.82)
        val = metrics.strategy_score.labels(strategy="dense", query_type="semantic")._value.get()
        assert val == pytest.approx(0.82)

    def test_exploration_counter_incremented_when_exploring(self):
        metrics = self._metrics()
        result = _router_result()
        result = RouterResult(
            retrieval_result=result.retrieval_result,
            selected_strategy="hyde",
            query_type=QueryType.SEMANTIC,
            was_exploration=True,
            classifier_latency_ms=0.0,
            router_latency_ms=0.0,
            reranked=False,
        )
        metrics.record_query(result, e2e_ms=500.0)
        val = metrics.exploration_total.labels(strategy="hyde")._value.get()
        assert val == 1.0
