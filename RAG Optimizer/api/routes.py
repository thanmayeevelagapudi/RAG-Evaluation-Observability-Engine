"""API route handlers — thin wrappers that delegate to domain objects.

No business logic lives here. Each handler: validate → call domain → return schema.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time

import asyncpg
from fastapi import APIRouter, Depends, HTTPException

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
    BenchmarkResponse,
    DocumentResponse,
    FeedbackRequest,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    StrategyScoresResponse,
)
from benchmarks.beir_eval import BEIREvaluator
from benchmarks.llm_judge import LLMJudge
from benchmarks.ragas_scorer import RAGASScorer
from config.settings import get_settings
from feedback.collector import FeedbackCollector
from observability.metrics_collector import PrometheusMetrics
from observability.tracer import LangfuseTracer
from retrieval.base import QueryType
from retrieval.compressor.contextual_compressor import ContextualCompressor
from router.registry import StrategyRegistry
from router.router import AdaptiveRouter

log = logging.getLogger(__name__)
router = APIRouter()


def _approx_tokens(text: str) -> int:
    # Fast deterministic estimate (close enough for live cost telemetry).
    return max(1, int(len(text.split()) * 1.3))


def _estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    settings = get_settings()
    prompt_cost = (prompt_tokens / 1000.0) * settings.input_cost_per_1k_tokens_usd
    completion_cost = (completion_tokens / 1000.0) * settings.output_cost_per_1k_tokens_usd
    return round(prompt_cost + completion_cost, 6)


# ---------------------------------------------------------------------------
# POST /query — main pipeline: retrieve → compress → generate
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def query_endpoint(
    request: QueryRequest,
    rag_router: AdaptiveRouter = Depends(get_router),
    compressor: ContextualCompressor = Depends(get_compressor),
    generator=Depends(get_generator),
    scorer: RAGASScorer = Depends(get_scorer),
    judge: LLMJudge = Depends(get_judge),
    tracer: LangfuseTracer = Depends(get_tracer),
    metrics: PrometheusMetrics = Depends(get_metrics),
    db_pool=Depends(get_db_pool),
) -> QueryResponse:
    start = time.perf_counter()
    trace = tracer.create_trace(request.query, request.session_id) if request.include_traces else None

    try:
        router_result = await rag_router.route(
            request.query,
            top_k=request.top_k,
            force_strategy=request.force_strategy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if trace is not None:
        trace.log_retrieval(router_result)

    compressed = await compressor.compress(request.query, router_result.retrieval_result.docs)
    contexts = [c.compressed_text for c in compressed]
    try:
        answer = await generator.generate(request.query, contexts)
    except Exception as exc:
        log.error("LLM generation failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="LLM generation is temporarily unavailable — please retry",
        )

    if trace is not None:
        trace.log_generation(request.query, answer, contexts, model="claude-3-5-sonnet-20241022")
        trace.end()

    # Scoring is slow (2–5 s) — fire and forget so the user gets the answer immediately
    evaluation_status = "skipped"
    if random.random() < get_settings().eval_sample_rate:
        asyncio.create_task(
            _score_and_store(request.query, answer, contexts, router_result, scorer, judge, db_pool)
        )
        evaluation_status = "queued"

    prompt_tokens = _approx_tokens(request.query + "\n" + "\n".join(contexts))
    completion_tokens = _approx_tokens(answer)
    total_tokens = prompt_tokens + completion_tokens

    e2e_ms = (time.perf_counter() - start) * 1_000
    try:
        metrics.record_query(router_result, e2e_ms)
    except Exception as exc:
        log.debug("Prometheus metric recording failed: %s", exc)

    return QueryResponse(
        query=request.query,
        answer=answer,
        documents=[
            DocumentResponse(
                doc_id=doc.doc_id,
                text=doc.text,
                score=doc.score,
                rank=doc.rank,
                strategy=router_result.selected_strategy,
                metadata=doc.metadata,
            )
            for doc in router_result.retrieval_result.docs
        ],
        strategy_used=router_result.selected_strategy,
        query_type=router_result.query_type.value,
        was_exploration=router_result.was_exploration,
        latency_ms=e2e_ms,
        evaluation_status=evaluation_status,
        trace_id=trace.trace_id if trace is not None else None,
        prompt_tokens_estimate=prompt_tokens,
        completion_tokens_estimate=completion_tokens,
        total_tokens_estimate=total_tokens,
        estimated_cost_usd=_estimate_cost_usd(prompt_tokens, completion_tokens),
    )


async def _score_and_store(
    query: str,
    answer: str,
    contexts: list[str],
    router_result,
    scorer: RAGASScorer,
    judge: LLMJudge,
    db_pool: asyncpg.Pool | None,
) -> None:
    """Background task: compute evaluation scores and store in Postgres."""
    try:
        ragas_scores, judge_scores = await asyncio.gather(
            scorer.score(query, answer, contexts),
            judge.judge(query, answer, contexts),
        )

        if db_pool is not None:
            await db_pool.execute(
                """INSERT INTO ragas_scores
                   (query, strategy, query_type, faithfulness, answer_relevancy,
                    context_precision, composite, was_exploration)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                query,
                router_result.selected_strategy,
                router_result.query_type.value,
                ragas_scores.faithfulness,
                ragas_scores.answer_relevancy,
                ragas_scores.context_precision,
                ragas_scores.composite,
                router_result.was_exploration,
            )
            await db_pool.execute(
                """INSERT INTO judge_scores
                   (query, strategy, query_type, relevance, completeness,
                    groundedness, clarity, composite, reasoning, was_exploration)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
                query,
                router_result.selected_strategy,
                router_result.query_type.value,
                judge_scores.relevance,
                judge_scores.completeness,
                judge_scores.groundedness,
                judge_scores.clarity,
                judge_scores.composite,
                judge_scores.reasoning,
                router_result.was_exploration,
            )
    except Exception as exc:
        log.warning("Background scoring failed: %s", exc)


# ---------------------------------------------------------------------------
# POST /feedback — explicit user signal
# ---------------------------------------------------------------------------

@router.post("/feedback", status_code=204, dependencies=[Depends(verify_api_key)])
async def feedback_endpoint(
    request: FeedbackRequest,
    collector: FeedbackCollector = Depends(get_collector),
) -> None:
    try:
        qt = QueryType(request.query_type)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown query_type: {request.query_type}")
    await collector.record_explicit(
        query=request.query,
        strategy=request.strategy,
        query_type=qt,
        positive=request.positive,
        was_exploration=request.was_exploration,
        session_id=request.session_id,
    )


# ---------------------------------------------------------------------------
# GET /strategies/scores — inspect what the bandit has learned
# ---------------------------------------------------------------------------

@router.get("/strategies/scores", response_model=StrategyScoresResponse, dependencies=[Depends(verify_api_key)])
async def strategy_scores(
    query_type: str = "semantic",
    registry: StrategyRegistry = Depends(get_registry),
) -> StrategyScoresResponse:
    try:
        qt = QueryType(query_type)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown query_type: {query_type}")

    scores, counts = await registry.get_weights(qt)
    min_samples = get_settings().bandit_min_samples
    using_cold_start = all(c < min_samples for c in counts.values())
    return StrategyScoresResponse(
        query_type=query_type,
        scores=scores,
        sample_counts=counts,
        using_cold_start=using_cold_start,
    )


# ---------------------------------------------------------------------------
# POST /benchmark — run BEIR eval
# ---------------------------------------------------------------------------

@router.post("/benchmark", response_model=BenchmarkResponse, dependencies=[Depends(verify_api_key)])
async def benchmark_endpoint(
    request: BenchmarkRequest,
    rag_router: AdaptiveRouter = Depends(get_router),
) -> BenchmarkResponse:
    start = time.perf_counter()
    evaluator = BEIREvaluator(router=rag_router)
    try:
        results = await evaluator.evaluate(
            dataset=request.dataset,
            strategies=request.strategies,
            k_values=request.k_values,
            max_queries=request.max_queries,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    best = max(results, key=lambda s: results[s].primary()) if results else ""
    return BenchmarkResponse(
        dataset=request.dataset,
        results={s: m.metrics for s, m in results.items()},
        best_strategy=best,
        duration_seconds=(time.perf_counter() - start),
    )


# ---------------------------------------------------------------------------
# GET /health — readiness probe
# ---------------------------------------------------------------------------

@router.get("/health")
async def health(
    db_pool=Depends(get_db_pool),
    redis_client=Depends(get_redis_client),
) -> dict:
    qdrant_ok   = await _ping_qdrant()
    es_ok       = await _ping_es()
    pg_ok       = await _ping_postgres(db_pool)
    redis_ok    = await _ping_redis(redis_client)

    services = {"qdrant": qdrant_ok, "elasticsearch": es_ok,
                "postgres": pg_ok, "redis": redis_ok}
    status = "ok" if all(v == "ok" for v in services.values()) else "degraded"
    return {"status": status, **services}


async def _ping_qdrant() -> str:
    try:
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(url=get_settings().qdrant_url)
        await client.get_collections()
        await client.close()
        return "ok"
    except Exception:
        return "unavailable"


async def _ping_es() -> str:
    try:
        from elasticsearch import AsyncElasticsearch
        client = AsyncElasticsearch(get_settings().elasticsearch_url)
        await client.ping()
        await client.close()
        return "ok"
    except Exception:
        return "unavailable"


async def _ping_postgres(db_pool) -> str:
    if db_pool is None:
        return "unavailable"
    try:
        await db_pool.fetchval("SELECT 1")
        return "ok"
    except Exception:
        return "unavailable"


async def _ping_redis(redis_client) -> str:
    if redis_client is None:
        return "unavailable"
    try:
        await redis_client.ping()
        return "ok"
    except Exception:
        return "unavailable"
