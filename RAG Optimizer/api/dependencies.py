"""Dependency injection — builds the entire object graph once at startup.

All route handlers receive objects via Depends(); they never instantiate anything.
The module-level _state dict is the single source of live singletons.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic
import asyncpg
import redis.asyncio as aioredis
from fastapi import Header, HTTPException

from benchmarks.generator import RAGGenerator
from benchmarks.llm_judge import LLMJudge
from benchmarks.ragas_scorer import RAGASScorer
from config.settings import get_settings
from feedback.collector import FeedbackCollector
from feedback.updater import FeedbackUpdater
from observability.metrics_collector import PrometheusMetrics
from observability.tracer import LangfuseTracer
from retrieval.reranker.bge_reranker import BGEReranker
from retrieval.compressor.contextual_compressor import ContextualCompressor
from retrieval.reranker.cohere_reranker import CohereReranker
from retrieval.strategies.bm25 import BM25Retriever
from retrieval.strategies.dense import DenseRetriever
from retrieval.strategies.hybrid import HybridRetriever
from retrieval.strategies.hyde import HyDERetriever
from retrieval.strategies.multi_query import MultiQueryRetriever
from retrieval.strategies.parent_child import ParentChildRetriever
from router.bandit import EpsilonGreedyBandit
from router.classifier import QueryClassifier
from router.registry import StrategyRegistry
from router.router import AdaptiveRouter

log = logging.getLogger(__name__)

_state: dict[str, Any] = {}


def _build_reranker(settings) -> CohereReranker | BGEReranker | None:
    provider = settings.reranker_provider.strip().lower()

    if provider == "none":
        log.info("Reranker disabled (RERANKER_PROVIDER=none)")
        return None

    if provider == "bge":
        log.info("Using local BGE reranker (%s)", settings.bge_reranker_model)
        return BGEReranker(model_name=settings.bge_reranker_model)

    if provider == "cohere":
        if not settings.cohere_api_key:
            log.warning(
                "RERANKER_PROVIDER=cohere but COHERE_API_KEY is missing; "
                "falling back to local BGE reranker"
            )
            return BGEReranker(model_name=settings.bge_reranker_model)
        log.info("Using Cohere reranker")
        return CohereReranker(api_key=settings.cohere_api_key)

    # auto (default): choose managed API when available, otherwise local model
    if settings.cohere_api_key:
        log.info("Using Cohere reranker (auto)")
        return CohereReranker(api_key=settings.cohere_api_key)

    log.info("Using local BGE reranker (auto fallback, no COHERE_API_KEY)")
    return BGEReranker(model_name=settings.bge_reranker_model)


async def init_dependencies() -> None:
    """Build every object in dependency order. Called once from the FastAPI lifespan."""
    settings = get_settings()

    anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    redis_client = aioredis.from_url(settings.redis_url, decode_responses=False)

    try:
        db_pool: asyncpg.Pool | None = await asyncpg.create_pool(settings.postgres_dsn)
    except Exception as exc:
        log.warning("Postgres unavailable (%s); registry will use cold-start defaults", exc)
        db_pool = None

    # Base retrievers — shared instances injected into composite strategies
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    child_dense = DenseRetriever(collection="rag_dense_children")

    strategies: dict[str, Any] = {
        "dense":        dense,
        "bm25":         bm25,
        "hybrid":       HybridRetriever(dense=dense, bm25=bm25),
        "hyde":         HyDERetriever(dense=dense, anthropic_client=anthropic_client),
        "multi_query":  MultiQueryRetriever(dense=dense, anthropic_client=anthropic_client),
        "parent_child": ParentChildRetriever(dense=child_dense),
    }

    reranker = _build_reranker(settings)
    metrics   = PrometheusMetrics()
    registry  = StrategyRegistry(db_pool=db_pool, redis_client=redis_client)
    classifier = QueryClassifier(anthropic_client=anthropic_client, redis_client=redis_client)
    bandit     = EpsilonGreedyBandit()

    router = AdaptiveRouter(
        strategies=strategies,
        reranker=reranker,
        classifier=classifier,
        registry=registry,
        bandit=bandit,
    )
    await router.warm_up()

    compressor = ContextualCompressor(anthropic_client=anthropic_client)
    generator  = RAGGenerator(anthropic_client=anthropic_client)
    scorer     = RAGASScorer(anthropic_client=anthropic_client)
    judge      = LLMJudge(anthropic_client=anthropic_client)
    tracer     = LangfuseTracer()
    collector  = FeedbackCollector(redis_client=redis_client)
    updater    = FeedbackUpdater(
        db_pool=db_pool,
        redis_client=redis_client,
        registry=registry,
        metrics=metrics,
    )

    _state.update({
        "router":       router,
        "updater":      updater,
        "collector":    collector,
        "compressor":   compressor,
        "generator":    generator,
        "scorer":       scorer,
        "judge":        judge,
        "registry":     registry,
        "tracer":       tracer,
        "metrics":      metrics,
        "db_pool":      db_pool,
        "redis_client": redis_client,
    })
    log.info("All dependencies initialised")


async def close_dependencies() -> None:
    router = _state.get("router")
    if router:
        await router.close_all()
    db_pool = _state.get("db_pool")
    if db_pool:
        await db_pool.close()


# ---------------------------------------------------------------------------
# Getter functions — used via FastAPI Depends()
# ---------------------------------------------------------------------------

def get_router() -> AdaptiveRouter:
    return _state["router"]

def get_collector() -> FeedbackCollector:
    return _state["collector"]

def get_registry() -> StrategyRegistry:
    return _state["registry"]

def get_compressor() -> ContextualCompressor:
    return _state["compressor"]

def get_generator() -> RAGGenerator:
    return _state["generator"]

def get_scorer() -> RAGASScorer:
    return _state["scorer"]

def get_judge() -> LLMJudge:
    return _state["judge"]

def get_tracer() -> LangfuseTracer:
    return _state["tracer"]

def get_metrics() -> PrometheusMetrics:
    return _state["metrics"]

def get_updater() -> FeedbackUpdater:
    return _state["updater"]

def get_db_pool() -> asyncpg.Pool | None:
    return _state.get("db_pool")

def get_redis_client():
    return _state.get("redis_client")


async def verify_api_key(x_api_key: str | None = Header(None)) -> None:
    """Validate X-Api-Key header. No-op when API_KEY setting is empty (dev mode)."""
    key = get_settings().api_key
    if key and x_api_key != key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
