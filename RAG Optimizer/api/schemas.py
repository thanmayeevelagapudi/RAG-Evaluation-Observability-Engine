"""Pydantic request / response models — no business logic here."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    force_strategy: str | None = None
    session_id: str | None = None
    include_traces: bool = False


class FeedbackRequest(BaseModel):
    query: str
    strategy: str
    query_type: str
    positive: bool
    session_id: str | None = None
    was_exploration: bool = False


class BenchmarkRequest(BaseModel):
    dataset: str = "nfcorpus"
    strategies: list[str] | None = None
    max_queries: int = Field(50, ge=1, le=500)  # hard cap for API endpoint
    k_values: list[int] = Field(default_factory=lambda: [5, 10, 100])


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class DocumentResponse(BaseModel):
    doc_id: str
    text: str
    score: float
    rank: int
    strategy: str
    metadata: dict[str, Any] = {}


class RAGASScoresResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    composite: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    documents: list[DocumentResponse]
    strategy_used: str
    query_type: str
    was_exploration: bool
    latency_ms: float
    evaluation_status: str = "queued"
    trace_id: str | None = None
    prompt_tokens_estimate: int | None = None
    completion_tokens_estimate: int | None = None
    total_tokens_estimate: int | None = None
    estimated_cost_usd: float | None = None


class StrategyScoresResponse(BaseModel):
    query_type: str
    scores: dict[str, float]
    sample_counts: dict[str, int]
    using_cold_start: bool


class BenchmarkResponse(BaseModel):
    dataset: str
    results: dict[str, dict[str, float]]
    best_strategy: str
    duration_seconds: float


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    elasticsearch: str
    postgres: str
    redis: str
