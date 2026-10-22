"""Abstract base interface that every retrieval strategy must implement."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryType(str, Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    HYPOTHETICAL = "hypothetical"
    MULTI_QUERY = "multi_query"
    PARENT_CHILD = "parent_child"
    FACTUAL = "factual"
    ABSTRACTIVE = "abstractive"
    MULTI_HOP = "multi_hop"
    ENTITY = "entity"
    UNKNOWN = "unknown"


@dataclass
class RetrievedDoc:
    doc_id: str
    score: float
    text: str
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query: str
    docs: list[RetrievedDoc]
    strategy: str
    query_type: QueryType
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Every retrieval strategy inherits from this class."""

    strategy_name: str = "base"
    query_type: QueryType = QueryType.KEYWORD
    supported_query_types: frozenset[QueryType] = frozenset(QueryType)

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        ...

    @abstractmethod
    async def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        """Index a batch of documents. Each doc must have 'id' and 'text' keys."""
        ...

    async def warm_up(self) -> None:
        """Optional: pre-load models / warm connections before benchmarking."""

    def _make_result(
        self,
        query: str,
        docs: list[RetrievedDoc],
        start_time: float,
        metadata: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Stamp latency and assign 1-based ranks in one place."""
        latency_ms = (time.perf_counter() - start_time) * 1_000
        for i, doc in enumerate(docs, start=1):
            doc.rank = i
        return RetrievalResult(
            query=query,
            docs=docs,
            strategy=self.strategy_name,
            query_type=self.query_type,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
