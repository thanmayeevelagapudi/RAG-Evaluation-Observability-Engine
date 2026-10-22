"""Hybrid retriever: dense + BM25 in parallel, merged with RRF."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from retrieval.base import BaseRetriever, QueryType, RetrievalResult
from retrieval.fusion.reciprocal_rank_fusion import reciprocal_rank_fusion
from retrieval.strategies.bm25 import BM25Retriever
from retrieval.strategies.dense import DenseRetriever

# Fetch 3× top_k candidates per strategy to give RRF enough material
_CANDIDATE_MULTIPLIER = 3


class HybridRetriever(BaseRetriever):
    strategy_name = "hybrid"
    query_type = QueryType.HYBRID
    supported_query_types = frozenset(QueryType)  # safe default for all types

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever) -> None:
        self.dense = dense
        self.bm25 = bm25

    async def warm_up(self) -> None:
        # warm_up must delegate to both sub-retrievers or models won't be loaded
        await asyncio.gather(self.dense.warm_up(), self.bm25.warm_up())

    async def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start = time.perf_counter()
        candidates = top_k * _CANDIDATE_MULTIPLIER

        dense_result, bm25_result = await asyncio.gather(
            self.dense.retrieve(query, top_k=candidates),
            self.bm25.retrieve(query, top_k=candidates),
        )

        merged = reciprocal_rank_fusion(
            [dense_result.docs, bm25_result.docs],
            top_n=top_k,
        )
        return self._make_result(
            query,
            merged,
            start,
            metadata={
                "dense_latency_ms": dense_result.latency_ms,
                "bm25_latency_ms": bm25_result.latency_ms,
                "dense_candidates": candidates,
                "bm25_candidates": candidates,
            },
        )

    async def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        await asyncio.gather(
            self.dense.upsert_documents(documents),
            self.bm25.upsert_documents(documents),
        )

    async def close(self) -> None:
        await self.bm25.close()
