"""Abstract base for all rerankers.

Rerankers differ from retrievers: they never touch an index.
They take (query, candidate docs) and return a reordered subset.
Cross-encoders score query+doc jointly — slower but more accurate than
the bi-encoder cosine similarity used in dense retrieval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from retrieval.base import RetrievedDoc


class BaseReranker(ABC):

    @abstractmethod
    async def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        top_n: int = 5,
    ) -> list[RetrievedDoc]:
        """Reorder docs by relevance to query; return the top_n most relevant."""
        ...

    async def warm_up(self) -> None:
        """Optional: pre-load models before benchmarking."""
