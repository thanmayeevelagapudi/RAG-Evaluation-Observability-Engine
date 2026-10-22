"""Cohere Rerank API reranker.

Sends (query, candidate docs) to Cohere's cross-encoder endpoint.
Falls back to the original ordering if the API call fails (fail-open).

Free tier: 1 000 API calls/month — sufficient for local dev and benchmarking.
Rerank only the top-20 candidates, not the full retrieval set, to stay frugal.
"""

from __future__ import annotations

import logging
import os

import cohere

from retrieval.base import RetrievedDoc
from retrieval.reranker.base_reranker import BaseReranker

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "rerank-english-v3.0"


class CohereReranker(BaseReranker):
    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.getenv("COHERE_API_KEY", "")
        self._client: cohere.AsyncClientV2 | None = None

    async def warm_up(self) -> None:
        self._client = cohere.AsyncClientV2(self._api_key)

    async def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        top_n: int = 5,
    ) -> list[RetrievedDoc]:
        if not docs:
            return docs
        try:
            response = await self._get_client().rerank(
                model=self.model,
                query=query,
                documents=[doc.text for doc in docs],
                top_n=min(top_n, len(docs)),
            )
        except Exception as exc:
            log.warning("Cohere rerank failed (%s); returning original order", exc)
            return docs[:top_n]

        result = []
        for i, item in enumerate(response.results, start=1):
            original = docs[item.index]
            result.append(
                RetrievedDoc(
                    doc_id=original.doc_id,
                    score=item.relevance_score,
                    text=original.text,
                    rank=i,
                    metadata={**original.metadata, "rerank_score": item.relevance_score},
                )
            )
        return result

    def _get_client(self) -> cohere.AsyncClientV2:
        if self._client is None:
            self._client = cohere.AsyncClientV2(self._api_key)
        return self._client
