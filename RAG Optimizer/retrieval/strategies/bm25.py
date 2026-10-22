"""Elasticsearch BM25 retriever."""

from __future__ import annotations

import time
from typing import Any

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from retrieval.base import BaseRetriever, QueryType, RetrievedDoc, RetrievalResult
from config.settings import get_settings

_INDEX_SETTINGS = {
    "settings": {
        "analysis": {
            "analyzer": {
                "english_analyzer": {
                    "type": "standard",
                    "stopwords": "_english_",
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "english_analyzer"},
            "doc_id": {"type": "keyword"},
        }
    },
}


class BM25Retriever(BaseRetriever):
    strategy_name = "bm25"
    query_type = QueryType.KEYWORD

    def __init__(
        self,
        index: str = "rag_bm25",
        es_url: str | None = None,
    ) -> None:
        self.index = index
        self.es_url = es_url or get_settings().elasticsearch_url
        self._client: AsyncElasticsearch | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warm_up(self) -> None:
        if self._client is not None:
            return  # already connected
        self._client = AsyncElasticsearch(self.es_url)
        await self._ensure_index()

    async def _ensure_index(self) -> None:
        client = self._get_client()
        if not await client.indices.exists(index=self.index):
            await client.indices.create(index=self.index, body=_INDEX_SETTINGS)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start = time.perf_counter()
        resp = await self._get_client().search(
            index=self.index,
            body={
                "query": {"match": {"text": {"query": query, "operator": "or"}}},
                "size": top_k,
            },
        )
        hits = resp["hits"]["hits"]
        max_score = resp["hits"]["max_score"] or 1.0

        docs = [
            RetrievedDoc(
                doc_id=hit["_source"].get("doc_id", hit["_id"]),
                # normalise to [0, 1] so scores are comparable across queries
                score=hit["_score"] / max_score,
                text=hit["_source"].get("text", ""),
                metadata={
                    k: v for k, v in hit["_source"].items() if k not in ("text", "doc_id")
                },
            )
            for hit in hits
        ]
        return self._make_result(query, docs, start)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        await self._ensure_index()

        def _gen():
            for doc in documents:
                yield {
                    "_index": self.index,
                    "_id": str(doc["id"]),
                    "_source": {
                        "doc_id": str(doc["id"]),
                        "text": doc["text"],
                        **{k: v for k, v in doc.items() if k not in ("id", "text")},
                    },
                }

        await async_bulk(self._get_client(), _gen())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> AsyncElasticsearch:
        if self._client is None:
            self._client = AsyncElasticsearch(self.es_url)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
