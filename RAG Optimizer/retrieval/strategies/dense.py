"""BGE dense retriever backed by Qdrant."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from retrieval.base import BaseRetriever, QueryType, RetrievedDoc, RetrievalResult
from retrieval.utils import stable_qdrant_id
from config.settings import get_settings

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bge-encode")
log = logging.getLogger(__name__)

# One lock per DenseRetriever instance prevents concurrent warm_up() calls from loading
# the SentenceTransformer model multiple times (the idempotency check is not race-safe).
_WARM_UP_LOCKS: dict[int, asyncio.Lock] = {}


class DenseRetriever(BaseRetriever):
    strategy_name = "dense"
    query_type = QueryType.SEMANTIC

    def __init__(
        self,
        collection: str = "rag_dense",
        model_name: str = "BAAI/bge-small-en-v1.5",
        qdrant_url: str | None = None,
        vector_size: int = 384,
    ) -> None:
        self.collection = collection
        self.model_name = model_name
        self.qdrant_url = qdrant_url or get_settings().qdrant_url
        self.vector_size = vector_size
        self._model: SentenceTransformer | None = None
        self._client: AsyncQdrantClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warm_up(self) -> None:
        # Acquire per-instance lock so concurrent callers (hybrid, hyde, multi_query all share
        # this object) don't each load the 400MB model simultaneously → OOM on 2GB containers.
        lock = _WARM_UP_LOCKS.setdefault(id(self), asyncio.Lock())
        async with lock:
            if self._model is not None and self._client is not None:
                return  # already warmed up; skip expensive model reload
            loop = asyncio.get_running_loop()
            self._model = await loop.run_in_executor(
                _EXECUTOR, lambda: SentenceTransformer(self.model_name)
            )
            self._client = AsyncQdrantClient(url=self.qdrant_url)
            await self._ensure_collection()

    async def _ensure_collection(self) -> None:
        client = self._get_client()
        exists = await client.collection_exists(self.collection)
        if not exists:
            await client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start = time.perf_counter()
        embedding = await self._encode(query)
        hits = await self._vector_search(embedding, top_k=top_k)
        docs = [
            RetrievedDoc(
                # payload["doc_id"] holds the original string ID; hit.id is the stable integer hash
                doc_id=hit.payload.get("doc_id", str(hit.id)) if hit.payload else str(hit.id),
                score=float(hit.score),
                text=hit.payload.get("text", "") if hit.payload else "",
                metadata=hit.payload or {},
            )
            for hit in hits
        ]
        return self._make_result(query, docs, start)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        await self._ensure_collection()
        texts = [d["text"] for d in documents]
        embeddings = await self._encode_batch(texts)
        points = [
            qmodels.PointStruct(
                id=stable_qdrant_id(d["id"]),
                vector=embeddings[i].tolist(),
                payload={"doc_id": str(d["id"]), **{k: v for k, v in d.items() if k != "id"}},
            )
            for i, d in enumerate(documents)
        ]
        await self._get_client().upsert(collection_name=self.collection, points=points)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def embed_text(self, text: str) -> list[float]:
        """Public interface for embedding a single text (used by HyDE)."""
        return await self._encode(text)

    async def retrieve_by_embedding(
        self, embedding: list[float], top_k: int = 10
    ) -> list[RetrievedDoc]:
        """Search Qdrant with a pre-computed vector (used by HyDE)."""
        hits = await self._vector_search(embedding, top_k=top_k)
        return [
            RetrievedDoc(
                doc_id=hit.payload.get("doc_id", str(hit.id)) if hit.payload else str(hit.id),
                score=float(hit.score),
                text=hit.payload.get("text", "") if hit.payload else "",
                metadata=hit.payload or {},
            )
            for hit in hits
        ]

    async def _vector_search(self, embedding: list[float], top_k: int) -> list[Any]:
        """Compat shim for qdrant-client APIs across versions.

        Older clients expose `search(...)`; newer clients prefer `query_points(...)`.
        """
        client = self._get_client()
        if hasattr(client, "search"):
            return await client.search(
                collection_name=self.collection,
                query_vector=embedding,
                limit=top_k,
                with_payload=True,
            )

        response = await client.query_points(
            collection_name=self.collection,
            query=embedding,
            limit=top_k,
            with_payload=True,
        )

        # qdrant-client >= 1.11 returns QueryResponse(points=[...]).
        points = getattr(response, "points", None)
        if points is not None:
            return list(points)

        # Defensive fallback for any alternative response wrapper.
        result = getattr(response, "result", None)
        if result is not None:
            return list(result)

        if isinstance(response, list):
            return response
        raise TypeError(f"Unexpected Qdrant response type: {type(response)!r}")

    async def _encode(self, text: str) -> list[float]:
        model = self._get_model()
        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(
            _EXECUTOR, lambda: model.encode(text, normalize_embeddings=True)
        )
        return vec.tolist()

    async def _encode_batch(self, texts: list[str]) -> Any:
        model = self._get_model()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _EXECUTOR,
            lambda: model.encode(
                texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
            ),
        )

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            # warm_up() was not called — loading model synchronously blocks the event loop
            log.warning("SentenceTransformer loaded outside warm_up(); call warm_up() first")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(url=self.qdrant_url)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

