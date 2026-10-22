"""BGE cross-encoder reranker — runs locally, zero API cost.

Cross-encoder vs bi-encoder:
  Bi-encoder:    embed(query) · embed(doc) → cosine sim  (fast, approximate)
  Cross-encoder: model([CLS] query [SEP] doc [SEP]) → relevance float  (slow, accurate)

BAAI/bge-reranker-base is 278 MB and runs on CPU or MPS (Apple Silicon).
model.predict() is blocking, so it runs in a thread pool executor.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sentence_transformers import CrossEncoder

from retrieval.base import RetrievedDoc
from retrieval.reranker.base_reranker import BaseReranker

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bge-rerank")
_DEFAULT_MODEL = "BAAI/bge-reranker-base"


class BGEReranker(BaseReranker):
    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    async def warm_up(self) -> None:
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            _EXECUTOR, lambda: CrossEncoder(self.model_name)
        )

    async def rerank(
        self,
        query: str,
        docs: list[RetrievedDoc],
        top_n: int = 5,
    ) -> list[RetrievedDoc]:
        if not docs:
            return docs
        model = self._get_model()
        pairs = [(query, doc.text) for doc in docs]
        loop = asyncio.get_running_loop()
        scores: Any = await loop.run_in_executor(
            _EXECUTOR, lambda: model.predict(pairs)
        )
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        result = []
        for i, (doc, score) in enumerate(ranked[:top_n], start=1):
            result.append(
                RetrievedDoc(
                    doc_id=doc.doc_id,
                    score=float(score),
                    text=doc.text,
                    rank=i,
                    metadata={**doc.metadata, "rerank_score": float(score)},
                )
            )
        return result

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model
