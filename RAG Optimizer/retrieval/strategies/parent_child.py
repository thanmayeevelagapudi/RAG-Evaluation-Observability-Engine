"""Parent-child retriever: search small child chunks, return full parent context.

Small chunks (≈128 words) are retrieved with high precision.
The parent text (≈512 words) is returned as context so the LLM gets
enough surrounding information to generate a good answer.

parent_text is stored directly in each child's Qdrant payload — one
collection, zero extra fetches at query time.
"""

from __future__ import annotations

import time
from typing import Any

from retrieval.base import BaseRetriever, QueryType, RetrievedDoc, RetrievalResult
from retrieval.strategies.dense import DenseRetriever

_CHILD_WORDS = 128
_PARENT_WORDS = 512


def _word_chunks(text: str, size: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + size]) for i in range(0, max(len(words), 1), size)]


class ParentChildRetriever(BaseRetriever):
    strategy_name = "parent_child"
    query_type = QueryType.PARENT_CHILD
    supported_query_types = frozenset({QueryType.FACTUAL, QueryType.MULTI_HOP})

    def __init__(self, dense: DenseRetriever) -> None:
        # dense should point to the child-chunk collection (e.g. "rag_dense_children")
        self.dense = dense

    async def warm_up(self) -> None:
        await self.dense.warm_up()

    async def close(self) -> None:
        await self.dense.close()

    async def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start = time.perf_counter()
        # Retrieve more children than needed; many will collapse to the same parent
        child_result = await self.dense.retrieve(query, top_k=top_k * 4)

        # Deduplicate by parent_id, keeping the child with the highest score
        best: dict[str, RetrievedDoc] = {}
        for child in child_result.docs:
            parent_id = child.metadata.get("parent_id", child.doc_id)
            parent_text = child.metadata.get("parent_text", child.text)
            if parent_id not in best or child.score > best[parent_id].score:
                best[parent_id] = RetrievedDoc(
                    doc_id=parent_id,
                    score=child.score,
                    text=parent_text,
                    metadata={"best_child_id": child.doc_id, "child_score": child.score},
                )

        parents = sorted(best.values(), key=lambda d: d.score, reverse=True)[:top_k]
        return self._make_result(query, parents, start)

    async def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        """Split each document into child chunks and index them with parent context."""
        child_docs: list[dict[str, Any]] = []
        for doc in documents:
            parent_id = str(doc["id"])
            parent_text = doc["text"]
            extra = {k: v for k, v in doc.items() if k not in ("id", "text")}

            # Non-overlapping parent windows; each window owns its child chunks
            parent_windows = _word_chunks(parent_text, _PARENT_WORDS)
            for p_idx, parent_window in enumerate(parent_windows):
                for c_idx, child_chunk in enumerate(_word_chunks(parent_window, _CHILD_WORDS)):
                    child_docs.append({
                        "id": f"{parent_id}_p{p_idx}_c{c_idx}",
                        "text": child_chunk,
                        "parent_id": parent_id,
                        "parent_text": parent_window,  # stored in payload for zero-latency lookup
                        **extra,
                    })

        await self.dense.upsert_documents(child_docs)
