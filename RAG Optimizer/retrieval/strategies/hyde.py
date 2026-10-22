"""HyDE — Hypothetical Document Embeddings.

Generates a fake answer to the query with Claude, embeds that answer,
then searches the corpus for real documents similar to the fake answer.

Why it works: questions and answers live in different embedding spaces.
Embedding a plausible answer searches the document space directly.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import anthropic

from retrieval.base import BaseRetriever, QueryType, RetrievedDoc, RetrievalResult
from retrieval.strategies.dense import DenseRetriever

log = logging.getLogger(__name__)

_HYDE_PROMPT = """\
You are a document from a knowledge base.
Given this question, write a short passage (2-3 sentences) that directly \
answers it, as if you were a relevant document.

Question: {query}

Write only the answer passage, no preamble:\
"""

_HYDE_MODEL = "claude-3-5-haiku-20241022"


class HyDERetriever(BaseRetriever):
    strategy_name = "hyde"
    query_type = QueryType.HYPOTHETICAL
    supported_query_types = frozenset({
        QueryType.SEMANTIC, QueryType.ABSTRACTIVE, QueryType.MULTI_HOP
    })

    def __init__(
        self,
        dense: DenseRetriever,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.dense = dense
        self._client = anthropic_client or anthropic.AsyncAnthropic()

    async def warm_up(self) -> None:
        await self.dense.warm_up()

    async def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start = time.perf_counter()
        try:
            hypothesis = await self._generate_hypothesis(query)
            embedding = await self.dense.embed_text(hypothesis)
            docs = await self.dense.retrieve_by_embedding(embedding, top_k=top_k)
            return self._make_result(
                query, docs, start, metadata={"hypothesis": hypothesis}
            )
        except Exception as exc:
            log.warning("HyDE hypothesis failed (%s); falling back to dense", exc)
            fallback = await self.dense.retrieve(query, top_k=top_k)
            return self._make_result(
                query, fallback.docs, start, metadata={"fallback": True}
            )

    async def _generate_hypothesis(self, query: str) -> str:
        message = await self._client.messages.create(
            model=_HYDE_MODEL,
            max_tokens=128,
            temperature=0.0,
            messages=[{"role": "user", "content": _HYDE_PROMPT.format(query=query)}],
        )
        return message.content[0].text.strip()

    async def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        await self.dense.upsert_documents(documents)
