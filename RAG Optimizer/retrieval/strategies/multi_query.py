"""Multi-query retriever: expand query with Claude, retrieve per variant, merge with RRF.

Multiple paraphrases cast a wider semantic net — "ML model accuracy" also finds
documents that say "neural network performance" or "classifier precision".
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import anthropic

from retrieval.base import BaseRetriever, QueryType, RetrievedDoc, RetrievalResult
from retrieval.fusion.reciprocal_rank_fusion import reciprocal_rank_fusion
from retrieval.strategies.dense import DenseRetriever

log = logging.getLogger(__name__)

_EXPAND_PROMPT = """\
Generate {n} different search queries that would retrieve documents relevant \
to answering this question. Each query should approach the topic from a \
slightly different angle.

Original question: {query}

Return ONLY a JSON array of strings, no explanation:
["query 1", "query 2", "query 3"]\
"""

_EXPAND_MODEL = "claude-3-5-haiku-20241022"


class MultiQueryRetriever(BaseRetriever):
    strategy_name = "multi_query"
    query_type = QueryType.MULTI_QUERY
    supported_query_types = frozenset({
        QueryType.SEMANTIC, QueryType.MULTI_HOP, QueryType.ABSTRACTIVE
    })

    def __init__(
        self,
        dense: DenseRetriever,
        n_variants: int = 4,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.dense = dense
        self.n_variants = n_variants
        self._client = anthropic_client or anthropic.AsyncAnthropic()

    async def warm_up(self) -> None:
        await self.dense.warm_up()

    async def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start = time.perf_counter()
        variants = await self._expand_query(query)

        # All variant retrievals run in parallel
        raw_results = await asyncio.gather(
            *[self.dense.retrieve(v, top_k=top_k * 2) for v in variants],
            return_exceptions=True,
        )
        valid_lists = [r.docs for r in raw_results if not isinstance(r, Exception)]

        if not valid_lists:
            log.warning("All %d variant retrievals failed; falling back to dense", len(variants))
            fallback = await self.dense.retrieve(query, top_k=top_k)
            return self._make_result(
                query, fallback.docs, start, metadata={"fallback": True}
            )

        merged = reciprocal_rank_fusion(valid_lists, top_n=top_k)
        return self._make_result(
            query,
            merged,
            start,
            metadata={"variants": variants, "n_lists_merged": len(valid_lists)},
        )

    async def _expand_query(self, query: str) -> list[str]:
        prompt = _EXPAND_PROMPT.format(n=self.n_variants, query=query)
        try:
            message = await self._client.messages.create(
                model=_EXPAND_MODEL,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            variants = json.loads(raw)
            if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
                # Always include the original query; cap at n_variants expansions
                return [query, *variants[: self.n_variants]]
        except Exception as exc:
            log.warning("Query expansion failed (%s); using original query only", exc)
        return [query]

    async def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        await self.dense.upsert_documents(documents)
