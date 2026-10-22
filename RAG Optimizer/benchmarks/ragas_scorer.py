"""RAGAS-style scorer — faithfulness, answer relevancy, and context precision.

Implemented from scratch (no ragas library dependency) so it's transparent
and interview-explainable. Each metric is a set of concurrent Claude calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import anthropic

log = logging.getLogger(__name__)

_MODEL = "claude-3-5-haiku-20241022"

_CLAIMS_PROMPT = """\
List every distinct factual claim made in this answer as a JSON array of strings.
Include only specific, verifiable claims, not general statements.

Answer: {answer}

Return ONLY a JSON array, no explanation:
["claim 1", "claim 2", ...]\
"""

_VERIFY_PROMPT = """\
Does the following context support this claim?
Answer YES or NO only.

Claim: {claim}

Context:
{context}\
"""

_RELEVANCY_PROMPT = """\
Given this query and answer, score how well the answer directly addresses the query.
Score 0.0 to 1.0 (1.0 = perfectly addresses the query, 0.0 = completely off-topic).
Reply with ONLY a decimal number between 0.0 and 1.0.

Query: {query}
Answer: {answer}\
"""

_PRECISION_PROMPT = """\
Is this context chunk useful for answering the following query?
Query: {query}
Context chunk: {chunk}
Answer YES or NO only.\
"""


@dataclass
class RAGASScores:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float | None = None
    query: str = ""
    strategy: str = ""
    latency_ms: float = 0.0

    @property
    def composite(self) -> float:
        """Mean of the three primary metrics."""
        parts = [self.faithfulness, self.answer_relevancy, self.context_precision]
        valid = [s for s in parts if s >= 0.0]
        return sum(valid) / len(valid) if valid else 0.0


class RAGASScorer:
    def __init__(self, anthropic_client: anthropic.AsyncAnthropic | None = None) -> None:
        self._client = anthropic_client or anthropic.AsyncAnthropic()

    async def score(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGASScores:
        start = time.perf_counter()
        try:
            faithfulness, relevancy, precision = await asyncio.gather(
                self._faithfulness(answer, contexts),
                self._answer_relevancy(query, answer),
                self._context_precision(query, contexts),
            )
        except Exception as exc:
            log.warning("RAGASScorer failed (%s); returning zero scores", exc)
            return RAGASScores(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                query=query,
                latency_ms=(time.perf_counter() - start) * 1_000,
            )

        return RAGASScores(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            context_precision=precision,
            query=query,
            latency_ms=(time.perf_counter() - start) * 1_000,
        )

    # ------------------------------------------------------------------
    # Faithfulness
    # ------------------------------------------------------------------

    async def _faithfulness(self, answer: str, contexts: list[str]) -> float:
        claims = await self._extract_claims(answer)
        if not claims:
            return 1.0  # vacuous truth — no claims can be unfaithful

        joined_context = "\n\n".join(contexts)
        verdicts = await asyncio.gather(
            *[self._verify_claim(c, joined_context) for c in claims]
        )
        return sum(verdicts) / len(verdicts)

    async def _extract_claims(self, answer: str) -> list[str]:
        try:
            msg = await self._client.messages.create(
                model=_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": _CLAIMS_PROMPT.format(answer=answer)}],
            )
            raw = msg.content[0].text.strip()
            # Strip optional ```json fences
            raw = raw.strip("`").lstrip("json").strip()
            claims = json.loads(raw)
            return claims if isinstance(claims, list) else []
        except Exception:
            return []

    async def _verify_claim(self, claim: str, context: str) -> float:
        try:
            msg = await self._client.messages.create(
                model=_MODEL,
                max_tokens=5,
                messages=[{
                    "role": "user",
                    "content": _VERIFY_PROMPT.format(claim=claim, context=context),
                }],
            )
            return 1.0 if msg.content[0].text.strip().upper().startswith("YES") else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Answer relevancy
    # ------------------------------------------------------------------

    async def _answer_relevancy(self, query: str, answer: str) -> float:
        try:
            msg = await self._client.messages.create(
                model=_MODEL,
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": _RELEVANCY_PROMPT.format(query=query, answer=answer),
                }],
            )
            return max(0.0, min(1.0, float(msg.content[0].text.strip())))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Context precision
    # ------------------------------------------------------------------

    async def _context_precision(self, query: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        verdicts = await asyncio.gather(
            *[self._is_chunk_useful(query, chunk) for chunk in contexts]
        )
        return sum(verdicts) / len(verdicts)

    async def _is_chunk_useful(self, query: str, chunk: str) -> float:
        try:
            msg = await self._client.messages.create(
                model=_MODEL,
                max_tokens=5,
                messages=[{
                    "role": "user",
                    "content": _PRECISION_PROMPT.format(query=query, chunk=chunk),
                }],
            )
            return 1.0 if msg.content[0].text.strip().upper().startswith("YES") else 0.0
        except Exception:
            return 0.0
