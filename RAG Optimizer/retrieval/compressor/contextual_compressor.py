"""Contextual compressor — strips irrelevant sentences before the LLM sees retrieved docs.

A good run should average compression_ratio ≈ 0.3–0.5 (keeping 30–50 % of tokens
with minimal quality loss). Docs already under max_tokens_per_doc are skipped — no
point paying for an API call that returns the full document unchanged.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

import anthropic

from retrieval.base import RetrievedDoc

_COMPRESS_PROMPT = """\
Given this query and document, extract ONLY the sentences that are directly \
relevant to answering the query.
If no sentences are relevant, return an empty string.
Do not paraphrase — copy the relevant sentences exactly.

Query: {query}

Document:
{document_text}

Relevant sentences only:\
"""

_MODEL = "claude-3-5-haiku-20241022"


def _approx_tokens(text: str) -> int:
    """Word-count heuristic: 1 word ≈ 1.3 tokens. Avoids tiktoken import."""
    return max(1, int(len(text.split()) * 1.3))


def _first_two_sentences(text: str) -> str:
    """Fallback when compression returns empty — never hand the LLM a blank context."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(parts) >= 2:  # actual sentence boundary was found
        return " ".join(parts[:2])
    return (parts[0] if parts else text)[:400]  # no boundary — hard truncate


@dataclass
class CompressedDoc:
    original: RetrievedDoc
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float  # compressed / original; lower = more aggressive


class ContextualCompressor:
    def __init__(self, anthropic_client: anthropic.AsyncAnthropic | None = None) -> None:
        self._client = anthropic_client or anthropic.AsyncAnthropic()

    async def compress(
        self,
        query: str,
        docs: list[RetrievedDoc],
        max_tokens_per_doc: int = 200,
    ) -> list[CompressedDoc]:
        """Compress all docs concurrently. Docs already under the limit are passed through."""
        return list(
            await asyncio.gather(
                *[self._compress_one(query, doc, max_tokens_per_doc) for doc in docs]
            )
        )

    async def _compress_one(
        self, query: str, doc: RetrievedDoc, max_tokens: int
    ) -> CompressedDoc:
        original_tokens = _approx_tokens(doc.text)

        if original_tokens <= max_tokens:
            return CompressedDoc(
                original=doc,
                compressed_text=doc.text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
            )

        try:
            message = await self._client.messages.create(
                model=_MODEL,
                max_tokens=max_tokens,
                messages=[{
                    "role": "user",
                    "content": _COMPRESS_PROMPT.format(
                        query=query, document_text=doc.text
                    ),
                }],
            )
            compressed = message.content[0].text.strip()
        except Exception:
            compressed = ""

        if not compressed:
            compressed = _first_two_sentences(doc.text)

        compressed_tokens = _approx_tokens(compressed)
        return CompressedDoc(
            original=doc,
            compressed_text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens,
        )
