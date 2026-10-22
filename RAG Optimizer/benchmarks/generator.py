"""RAG answer generator — calls Claude with retrieved + compressed context."""

from __future__ import annotations

import anthropic

_GENERATE_PROMPT = """\
Answer the question using ONLY the provided context.
If the context doesn't contain enough information, say so explicitly.
Do not use any knowledge outside the provided context.

Context:
{contexts}

Question: {query}

Answer:\
"""

_MODEL = "claude-3-5-sonnet-20241022"


class RAGGenerator:
    def __init__(self, anthropic_client: anthropic.AsyncAnthropic | None = None) -> None:
        self._client = anthropic_client or anthropic.AsyncAnthropic()

    async def generate(
        self,
        query: str,
        contexts: list[str],
        max_tokens: int = 512,
    ) -> str:
        """Generate an answer grounded in the provided context chunks."""
        numbered = "\n\n".join(f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts))
        message = await self._client.messages.create(
            model=_MODEL,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": _GENERATE_PROMPT.format(contexts=numbered, query=query),
            }],
        )
        return message.content[0].text.strip()
