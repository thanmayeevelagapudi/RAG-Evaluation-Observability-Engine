"""LLM judge — holistic quality scoring on a 4-dimension rubric.

Complements RAGAS: RAGAS catches hallucinations; the judge catches answers
that are technically faithful but unhelpful or incomplete.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import anthropic

log = logging.getLogger(__name__)

_MODEL = "claude-3-5-haiku-20241022"

_JUDGE_PROMPT = """\
You are an expert evaluator of question-answering systems.

Score this answer on 4 dimensions, each from 1 to 5.

Query: {query}

Retrieved context:
{context}

Answer: {answer}

Respond ONLY with a JSON object, no explanation:
{{
  "relevance": <1-5>,
  "completeness": <1-5>,
  "groundedness": <1-5>,
  "clarity": <1-5>,
  "reasoning": "<one sentence explaining the scores>"
}}\
"""

# Relevance weighted highest — an off-topic answer is always wrong
_WEIGHTS = {"relevance": 0.35, "completeness": 0.25, "groundedness": 0.25, "clarity": 0.15}


def _clamp(value: int | float) -> int:
    return max(1, min(5, int(round(value))))


def _composite(relevance: int, completeness: int, groundedness: int, clarity: int) -> float:
    return (
        relevance    * _WEIGHTS["relevance"] +
        completeness * _WEIGHTS["completeness"] +
        groundedness * _WEIGHTS["groundedness"] +
        clarity      * _WEIGHTS["clarity"]
    ) / 5.0


@dataclass
class JudgeScores:
    relevance:    int
    completeness: int
    groundedness: int
    clarity:      int
    reasoning:    str
    composite:    float
    query:        str = ""
    strategy:     str = ""


class LLMJudge:
    def __init__(self, anthropic_client: anthropic.AsyncAnthropic | None = None) -> None:
        self._client = anthropic_client or anthropic.AsyncAnthropic()

    async def judge(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> JudgeScores:
        joined_context = "\n\n".join(f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts))
        try:
            msg = await self._client.messages.create(
                model=_MODEL,
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": _JUDGE_PROMPT.format(
                        query=query, context=joined_context, answer=answer
                    ),
                }],
            )
            return self._parse(msg.content[0].text, query)
        except Exception as exc:
            log.warning("LLMJudge API call failed (%s)", exc)
            return self._zero(query)

    def _parse(self, raw: str, query: str) -> JudgeScores:
        # Strip optional ```json fences before parsing
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            data = json.loads(cleaned)
            r = _clamp(data["relevance"])
            co = _clamp(data["completeness"])
            g = _clamp(data["groundedness"])
            cl = _clamp(data["clarity"])
            return JudgeScores(
                relevance=r,
                completeness=co,
                groundedness=g,
                clarity=cl,
                reasoning=str(data.get("reasoning", ""))[:500],
                composite=_composite(r, co, g, cl),
                query=query,
            )
        except Exception:
            return self._zero(query, reasoning="parse_error")

    @staticmethod
    def _zero(query: str = "", reasoning: str = "parse_error") -> JudgeScores:
        return JudgeScores(
            relevance=0,
            completeness=0,
            groundedness=0,
            clarity=0,
            reasoning=reasoning,
            composite=0.0,
            query=query,
        )
