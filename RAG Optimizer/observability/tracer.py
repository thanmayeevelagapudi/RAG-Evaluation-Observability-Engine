"""Langfuse tracing — one TraceHandle per request, concurrency-safe.

When Langfuse keys are not configured, create_trace() returns _NoOpTraceHandle
so all call sites stay identical regardless of whether tracing is active.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config.settings import get_settings

if TYPE_CHECKING:
    from benchmarks.llm_judge import JudgeScores
    from benchmarks.ragas_scorer import RAGASScores
    from router.router import RouterResult

log = logging.getLogger(__name__)


class TraceHandle:
    """Per-request trace context — created by LangfuseTracer.create_trace()."""

    def __init__(self, trace) -> None:
        self._trace = trace
        self.trace_id: str | None = getattr(trace, "id", None)

    def log_retrieval(self, router_result: "RouterResult") -> None:
        try:
            self._trace.span(
                name="retrieval",
                input={"query": router_result.retrieval_result.query},
                output={
                    "strategy":        router_result.selected_strategy,
                    "query_type":      router_result.query_type.value,
                    "was_exploration": router_result.was_exploration,
                    "latency_ms":      router_result.retrieval_result.latency_ms,
                    "num_docs":        len(router_result.retrieval_result.docs),
                    "top_doc_ids": [
                        d.doc_id for d in router_result.retrieval_result.docs[:3]
                    ],
                },
            )
        except Exception as exc:
            log.debug("Langfuse retrieval span failed: %s", exc)

    def log_generation(
        self, query: str, answer: str, contexts: list[str], model: str
    ) -> None:
        try:
            self._trace.generation(
                name="rag-generation",
                model=model,
                input={"query": query, "num_contexts": len(contexts)},
                output=answer,
            )
        except Exception as exc:
            log.debug("Langfuse generation span failed: %s", exc)

    def log_scores(
        self,
        ragas: "RAGASScores | None" = None,
        judge: "JudgeScores | None" = None,
    ) -> None:
        try:
            if ragas:
                self._trace.score(name="faithfulness",      value=ragas.faithfulness)
                self._trace.score(name="context_precision", value=ragas.context_precision)
                self._trace.score(name="ragas_composite",   value=ragas.composite)
            if judge:
                self._trace.score(name="judge_composite", value=judge.composite)
        except Exception as exc:
            log.debug("Langfuse score logging failed: %s", exc)

    def end(self) -> None:
        try:
            self._trace.update(output={"status": "complete"})
        except Exception as exc:
            log.debug("Langfuse trace end failed: %s", exc)


class _NoOpTraceHandle:
    """All methods are no-ops — returned when Langfuse is not configured."""

    trace_id: str | None = None

    def log_retrieval(self, *_, **__) -> None:
        pass

    def log_generation(self, *_, **__) -> None:
        pass

    def log_scores(self, *_, **__) -> None:
        pass

    def end(self) -> None:
        pass


class LangfuseTracer:
    """Application-level singleton. create_trace() returns a per-request handle."""

    def __init__(self) -> None:
        settings = get_settings()
        self._enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
        self._client = None
        if self._enabled:
            try:
                from langfuse import Langfuse  # type: ignore[import]
                self._client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
            except Exception as exc:
                log.warning("Langfuse init failed (%s); tracing disabled", exc)
                self._enabled = False

    def create_trace(
        self, query: str, session_id: str | None = None
    ) -> TraceHandle | _NoOpTraceHandle:
        if not self._enabled or self._client is None:
            return _NoOpTraceHandle()
        try:
            trace = self._client.trace(
                name="rag-query",
                input={"query": query},
                session_id=session_id,
                tags=["production"],
            )
            return TraceHandle(trace)
        except Exception as exc:
            log.warning("Langfuse trace creation failed (%s)", exc)
            return _NoOpTraceHandle()
