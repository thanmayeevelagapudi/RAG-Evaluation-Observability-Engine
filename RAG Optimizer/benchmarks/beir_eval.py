"""BEIR dataset evaluator — pure retrieval quality (nDCG, MRR, Recall).

No LLM generation here — only measures whether retrieval finds the right docs.
RAGAS/judge scoring is a separate pipeline that lives on top of this.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from benchmarks.metrics import compute_metrics, per_query_ndcg

log = logging.getLogger(__name__)

_DATA_DIR = Path("data")
_BEIR_URL_BASE = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
)


@dataclass
class MetricsResult:
    metrics: dict[str, float]  # keyed by "ndcg@10", "mrr", "recall@10", etc.
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    # Per-query nDCG@10 for paired significance tests; in-memory only, never written to baselines.
    per_query_ndcg10: dict[str, float] = field(default_factory=dict, repr=False)

    def primary(self) -> float:
        """nDCG@10 — the number used for CI regression comparisons."""
        return self.metrics.get("ndcg@10", 0.0)

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics,
            "avg_latency_ms": self.avg_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MetricsResult":
        return cls(
            metrics=d["metrics"],
            avg_latency_ms=d.get("avg_latency_ms", 0.0),
            p50_latency_ms=d.get("p50_latency_ms", 0.0),
            p95_latency_ms=d.get("p95_latency_ms", 0.0),
        )


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; 0.0 for an empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def check_regression(
    current: dict[str, MetricsResult],
    baseline: dict[str, MetricsResult],
    threshold: float = 0.02,
    latency_regression_pct: float | None = None,
) -> tuple[bool, str]:
    """Return (passed, message) for quality and optional latency regressions.

    Quality gate: fail if any strategy drops more than ``threshold`` nDCG@10.
    Latency gate: when ``latency_regression_pct`` is set, fail if avg latency grows
    beyond baseline * (1 + latency_regression_pct / 100).
    """
    issues: list[str] = []

    for strategy, cur in current.items():
        if strategy not in baseline:
            continue

        base = baseline[strategy]
        delta = cur.primary() - base.primary()
        if delta < -threshold:
            issues.append(
                f"REGRESSION: {strategy} nDCG@10 dropped {abs(delta):.3f} "
                f"(was {base.primary():.3f}, "
                f"now {cur.primary():.3f})"
            )

        if latency_regression_pct is not None and base.avg_latency_ms > 0:
            allowed = base.avg_latency_ms * (1.0 + latency_regression_pct / 100.0)
            if cur.avg_latency_ms > allowed:
                growth_pct = ((cur.avg_latency_ms / base.avg_latency_ms) - 1.0) * 100.0
                issues.append(
                    f"REGRESSION: {strategy} avg latency rose {growth_pct:.1f}% "
                    f"(was {base.avg_latency_ms:.1f} ms, now {cur.avg_latency_ms:.1f} ms; "
                    f"limit +{latency_regression_pct:.1f}%)"
                )

    if issues:
        return False, "\n".join(issues)
    return True, "All strategies within configured quality/latency thresholds"


class BEIREvaluator:
    def __init__(self, router) -> None:  # AdaptiveRouter; avoid circular import
        self._router = router

    async def evaluate(
        self,
        dataset: str,
        strategies: list[str] | None = None,
        k_values: list[int] | None = None,
        max_queries: int | None = None,
    ) -> dict[str, MetricsResult]:
        """Download (if needed) and evaluate all strategies on a BEIR dataset."""
        corpus, queries, qrels = self._load_dataset(dataset)
        return await self.evaluate_from_data(
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            strategies=strategies,
            k_values=k_values,
            max_queries=max_queries,
        )

    async def evaluate_from_data(
        self,
        corpus: dict,
        queries: dict,
        qrels: dict,
        strategies: list[str] | None = None,
        k_values: list[int] | None = None,
        max_queries: int | None = None,
    ) -> dict[str, MetricsResult]:
        """Evaluate with pre-loaded data — used by tests and the CLI."""
        if k_values is None:
            k_values = [5, 10, 100]

        all_strategies = list(self._router._strategies.keys())
        target = [s for s in (strategies or all_strategies) if s in all_strategies]

        query_items = list(queries.items())
        if max_queries:
            query_items = query_items[:max_queries]

        results: dict[str, MetricsResult] = {}
        for strategy in target:
            log.info("Evaluating strategy: %s (%d queries)", strategy, len(query_items))
            strategy_result = await self._evaluate_strategy(
                strategy, query_items, qrels, k_values
            )
            results[strategy] = strategy_result
            log.info(
                "[%s] nDCG@10=%.3f  MRR=%.3f",
                strategy,
                strategy_result.primary(),
                strategy_result.metrics.get("mrr", 0.0),
            )

        return results

    async def _evaluate_strategy(
        self,
        strategy: str,
        query_items: list[tuple[str, str]],
        qrels: dict,
        k_values: list[int],
    ) -> MetricsResult:
        semaphore = asyncio.Semaphore(10)
        latencies: list[float] = []
        # Without an explicit depth the router returns rerank_top_n (5) docs, which makes
        # nDCG@10 and Recall@100 meaningless; retrieve as deep as the deepest metric.
        depth = max(k_values)

        async def _one(query_id: str, query_text: str) -> tuple[str, dict]:
            async with semaphore:
                try:
                    result = await asyncio.wait_for(
                        self._router.route(
                            query_text, top_k=depth, force_strategy=strategy, rerank=False
                        ),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    log.warning("Query '%s' timed out for strategy %s — skipped", query_id, strategy)
                    return query_id, {}
                latencies.append(result.retrieval_result.latency_ms)
                return query_id, {d.doc_id: d.score for d in result.retrieval_result.docs}

        try:
            from tqdm.asyncio import tqdm_asyncio
            raw = await tqdm_asyncio.gather(
                *[_one(qid, qtxt) for qid, qtxt in query_items],
                desc=strategy,
            )
        except ImportError:
            raw = await asyncio.gather(*[_one(qid, qtxt) for qid, qtxt in query_items])

        run: dict[str, dict[str, float]] = dict(raw)
        metrics = compute_metrics(qrels, run, k_values=k_values)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return MetricsResult(
            metrics=metrics,
            avg_latency_ms=avg_latency,
            p50_latency_ms=_percentile(latencies, 50),
            p95_latency_ms=_percentile(latencies, 95),
            per_query_ndcg10=per_query_ndcg(qrels, run, k=10),
        )

    @staticmethod
    def _load_dataset(dataset: str) -> tuple[dict, dict, dict]:
        dataset_path = _DATA_DIR / dataset
        if not dataset_path.exists():
            try:
                from beir import util
                url = f"{_BEIR_URL_BASE}/{dataset}.zip"
                util.download_and_unzip(url, str(_DATA_DIR))
            except Exception as exc:
                raise RuntimeError(f"Cannot load dataset '{dataset}': {exc}") from exc
        try:
            from beir.datasets.data_loader import GenericDataLoader
            corpus, queries, qrels = GenericDataLoader(
                data_folder=str(dataset_path)
            ).load(split="test")
        except Exception as exc:
            raise RuntimeError(f"Failed to parse dataset '{dataset}': {exc}") from exc
        return corpus, queries, qrels
