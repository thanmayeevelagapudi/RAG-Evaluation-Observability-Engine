"""Prometheus counters, histograms, and gauges.

Pass a fresh CollectorRegistry() in tests to avoid duplicate-registration errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    REGISTRY,
)

if TYPE_CHECKING:
    from router.router import RouterResult


class PrometheusMetrics:
    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        self.queries_total = Counter(
            "rag_queries_total",
            "Total queries processed",
            labelnames=["strategy", "query_type", "was_exploration"],
            registry=registry,
        )
        self.exploration_total = Counter(
            "rag_exploration_total",
            "Total exploration decisions by bandit",
            labelnames=["strategy"],
            registry=registry,
        )
        self.retrieval_latency = Histogram(
            "rag_retrieval_latency_ms",
            "Retrieval latency in milliseconds",
            labelnames=["strategy"],
            buckets=[50, 100, 200, 400, 800, 1600, 3200],
            registry=registry,
        )
        self.e2e_latency = Histogram(
            "rag_e2e_latency_ms",
            "End-to-end query latency in milliseconds",
            buckets=[200, 400, 800, 1600, 3200, 6400],
            registry=registry,
        )
        self.strategy_score = Gauge(
            "rag_strategy_score",
            "Current avg score for strategy+query_type combination",
            labelnames=["strategy", "query_type"],
            registry=registry,
        )
        self.feedback_buffer_size = Gauge(
            "rag_feedback_buffer_size",
            "Current number of events in the Redis feedback buffer",
            registry=registry,
        )

    def record_query(self, router_result: "RouterResult", e2e_ms: float) -> None:
        self.queries_total.labels(
            strategy=router_result.selected_strategy,
            query_type=router_result.query_type.value,
            was_exploration=str(router_result.was_exploration),
        ).inc()
        if router_result.was_exploration:
            self.exploration_total.labels(strategy=router_result.selected_strategy).inc()
        self.retrieval_latency.labels(
            strategy=router_result.selected_strategy,
        ).observe(router_result.retrieval_result.latency_ms)
        self.e2e_latency.observe(e2e_ms)

    def update_strategy_score(self, strategy: str, query_type: str, score: float) -> None:
        self.strategy_score.labels(strategy=strategy, query_type=query_type).set(score)
