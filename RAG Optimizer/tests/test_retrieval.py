"""Unit tests for retrieval base, metrics, and strategy logic (no live infra needed)."""

from __future__ import annotations

import math
import time
from typing import Any

import pytest

from benchmarks.metrics import (
    _dcg,
    _ideal_dcg,
    _mrr,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
    compute_metrics,
    per_query_ndcg,
)
from retrieval.base import BaseRetriever, QueryType, RetrievedDoc, RetrievalResult
from tests.conftest import SAMPLE_QRELS, SAMPLE_RESULTS


# ---------------------------------------------------------------------------
# Concrete stub used to test the abstract base class
# ---------------------------------------------------------------------------

class _StubRetriever(BaseRetriever):
    strategy_name = "stub"
    query_type = QueryType.KEYWORD

    async def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        start = time.perf_counter()
        docs = [
            RetrievedDoc(doc_id=f"d{i}", score=1.0 / i, text=f"doc {i}")
            for i in range(1, top_k + 1)
        ]
        return self._make_result(query, docs, start)

    async def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        pass


# ---------------------------------------------------------------------------
# BaseRetriever
# ---------------------------------------------------------------------------

class TestBaseRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_returns_result(self):
        result = await _StubRetriever().retrieve("test query", top_k=5)
        assert isinstance(result, RetrievalResult)
        assert len(result.docs) == 5
        assert result.query == "test query"
        assert result.strategy == "stub"

    @pytest.mark.asyncio
    async def test_ranks_are_one_based(self):
        result = await _StubRetriever().retrieve("test", top_k=3)
        assert [d.rank for d in result.docs] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_latency_is_positive(self):
        result = await _StubRetriever().retrieve("test")
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_warm_up_is_noop_by_default(self):
        await _StubRetriever().warm_up()  # must not raise


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_retrieved_doc_defaults(self):
        doc = RetrievedDoc(doc_id="x", score=0.5, text="hello")
        assert doc.rank == 0
        assert doc.metadata == {}

    def test_retrieval_result_metadata_defaults_empty(self):
        doc = RetrievedDoc(doc_id="x", score=1.0, text="t", rank=1)
        result = RetrievalResult(
            query="q",
            docs=[doc],
            strategy="bm25",
            query_type=QueryType.KEYWORD,
            latency_ms=12.5,
        )
        assert result.metadata == {}
        assert result.latency_ms == 12.5


# ---------------------------------------------------------------------------
# DCG primitives
# ---------------------------------------------------------------------------

class TestDCGPrimitives:
    def test_dcg_formula(self):
        expected = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
        assert _dcg([3, 2, 1], k=3) == pytest.approx(expected)

    def test_dcg_empty_list(self):
        assert _dcg([], k=10) == 0.0

    def test_ideal_dcg_sorts_descending(self):
        assert _ideal_dcg([1, 3, 2], k=3) == pytest.approx(_dcg([3, 2, 1], k=3))

    def test_ndcg_perfect_ranking_is_one(self):
        assert _ndcg_at_k(["d1", "d2", "d3"], {"d1": 2, "d2": 1}, k=3) == pytest.approx(1.0)

    def test_ndcg_no_relevant_docs(self):
        assert _ndcg_at_k(["d1"], {}, k=5) == 0.0

    def test_ndcg_wrong_order_below_one(self):
        score = _ndcg_at_k(["d2", "d1"], {"d1": 2, "d2": 1}, k=2)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------

class TestMRR:
    def test_first_hit_gives_mrr_one(self):
        assert _mrr(["d1", "d2", "d3"], {"d1": 1}) == pytest.approx(1.0)

    def test_second_hit_gives_mrr_half(self):
        assert _mrr(["d0", "d1", "d2"], {"d1": 1}) == pytest.approx(0.5)

    def test_no_hit_gives_zero(self):
        assert _mrr(["d0", "d1"], {"d9": 1}) == pytest.approx(0.0)

    def test_third_hit(self):
        assert _mrr(["d0", "d1", "d2"], {"d2": 1}) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Recall / Precision
# ---------------------------------------------------------------------------

class TestRecallPrecision:
    def test_recall_perfect(self):
        assert _recall_at_k(["d1", "d2"], {"d1": 1, "d2": 1}, k=2) == pytest.approx(1.0)

    def test_recall_partial(self):
        assert _recall_at_k(["d1", "d3"], {"d1": 1, "d2": 1}, k=2) == pytest.approx(0.5)

    def test_recall_no_relevant(self):
        assert _recall_at_k(["d1"], {}, k=5) == 0.0

    def test_precision_at_k1_perfect(self):
        assert _precision_at_k(["d1", "d2"], {"d1": 1}, k=1) == pytest.approx(1.0)

    def test_precision_at_k2_half(self):
        assert _precision_at_k(["d1", "d2"], {"d1": 1}, k=2) == pytest.approx(0.5)

    def test_precision_k0_zero(self):
        assert _precision_at_k(["d1"], {"d1": 1}, k=0) == 0.0


class TestPerQueryNdcg:
    def test_mean_matches_the_aggregate_metric(self):
        per_query = per_query_ndcg(SAMPLE_QRELS, SAMPLE_RESULTS, k=10)
        aggregate = compute_metrics(SAMPLE_QRELS, SAMPLE_RESULTS, k_values=[10])["ndcg@10"]
        assert sum(per_query.values()) / len(per_query) == pytest.approx(aggregate)

    def test_query_with_no_results_scores_zero(self):
        assert per_query_ndcg(SAMPLE_QRELS, {"q1": {}}, k=10) == {"q1": 0.0}

    def test_query_without_qrels_is_skipped(self):
        assert per_query_ndcg(SAMPLE_QRELS, {"unjudged": {"d1": 1.0}}) == {}

    def test_truncated_list_cannot_reach_full_ndcg_at_depth(self):
        # Regression: retrieving only 5 docs made the reported nDCG@10 a top-5 number.
        qrels = {"q": {f"d{i}": 1 for i in range(10)}}
        top5 = {"q": {f"d{i}": 1.0 - i / 100 for i in range(5)}}
        top10 = {"q": {f"d{i}": 1.0 - i / 100 for i in range(10)}}
        assert per_query_ndcg(qrels, top5)["q"] < per_query_ndcg(qrels, top10)["q"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_metrics integration
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_all_expected_keys(self):
        m = compute_metrics(SAMPLE_QRELS, SAMPLE_RESULTS, k_values=[5, 10])
        for key in ("ndcg@5", "ndcg@10", "recall@5", "recall@10", "mrr"):
            assert key in m, f"missing metric: {key}"

    def test_all_values_in_unit_interval(self):
        m = compute_metrics(SAMPLE_QRELS, SAMPLE_RESULTS)
        for key, val in m.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"

    def test_perfect_results_give_ndcg_one(self):
        perfect = {
            q_id: {doc_id: float(grade) for doc_id, grade in rel.items()}
            for q_id, rel in SAMPLE_QRELS.items()
        }
        m = compute_metrics(SAMPLE_QRELS, perfect, k_values=[1])
        assert m["ndcg@1"] == pytest.approx(1.0)

    def test_empty_results_give_all_zeros(self):
        m = compute_metrics(SAMPLE_QRELS, {})
        assert all(v == 0.0 for v in m.values())

    def test_default_k_values_include_10(self):
        m = compute_metrics(SAMPLE_QRELS, SAMPLE_RESULTS)
        assert "ndcg@10" in m
        assert "recall@10" in m

    def test_mrr_above_zero_for_relevant_results(self):
        m = compute_metrics(SAMPLE_QRELS, SAMPLE_RESULTS)
        assert m["mrr"] > 0.0
