"""Offline routing-policy evaluation: bootstrap CIs, cross-validated router, cost accounting, CLI glue."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from benchmarks import run_benchmark as rb
from benchmarks.beir_eval import MetricsResult
from benchmarks.router_eval import (
    cross_validated_router,
    evaluate_routing_policies,
    paired_bootstrap,
    render_markdown,
)
from retrieval.base import QueryType


def _type_structured(n_per_type: int = 20):
    """'semantic' queries are won by dense, 'factual' by bm25; hybrid and hyde are mediocre."""
    per_query: dict[str, dict[str, float]] = {"dense": {}, "bm25": {}, "hybrid": {}, "hyde": {}}
    types: dict[str, str] = {}
    for i in range(n_per_type):
        for prefix, qtype, winner in (("a", "semantic", "dense"), ("b", "factual", "bm25")):
            qid = f"{prefix}{i}"
            types[qid] = qtype
            for strategy in per_query:
                per_query[strategy][qid] = 0.8 if strategy == winner else 0.2
            per_query["hybrid"][qid] = 0.5
    return per_query, types


class TestPairedBootstrap:
    def test_detects_a_consistent_gain(self):
        a = [0.6 + (0.02 if i % 2 else -0.02) for i in range(40)]
        result = paired_bootstrap(a, [0.5] * 40, n_boot=2_000)
        assert result.mean_diff == pytest.approx(0.1)
        assert result.significant and result.ci_low > 0.0
        assert result.p_value < 0.05

    def test_null_difference_spans_zero(self):
        result = paired_bootstrap([0.1, 0.9] * 20, [0.9, 0.1] * 20, n_boot=2_000)
        assert result.mean_diff == pytest.approx(0.0)
        assert not result.significant
        assert result.p_value > 0.5

    def test_is_deterministic_for_a_seed(self):
        a, b = [0.2, 0.5, 0.9, 0.4], [0.1, 0.6, 0.7, 0.4]
        assert paired_bootstrap(a, b, n_boot=500, seed=7) == paired_bootstrap(a, b, n_boot=500, seed=7)

    def test_rejects_mismatched_or_empty_samples(self):
        with pytest.raises(ValueError):
            paired_bootstrap([0.1], [0.1, 0.2])
        with pytest.raises(ValueError):
            paired_bootstrap([], [])


class TestCrossValidatedRouter:
    def test_picks_the_winner_for_each_query_type(self):
        per_query, types = _type_structured()
        choice = cross_validated_router(per_query, types)
        assert all(choice[q] == "dense" for q in types if types[q] == "semantic")
        assert all(choice[q] == "bm25" for q in types if types[q] == "factual")

    def test_rare_types_fall_back_to_the_overall_best(self):
        per_query: dict[str, dict[str, float]] = {"dense": {}, "bm25": {}}
        types: dict[str, str] = {}
        for i in range(30):  # common type: dense wins
            types[f"c{i}"] = "semantic"
            per_query["dense"][f"c{i}"], per_query["bm25"][f"c{i}"] = 0.9, 0.1
        for i in range(2):  # rare type: bm25 wins, but two queries are too few to trust
            types[f"r{i}"] = "entity"
            per_query["dense"][f"r{i}"], per_query["bm25"][f"r{i}"] = 0.1, 0.9
        choice = cross_validated_router(per_query, types, min_type_samples=5)
        assert choice["r0"] == choice["r1"] == "dense"

    def test_needs_enough_labelled_queries(self):
        per_query = {
            "dense": {"q1": 0.5, "q2": 0.4, "q3": 0.3},
            "bm25": {"q1": 0.1, "q2": 0.2, "q3": 0.3},
        }
        with pytest.raises(ValueError, match="labelled queries"):
            cross_validated_router(per_query, {"q1": "a", "q2": "a", "q3": "a"}, folds=5)


class TestEvaluateRoutingPolicies:
    def _report(self):
        per_query, types = _type_structured()
        return evaluate_routing_policies(
            per_query,
            types,
            latency_ms={"dense": 10.0, "bm25": 5.0, "hybrid": 8.0, "hyde": 400.0},
            classifier_ms=300.0,
            n_boot=500,
        )

    def test_router_beats_every_static_policy_with_a_positive_ci(self):
        by_name = {p.name: p for p in self._report().policies}
        assert by_name["router_cv"].mean_ndcg == pytest.approx(0.8)
        statics = [p for name, p in by_name.items() if name.startswith("always_")]
        assert len(statics) == 4
        assert all(p.router_minus_policy.ci_low > 0 for p in statics)

    def test_oracle_is_an_upper_bound(self):
        by_name = {p.name: p for p in self._report().policies}
        assert by_name["oracle"].mean_ndcg >= by_name["router_cv"].mean_ndcg

    def test_cost_accounting_includes_classification(self):
        by_name = {p.name: p for p in self._report().policies}
        router = by_name["router_cv"]
        assert router.mean_latency_ms == pytest.approx(300.0 + 7.5)  # half dense 10 ms, half bm25 5 ms
        assert router.mean_llm_calls == pytest.approx(1.0)  # the classifier call only
        assert by_name["always_hyde"].mean_llm_calls == pytest.approx(1.0)
        assert by_name["always_hybrid"].mean_llm_calls == 0.0

    def test_exhaustive_reference_uses_measured_latency(self):
        report = self._report()
        assert report.exhaustive_llm_calls == 1
        assert report.exhaustive_latency_parallel_ms == pytest.approx(400.0)
        assert report.exhaustive_latency_sequential_ms == pytest.approx(423.0)

    def test_markdown_and_json_cover_every_policy(self):
        report = self._report()
        text = render_markdown(report, dataset="nfcorpus")
        for name in ("router_cv", "always_dense", "always_bm25", "oracle"):
            assert name in text
        json.dumps(asdict(report))


@pytest.mark.asyncio
async def test_router_eval_cli_writes_artifacts(tmp_path, monkeypatch):
    per_query, types = _type_structured(10)
    results = {
        s: MetricsResult({"ndcg@10": 0.5}, avg_latency_ms=10.0, per_query_ndcg10=scores)
        for s, scores in per_query.items()
    }

    async def fake_classify(items):
        return types, 250.0

    monkeypatch.setattr(rb, "_RESULTS_DIR", tmp_path)
    monkeypatch.setattr(rb, "_classify_queries", fake_classify)
    monkeypatch.setattr(rb, "get_settings", lambda: SimpleNamespace(anthropic_api_key="test-key"))
    monkeypatch.setattr(
        rb.BEIREvaluator,
        "_load_dataset",
        staticmethod(lambda dataset: ({}, {q: f"query {q}" for q in types}, {})),
    )

    ok = await rb._run_router_eval(SimpleNamespace(dataset="nfcorpus", max_queries=None), results)

    assert ok is True
    written = {p.name for p in tmp_path.iterdir()}
    assert {"nfcorpus_router_eval_latest.json", "nfcorpus_router_eval_latest.md"} <= written


@pytest.mark.asyncio
async def test_router_eval_cli_refuses_without_api_key(monkeypatch):
    monkeypatch.setattr(rb, "get_settings", lambda: SimpleNamespace(anthropic_api_key=""))
    results = {
        "dense": MetricsResult({}, per_query_ndcg10={"q": 1.0}),
        "bm25": MetricsResult({}, per_query_ndcg10={"q": 1.0}),
    }
    assert await rb._run_router_eval(SimpleNamespace(dataset="nfcorpus", max_queries=None), results) is False


@pytest.mark.asyncio
async def test_classifier_outage_aborts_instead_of_collapsing_to_one_type(monkeypatch):
    class _Down:
        def __init__(self, **_):
            pass

        async def classify(self, _text):
            return QueryType.UNKNOWN

    monkeypatch.setattr(rb, "QueryClassifier", _Down)
    monkeypatch.setattr(rb, "get_settings", lambda: SimpleNamespace(anthropic_api_key="x"))
    with pytest.raises(RuntimeError, match="unknown"):
        await rb._classify_queries([("q1", "a"), ("q2", "b")])
